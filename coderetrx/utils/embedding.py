from coderetrx.utils.path import get_cache_dir
from ._extras import require_extra

require_extra("langchain", "builtin-impl")

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
import requests
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, SecretStr
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from coderetrx.utils.concurrency import abatch_func_call, run_coroutine_sync
from langchain_chroma.vectorstores import Chroma
import chromadb
import logging
from abc import ABC, abstractmethod
from functools import wraps

# Qdrant imports (conditional import to handle optional dependency)
try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


logger = logging.getLogger(__name__)

cache_path = get_cache_dir()
embedder_cache_store = LocalFileStore(cache_path / "embeddings")

chromadb_client = chromadb.PersistentClient(path=str(cache_path / "chroma"))
langchain_chroma_client = Chroma(client=chromadb_client)

underlying_embeddings = OpenAIEmbeddings(
    model=os.environ["EMBEDDING_MODEL_ID"],
    base_url=os.environ["EMBEDDING_BASE_URL"],
    api_key=SecretStr(os.environ["EMBEDDING_API_KEY"]),
    openai_proxy=None,
)  # type:ignore

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, embedder_cache_store, namespace=underlying_embeddings.model
)


# Common retry decorator for similarity search operations
def similarity_search_retry(func):
    """Decorator for retry logic on similarity search operations."""
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=30, max=600),
        retry=retry_if_exception_type(Exception),
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


# Define a retry-decorated function for embedding documents
@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(
        min=30, max=600
    ),  # Exponential backoff with a minimum wait of 30s and max of 10min
    retry=retry_if_exception_type(Exception),  # Retry only if an exception is raised
)
async def embed_batch_with_retry(batch: List[str]) -> List[List[float]]:
    """Embed a batch of documents with retry logic."""
    try:
        logger.debug(f"Embedding batch of {len(batch)} documents")
        return await cached_embedder.aembed_documents(batch)
    except Exception as e:
        logger.warning(
            f"Embedding batch failed, will retry: {str(e)}, batch is {batch}"
        )
        raise  # Re-raise to trigger retry


def create_documents_embedding(
    docs: List[str], 
    batch_size: int = 100, 
    max_concurrency: int = 5
) -> List[List[float]]:
    """Create embeddings for a list of documents with batching and concurrency."""
    # Prepare batches
    kwargs_list = [
        {"batch": docs[i : i + batch_size]} for i in range(0, len(docs), batch_size)
    ]

    # Run the embedding process concurrently
    try:
        embeddings = run_coroutine_sync(
            abatch_func_call(
                max_concurrency=max_concurrency,
                func=embed_batch_with_retry,
                kwargs_list=kwargs_list,
            )
        )
        # Flatten the list of results (since each task returns a list of embeddings)
        result = [
            embedding
            for batch_embeddings in embeddings
            for embedding in batch_embeddings
        ]
        logger.debug(
            f"Successfully created embeddings for {len(docs)} documents with {max_concurrency} concurrent workers"
        )
        return result
    except Exception as e:
        logger.error(f"Failed to embed documents: {str(e)}", exc_info=True)
        raise  # Re-raise the exception for the caller to handle


def determine_content_type(name: str) -> str:
    """Determine content type based on collection name."""
    if "symbol_names" in name:
        return "symbol names"
    elif "symbol_contents" in name:
        return "symbol contents"
    elif "symbol_codelines" in name:
        return "code lines"
    elif "keywords" in name:
        return "keywords"
    else:
        return "documents"


def process_search_results(results: List[Tuple[Any, float]]) -> List[Tuple[str, float]]:
    """Process search results to extract page content and scores."""
    return [(result[0].page_content, result[1]) for result in results]


class SearchConfig(BaseModel):
    """Configuration for similarity search operations."""
    k: int = 10
    threshold: Optional[float] = None
    where: Optional[Dict[str, Any]] = None


class CollectionInfo(BaseModel):
    """Information about a vector database collection."""
    name: str
    exists: bool
    document_count: int
    needs_recreation: bool = False


class SimilaritySearcher(ABC):
    """Abstract base class for similarity search implementations."""

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        use_cache: bool = True,
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """
        Initialize the SimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            use_cache: Whether to use cached collection if available.
            metadatas: Optional metadata list corresponding to each text.
        """
        self.texts = texts
        self.name = name
        self.use_cache = use_cache
        self.metadatas = metadatas
        self.content_type = determine_content_type(name)
        
        # Initialize the vector store implementation
        self._initialize_vector_store(embeddings, use_cache)

    @abstractmethod
    def _initialize_vector_store(
        self, 
        embeddings: Optional[List[List[float]]], 
        use_cache: bool
    ) -> None:
        """Initialize the specific vector store implementation."""
        pass

    @abstractmethod
    def _get_vector_store_client(self) -> Any:
        """Get the vector store client for search operations."""
        pass

    def _validate_embeddings_match_texts(
        self, 
        texts: List[str], 
        embeddings: List[List[float]]
    ) -> None:
        """Validate that embeddings match texts count."""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

    def _validate_metadatas_match_texts(
        self, 
        texts: List[str], 
        metadatas: List[dict]
    ) -> None:
        """Validate that metadatas match texts count."""
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadatas")

    def _prepare_add_kwargs(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ) -> Dict[str, Any]:
        """Prepare keyword arguments for adding texts to vector store."""
        add_kwargs = {
            "texts": texts,
            "embeddings": embeddings,
            "ids": [str(start_idx + i) for i in range(len(texts))],
        }
        if metadatas:
            add_kwargs["metadatas"] = metadatas
        return add_kwargs

    def _add_texts_in_batches(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Add texts to vector store in batches with progress tracking."""
        logger.info(f"Adding {len(texts)} documents to {self.__class__.__name__} collection '{self.name}'")
        
        for idx in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Adding {self.content_type} to {self.__class__.__name__.replace('SimilaritySearcher', '')}",
        ):
            text_batch = texts[idx : idx + batch_size]
            metadata_batch = metadatas[idx : idx + batch_size] if metadatas else None
            
            if embeddings:
                embedding_batch = embeddings[idx : idx + batch_size]
                self._add_texts_with_embeddings(text_batch, embedding_batch, metadata_batch, idx)
            else:
                add_kwargs = self._prepare_add_kwargs(
                    text_batch, 
                    create_documents_embedding(text_batch),
                    metadata_batch,
                    idx
                )
                self._get_vector_store_client().add_texts(**add_kwargs)

    @abstractmethod
    def _add_texts_with_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ) -> None:
        """Add texts with precomputed embeddings to the collection."""
        pass

    @similarity_search_retry
    async def asearch(
        self,
        query: str,
        k: int = 10,
        threshold: Optional[float] = None,
        where: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query.

        Args:
            query: Query string.
            k: Number of top results to return.
            threshold: Minimum similarity threshold.
            where: Optional filter conditions.

        Returns:
            A list of tuples (document, normalized_score).
        """
        logger.info(f"Performing similarity search in collection '{self.name}' with k={k}")
        
        search_kwargs = {"query": query, "k": k}
        if threshold is not None:
            search_kwargs["score_threshold"] = threshold
        if where:
            logger.debug(f"Applying filter: {where}")
            search_kwargs["filter"] = where
            
        results = await self._get_vector_store_client().asimilarity_search_with_relevance_scores(**search_kwargs)
        return process_search_results(results)

    @similarity_search_retry
    async def asearch_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        threshold: Optional[float] = None,
        where: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query vector.

        Args:
            query_vector: Query vector.
            k: Number of top results to return.
            threshold: Minimum similarity threshold.
            where: Optional filter conditions.

        Returns:
            A list of tuples (document, normalized_score).
        """
        logger.debug(f"Performing vector similarity search in collection '{self.name}' with k={k}")
        
        from langchain_core.runnables.config import run_in_executor

        try:
            search_kwargs = {"embedding": query_vector, "k": k}
            if where:
                logger.debug(f"Applying filter: {where}")
                search_kwargs["filter"] = where
            
            # Note: similarity_search_by_vector_with_relevance_scores doesn't support score_threshold
            # We'll filter results after retrieval if threshold is provided
            results = await run_in_executor(
                None,
                self._get_vector_store_client().similarity_search_by_vector_with_relevance_scores,
                **search_kwargs
            )
            
            # Apply threshold filtering if specified
            if threshold is not None:
                results = [(doc, score) for doc, score in results if score >= threshold]
                
        except Exception as e:
            logger.error(f"Error during similarity search: {repr(e)}")
            raise

        return process_search_results(results)


class ChromaSimilaritySearcher(SimilaritySearcher):
    """ChromaDB-based implementation of SimilaritySearcher."""

    def _initialize_vector_store(
        self, 
        embeddings: Optional[List[List[float]]], 
        use_cache: bool
    ) -> None:
        """Initialize the ChromaDB vector store."""
        # Check or create Chroma collection
        try:
            collection = chromadb_client.get_collection(self.name)
            if collection.count() != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection.count()} documents instead of {len(self.texts)}. Recreating collection."
                )
                use_cache = False
                chromadb_client.delete_collection(self.name)
                collection = chromadb_client.create_collection(
                    self.name, metadata={"hnsw:space": "cosine"}
                )
            else:
                logger.info(
                    f"Using cached {self.content_type} from ChromaDB collection '{self.name}' ({collection.count()} items)"
                )
        except Exception:
            logger.info(f"Creating new ChromaDB collection: '{self.name}'")
            collection = chromadb_client.create_collection(
                self.name, metadata={"hnsw:space": "cosine"}
            )
            use_cache = False

        self.langchain_chroma_client = Chroma(
            client=chromadb_client,
            collection_name=self.name,
            embedding_function=cached_embedder,
            persist_directory=str(cache_path / "chroma"),
        )

        # Add texts and embeddings to collection if not using cache
        if not use_cache:
            self._add_texts_in_batches(self.texts, embeddings, self.metadatas)

    def _get_vector_store_client(self) -> Chroma:
        """Get the ChromaDB vector store client for search operations."""
        return self.langchain_chroma_client

    def _add_texts_with_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ) -> None:
        """Add texts with precomputed embeddings to the collection."""
        self._validate_embeddings_match_texts(texts, embeddings)
        if metadatas:
            self._validate_metadatas_match_texts(texts, metadatas)

        add_kwargs = self._prepare_add_kwargs(texts, embeddings, metadatas, start_idx)
        self.langchain_chroma_client.add_texts(**add_kwargs)


class QdrantSimilaritySearcher(SimilaritySearcher):
    """Qdrant-based implementation of SimilaritySearcher."""

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        use_cache: bool = True,
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """
        Initialize the QdrantSimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            use_cache: Whether to use cached collection if available.
            metadatas: Optional metadata list corresponding to each text.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant dependencies not available. Install with: pip install qdrant-client langchain-qdrant"
            )
        
        # Call parent constructor
        super().__init__(name, texts, embeddings, use_cache, metadatas)

    def _initialize_vector_store(
        self, 
        embeddings: Optional[List[List[float]]], 
        use_cache: bool
    ) -> None:
        """Initialize the Qdrant vector store."""
        # Initialize Qdrant client
        qdrant_path = cache_path / "qdrant"
        qdrant_path.mkdir(exist_ok=True)
        
        self.qdrant_client = QdrantClient(path=str(qdrant_path))
        
        # Check if collection exists and has correct number of documents
        collection_exists = False
        try:
            collection_info = self.qdrant_client.get_collection(self.name)
            if collection_info.points_count != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection_info.points_count} documents instead of {len(self.texts)}. Recreating collection."
                )
                use_cache = False
                self.qdrant_client.delete_collection(self.name)
            else:
                logger.info(
                    f"Using cached {self.content_type} from Qdrant collection '{self.name}' ({collection_info.points_count} items)"
                )
                collection_exists = True
        except Exception:
            logger.info(f"Creating new Qdrant collection: '{self.name}'")
            use_cache = False

        # Create collection if it doesn't exist or we're not using cache
        if not collection_exists or not use_cache:
            # Get embedding dimension from the first embedding or create a sample
            if embeddings and len(embeddings) > 0:
                vector_size = len(embeddings[0])
            else:
                # Create a sample embedding to get the dimension
                sample_embedding = create_documents_embedding([self.texts[0] if self.texts else "sample"])
                vector_size = len(sample_embedding[0])

            self.qdrant_client.create_collection(
                collection_name=self.name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        # Initialize LangChain Qdrant vector store
        self.langchain_qdrant_client = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.name,
            embedding=cached_embedder,
        )

        # Add texts and embeddings to collection if not using cache
        if not use_cache:
            self._add_texts_in_batches(self.texts, embeddings, self.metadatas)

    def _get_vector_store_client(self) -> QdrantVectorStore:
        """Get the Qdrant vector store client for search operations."""
        return self.langchain_qdrant_client

    def _add_texts_with_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ) -> None:
        """Add texts with precomputed embeddings to the collection."""
        self._validate_embeddings_match_texts(texts, embeddings)
        if metadatas:
            self._validate_metadatas_match_texts(texts, metadatas)

        add_kwargs = self._prepare_add_kwargs(texts, embeddings, metadatas, start_idx)
        self.langchain_qdrant_client.add_texts(**add_kwargs)


def get_similarity_searcher(
    provider: str,
    name: str,
    texts: List[str],
    embeddings: Optional[List[List[float]]] = None,
    use_cache: bool = True,
    metadatas: Optional[List[dict]] = None,
) -> SimilaritySearcher:
    """
    Factory function to create similarity searcher instances based on provider.

    Args:
        provider: The vector database provider to use (e.g., "chroma", "qdrant")
        name: Name of the collection.
        texts: List of texts to be indexed.
        embeddings: Optional precomputed embeddings corresponding to the texts.
        use_cache: Whether to use cached collection if available.
        metadatas: Optional metadata list corresponding to each text.

    Returns:
        A SimilaritySearcher instance based on the specified provider.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider.lower() == "chroma":
        return ChromaSimilaritySearcher(
            name=name,
            texts=texts,
            embeddings=embeddings,
            use_cache=use_cache,
            metadatas=metadatas,
        )
    elif provider.lower() == "qdrant":
        return QdrantSimilaritySearcher(
            name=name,
            texts=texts,
            embeddings=embeddings,
            use_cache=use_cache,
            metadatas=metadatas,
        )
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")
