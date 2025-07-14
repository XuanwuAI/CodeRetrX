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
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    import httpx
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

    @abstractmethod
    async def asearch_with_score(
        self, 
        query: str, 
        k: int = 10, 
        threshold: float = 0.0,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query.

        Args:
            query: Query string.
            k: Number of top results to return.
            threshold: Minimum score threshold.
            where: Filter conditions.

        Returns:
            A list of tuples (document, score).
        """
        pass

    @abstractmethod
    async def asearch_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query vector.

        Args:
            query_vector: Query vector.
            k: Number of top results to return.
            where: Filter conditions.

        Returns:
            A list of tuples (document, score).
        """
        pass

    def search_with_score(self, query: str, k: int = 10, threshold: float = 0.0, where: Optional[dict] = None):
        """Synchronous wrapper for asearch_with_score."""
        from coderetrx.utils.concurrency import run_coroutine_sync
        return run_coroutine_sync(self.asearch_with_score(query, k, threshold, where))

    def search_by_vector(self, query_vector: List[float], k: int = 10, where: Optional[dict] = None):
        """Synchronous wrapper for asearch_by_vector."""
        from coderetrx.utils.concurrency import run_coroutine_sync
        return run_coroutine_sync(self.asearch_by_vector(query_vector, k, where))



class ChromaSimilaritySearcher(SimilaritySearcher):
    """ChromaDB-based implementation of SimilaritySearcher."""

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        use_cache: bool = True,
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        super().__init__(name, texts, embeddings, use_cache, metadatas)
        self.content_type = determine_content_type(name)
        self._initialize_vector_store(embeddings, use_cache)

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
                self.langchain_chroma_client.add_texts(**add_kwargs)

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

    @similarity_search_retry
    async def asearch_with_score(
        self, 
        query: str, 
        k: int = 10, 
        threshold: float = 0.0,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query.
        """
        logger.info(f"Performing similarity search in collection '{self.name}' with k={k}")
        
        try:
            # Use Chroma's similarity search with score
            results = await asyncio.to_thread(
                self.langchain_chroma_client.similarity_search_with_score,
                query, k=k, filter=where
            )
            
            # Filter by threshold and convert to our format
            filtered_results = []
            for doc, score in results:
                if score >= threshold:
                    filtered_results.append((doc.page_content, score))
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {repr(e)}")
            return []

    @similarity_search_retry
    async def asearch_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query vector.
        """
        logger.debug(f"Performing vector similarity search in collection '{self.name}' with k={k}")
        
        try:
            # Use Chroma's similarity search by vector
            results = await asyncio.to_thread(
                getattr(self.langchain_chroma_client, 'similarity_search_by_vector_with_score', None) or self.langchain_chroma_client.similarity_search_by_vector,
                query_vector, k=k, filter=where
            )
            
            # Convert to our format
            if hasattr(self.langchain_chroma_client, 'similarity_search_by_vector_with_score') and results and len(results[0]) == 2:
                return [(doc.page_content, score) for doc, score in results]
            else:
                # Fallback when scores aren't available
                return [(doc.page_content, 1.0) for doc in results]
            
        except Exception as e:
            logger.error(f"Error during vector similarity search: {repr(e)}")
            return []


class QdrantSimilaritySearcher(SimilaritySearcher):
    """High-performance Qdrant-based implementation using direct Qdrant API."""

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        use_cache: bool = True,
        metadatas: Optional[List[dict]] = None,
        vector_size: int = 3072,
    ) -> None:
        """
        Initialize the QdrantSimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            use_cache: Whether to use cached collection if available.
            metadatas: Optional metadata list corresponding to each text.
            vector_size: Size of the vector embeddings.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant dependencies not available. Install with: pip install qdrant-client"
            )
        
        super().__init__(name, texts, embeddings, use_cache, metadatas)
        self.vector_size = vector_size
        
        # Determine content type
        if "symbol_names" in name:
            content_type = "symbol names"
        elif "symbol_contents" in name:
            content_type = "symbol contents"
        elif "symbol_codelines" in name:
            content_type = "code lines"
        elif "keywords" in name:
            content_type = "keywords"
        else:
            content_type = "documents"

        self.content_type = content_type
        
        # Initialize Qdrant clients
        self._init_clients()
        
        # Initialize collection
        self._init_collection()
        
        # Add texts if not using cache or collection is empty
        if not use_cache or self._get_collection_count() != len(texts):
            self._add_texts_to_collection(embeddings)

    def _init_clients(self):
        """Initialize Qdrant clients with connection pooling."""
        QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
        
        connection_limits = httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=30,
            limits=connection_limits,
        )
        
        self.async_qdrant_client = AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=30,
            limits=connection_limits,
        )

    def _init_collection(self):
        """Initialize or create Qdrant collection."""
        try:
            collection_count = self.qdrant_client.count(collection_name=self.name).count
            
            if collection_count != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection_count} documents instead of {len(self.texts)}. Recreating collection."
                )
                self.qdrant_client.delete_collection(self.name)
                self._create_collection()
            else:
                logger.info(f"Using cached {self.content_type} from Qdrant collection '{self.name}' ({collection_count} items)")
                
        except Exception:
            logger.info(f"Creating new Qdrant collection: '{self.name}'")
            self._create_collection()

    def _create_collection(self):
        """Create a new Qdrant collection."""
        self.qdrant_client.create_collection(
            collection_name=self.name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )

    def _get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.qdrant_client.count(collection_name=self.name).count
        except:
            return 0

    def _add_texts_to_collection(self, embeddings: Optional[List[List[float]]] = None):
        """Add texts and embeddings to the Qdrant collection."""
        logger.info(f"Adding {len(self.texts)} documents to Qdrant collection '{self.name}'")
        
        # Calculate adaptive batch size based on content
        avg_text_length = sum(len(text) for text in self.texts[:100]) / min(100, len(self.texts))
        estimated_point_size = avg_text_length + self.vector_size * 4 + 1000
        max_payload_size = 30 * 1024 * 1024
        batch_size = max(1, min(300, int(max_payload_size // estimated_point_size)))
        
        logger.info(f"Using adaptive batch size: {batch_size} (avg text length: {avg_text_length:.0f} chars)")
        
        total_batches = (len(self.texts) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc=f"Adding {self.content_type} to Qdrant"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.texts))
            
            text_batch = self.texts[start_idx:end_idx]
            metadata_batch = self.metadatas[start_idx:end_idx] if self.metadatas else None
            
            if embeddings:
                embedding_batch = embeddings[start_idx:end_idx]
            else:
                embedding_batch = create_documents_embedding(text_batch)
            
            self._add_batch_to_qdrant(text_batch, embedding_batch, metadata_batch, start_idx)

    def _add_batch_to_qdrant(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0
    ):
        """Add a batch of texts with embeddings to Qdrant."""
        points = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {"content": text}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            
            points.append(
                PointStruct(
                    id=start_idx + i,
                    vector=embedding,
                    payload=payload
                )
            )
        
        try:
            self.qdrant_client.upsert(
                collection_name=self.name,
                points=points
            )
        except Exception as e:
            if "larger than allowed" in str(e) and len(points) > 1:
                logger.warning(f"Batch too large ({len(points)} points), splitting into smaller batches")
                mid = len(points) // 2
                
                first_texts = texts[:mid]
                first_embeddings = embeddings[:mid] 
                first_metadatas = metadatas[:mid] if metadatas else None
                self._add_batch_to_qdrant(first_texts, first_embeddings, first_metadatas, start_idx)
                
                second_texts = texts[mid:]
                second_embeddings = embeddings[mid:]
                second_metadatas = metadatas[mid:] if metadatas else None
                self._add_batch_to_qdrant(second_texts, second_embeddings, second_metadatas, start_idx + mid)
            else:
                raise

    @similarity_search_retry
    async def asearch_with_score(
        self, 
        query: str, 
        k: int = 10, 
        threshold: float = 0.0,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query.
        """
        logger.info(f"Performing similarity search in collection '{self.name}' with k={k}")
        
        try:
            # Get query embedding
            query_embedding = await cached_embedder.aembed_query(query)
            
            # Build filter if provided
            query_filter = None
            if where:
                logger.debug(f"Applying filter: {where}")
                query_filter = self._build_filter(where)
            
            # Perform search
            search_result = await self.async_qdrant_client.search(
                collection_name=self.name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=threshold,
                query_filter=query_filter
            )
            
            # Format results
            results = []
            for point in search_result:
                content = point.payload.get("content", "")
                score = point.score
                results.append((content, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {repr(e)}")
            # Return empty results on connection failure instead of crashing
            if "ConnectError" in str(e) or "ConnectTimeout" in str(e):
                logger.warning(f"Connection error in search, returning empty results: {e}")
                return []
            raise

    @similarity_search_retry
    async def asearch_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query vector.
        """
        logger.debug(f"Performing vector similarity search in collection '{self.name}' with k={k}")
        
        # Build filter if provided
        query_filter = None
        if where:
            logger.debug(f"Applying filter: {where}")
            query_filter = self._build_filter(where)
        
        try:
            search_result = await self.async_qdrant_client.search(
                collection_name=self.name,
                query_vector=query_vector,
                limit=k,
                query_filter=query_filter
            )
            
            # Extract content and scores from results
            results = []
            for point in search_result:
                content = point.payload.get("content", "")
                score = point.score
                results.append((content, score))
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {repr(e)}")
            # Return empty results on connection failure instead of crashing
            if "ConnectError" in str(e) or "ConnectTimeout" in str(e):
                logger.warning(f"Connection error in search, returning empty results: {e}")
                return []
            raise

    def _initialize_vector_store(
        self, 
        embeddings: Optional[List[List[float]]], 
        use_cache: bool
    ) -> None:
        """Initialize the Qdrant vector store."""
        # This method is called by the parent constructor, but we handle initialization
        # differently in QdrantSimilaritySearcher, so we do nothing here.
        pass

    def _get_vector_store_client(self) -> None:
        """Get the vector store client for search operations."""
        # QdrantSimilaritySearcher uses direct Qdrant client calls instead of langchain
        # This method is not used by QdrantSimilaritySearcher
        return None

    def _add_texts_with_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ) -> None:
        """Add texts with precomputed embeddings to the collection."""
        # Delegate to our custom Qdrant implementation
        self._add_batch_to_qdrant(texts, embeddings, metadatas, start_idx)

    def _build_filter(self, where: dict) -> Filter:
        """Build Qdrant filter from dictionary conditions."""
        conditions = []
        
        for key, value in where.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None


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
