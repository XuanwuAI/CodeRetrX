from httpx import AsyncClient
from pydantic_settings import BaseSettings
from coderetrx.utils.path import get_cache_dir
from coderetrx.utils.llm_cache import (
    get_llm_cache_provider,
    CacheSettings,
    SqliteCacheSettings,
)

import os
import asyncio
import json
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict, BaseSettings
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from coderetrx.utils.concurrency import abatch_func_call, run_coroutine_sync
from abc import ABC, abstractmethod
from functools import wraps
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
import httpx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Qdrant imports (conditional import to handle optional dependency)
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


logger = logging.getLogger(__name__)

cache_path = get_cache_dir()


class EmbeddingSettings(BaseSettings):
    """Settings for embedding configuration."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )

    model_id: str = Field(
        default="text-embedding-3-large",
        alias="EMBEDDING_MODEL_ID",
        description="Model ID for text embeddings",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="EMBEDDING_BASE_URL",
        description="Base URL for the embedding API",
    )
    api_key: str = Field(
        default="",
        alias="EMBEDDING_API_KEY",
        description="API key for the embedding service",
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for embedding requests",
    )
    max_concurrency: int = Field(
        default=25,
        description="Maximum number of concurrent requests for embedding",
    )
    max_trunc_chars: int = Field(
        default=8000,
        description="Max. characters per document to embed. Set to -1 to disable truncation.",
    )
    proxy: Optional[str] = Field(
        default=None,
        description="Proxy URL for HTTP requests, support socks5 and http(s) proxies",
        alias="EMBEDDING_PROXY",
    )
    is_local_embedding: bool = Field(
        default=True,
        description="Whether to use local embedding model instead of API",
        alias="IS_LOCAL_EMBEDDING",
    )

    def get_httpx_client(self) -> AsyncClient:
        if self.proxy:
            return AsyncClient(proxy=self.proxy)
        return AsyncClient()


# Global embedding settings
embedding_settings = EmbeddingSettings()


def get_embedding_settings() -> EmbeddingSettings:
    """
    Get the global embedding settings.

    Returns:
        EmbeddingSettings instance with current configuration.
    """
    return embedding_settings


def set_embedding_settings(settings: EmbeddingSettings) -> None:
    """
    Set the global embedding settings.

    Args:
        settings: EmbeddingSettings instance with new configuration.
    """
    global embedding_settings
    embedding_settings = settings
    logger.info(f"Embedding settings updated: {embedding_settings.model_dump()}")


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to a maximum number of characters.
    
    Args:
        text: The input text to truncate.
        max_chars: Maximum number of characters to keep. Use -1 to disable truncation.
        
    Returns:
        Truncated text or original text if max_chars is -1.
    """
    if max_chars == -1:
        return text
    return text[:max_chars]


async def _create_embeddings_with_cache(
    texts: List[str], settings: Optional[EmbeddingSettings] = None
) -> List[List[float]]:
    """
    Create embeddings for texts using OpenAI API or local model with individual text caching.

    Args:
        texts: List of texts to embed
        settings: Optional embedding settings, defaults to global settings

    Returns:
        List of embedding vectors
    """
    if settings is None:
        settings = embedding_settings
    
    # Truncate texts based on max_trunc_chars setting
    truncated_texts = [truncate_text(text, settings.max_trunc_chars) for text in texts]
    
    cache_settings = SqliteCacheSettings(
        SQLITE_DB_PATH=str(get_cache_dir() / "embedding" / "cache.db"),
    )

    cache_provider = get_llm_cache_provider(settings=cache_settings)

    # Check cache for each text individually
    cached_embeddings: List[Optional[List[float]]] = []
    uncached_texts: List[str] = []
    uncached_indices: List[int] = []

    for i, text in enumerate(truncated_texts):
        # Create cache key for individual text
        individual_request = {
            "model": settings.model_id,
            "input": text,
        }
        cache_key = cache_provider.hash_params(individual_request)
        cached_response = cache_provider.get(cache_key)

        if cached_response:
            # Cache hit for this text
            response_data = json.loads(cached_response)
            embedding = response_data["embedding"]
            cached_embeddings.append(embedding)
        else:
            # Cache miss for this text
            cached_embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)

    cache_hits = len(truncated_texts) - len(uncached_texts)
    if cache_hits > 0:
        logger.debug(f"Cache hit for {cache_hits}/{len(truncated_texts)} texts")

    # If all texts are cached, return cached results
    if not uncached_texts:
        result = [embedding for embedding in cached_embeddings if embedding is not None]
        assert len(result) == len(truncated_texts), "Cached embeddings count mismatch"
        return result

    # Generate embeddings for uncached texts
    if settings.is_local_embedding:
        # Use local model
        logger.debug(f"Using local model for {len(uncached_texts)} uncached texts")
        model = SentenceTransformer(settings.model_id)
        embeddings = await asyncio.to_thread(model.encode, uncached_texts)
        
        # Cache each individual text embedding
        for i, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
            individual_request = {
                "model": settings.model_id,
                "input": text,
            }
            cache_key = cache_provider.hash_params(individual_request)

            # Store individual embedding in cache
            individual_response = {"embedding": embedding.tolist()}
            cache_provider.insert(cache_key, individual_request, individual_response)

        # Combine cached and new embeddings in correct order
        result_embeddings: List[List[float]] = []
        uncached_iter = iter(embeddings)

        for i, cached_embedding in enumerate(cached_embeddings):
            if cached_embedding is not None:
                # Use cached embedding
                result_embeddings.append(cached_embedding)
            else:
                # Use newly computed embedding
                new_embedding = next(uncached_iter)
                result_embeddings.append(new_embedding.tolist())

        return result_embeddings
    
    else:
        # Make API call for uncached texts
        logger.debug(f"Making API call for {len(uncached_texts)} uncached texts")

        httpx_client = settings.get_httpx_client()
        client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=300.0,
            max_retries=0,
            http_client=httpx_client,
        )

        try:
            # Create request for uncached texts
            api_request = {
                "model": settings.model_id,
                "input": uncached_texts,
            }

            response: CreateEmbeddingResponse = await client.embeddings.create(
                **api_request
            )

            # Cache each individual text embedding
            for i, (text, embedding_data) in enumerate(zip(uncached_texts, response.data)):
                individual_request = {
                    "model": settings.model_id,
                    "input": text,
                }
                cache_key = cache_provider.hash_params(individual_request)

                # Store individual embedding in cache
                individual_response = {"embedding": embedding_data.embedding}
                cache_provider.insert(cache_key, individual_request, individual_response)

            # Combine cached and new embeddings in correct order
            result_embeddings: List[List[float]] = []
            uncached_iter = iter(response.data)

            for i, cached_embedding in enumerate(cached_embeddings):
                if cached_embedding is not None:
                    # Use cached embedding
                    result_embeddings.append(cached_embedding)
                else:
                    # Use newly computed embedding
                    new_embedding = next(uncached_iter)
                    result_embeddings.append(new_embedding.embedding)

            return result_embeddings

        finally:
            try:
                await client.close()
            except Exception as close_error:
                logger.warning(f"Error closing embedding client: {close_error}")


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
        return await _create_embeddings_with_cache(batch)
    except Exception as e:
        logger.warning(
            f"Embedding batch failed, will retry: {str(e)}, batch is {batch}"
        )
        raise  # Re-raise to trigger retry


def create_documents_embedding(
    docs: List[str], batch_size: int = 100, max_concurrency: int = 5
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
        metadatas: Optional[List[dict]] = None,
        vector_db_mode: str = "reuse_on_match",
    ) -> None:
        """
        Initialize the SimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            metadatas: Optional metadata list corresponding to each text.
            vector_db_mode: Vector DB reuse mode ("always_reuse", "never_reuse", "reuse_on_match")
        """
        self.texts = texts
        self.name = name
        self.vector_db_mode = vector_db_mode
        self.metadatas = metadatas

    @abstractmethod
    async def asearch_with_score(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        where: Optional[dict] = None,
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
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
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

    def search_with_score(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        where: Optional[dict] = None,
    ):
        """Synchronous wrapper for asearch_with_score."""
        from coderetrx.utils.concurrency import run_coroutine_sync

        return run_coroutine_sync(self.asearch_with_score(query, k, threshold, where))

    def search_by_vector(
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
    ):
        """Synchronous wrapper for asearch_by_vector."""
        from coderetrx.utils.concurrency import run_coroutine_sync

        return run_coroutine_sync(self.asearch_by_vector(query_vector, k, where))


class ChromaSimilaritySearcher(SimilaritySearcher):
    """ChromaDB-based implementation of SimilaritySearcher without langchain dependencies."""

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        vector_db_mode: str = "reuse_on_match",
    ) -> None:
        super().__init__(name, texts, embeddings, metadatas, vector_db_mode)
        self.content_type = determine_content_type(name)
        self._initialize_vector_store(embeddings)

    def _initialize_vector_store(
        self, embeddings: Optional[List[List[float]]]
    ) -> None:
        """Initialize the ChromaDB vector store."""
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB dependencies not available. Install with: pip install chromadb"
            )
        # Initialize ChromaDB client
        self.chromadb_client = chromadb.PersistentClient(
            path=str(cache_path / "chroma")
        )

        # Check or create Chroma collection
        collection_exists = False
        collection_count = 0
        
        try:
            self.collection = self.chromadb_client.get_collection(name=self.name)
            collection_count = self.collection.count()
            collection_exists = True
            
            if self.vector_db_mode == "always_reuse":
                logger.debug(
                    f"Using cached {self.content_type} from ChromaDB collection '{self.name}' ({collection_count} items) - always_reuse mode"
                )
            elif self.vector_db_mode == "never_reuse":
                logger.info(
                    f"Recreating ChromaDB collection '{self.name}' - never_reuse mode"
                )
                self.chromadb_client.delete_collection(self.name)
                self.collection = self.chromadb_client.create_collection(
                    name=self.name, metadata={"hnsw:space": "cosine", "hnsw:M": 1024}
                )
                collection_exists = False
            elif collection_count != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection_count} documents instead of {len(self.texts)}. Recreating collection."
                )
                self.chromadb_client.delete_collection(self.name)
                self.collection = self.chromadb_client.create_collection(
                    name=self.name, metadata={"hnsw:space": "cosine", "hnsw:M": 1024}
                )
                collection_exists = False
            else:
                logger.debug(
                    f"Using cached {self.content_type} from ChromaDB collection '{self.name}' ({collection_count} items)"
                )
        except Exception:
            logger.info(f"Creating new ChromaDB collection: '{self.name}'")
            self.collection = self.chromadb_client.create_collection(
                name=self.name, metadata={"hnsw:space": "cosine", "hnsw:M": 1024}
            )
            collection_exists = False

        # Add texts and embeddings to collection based on vector_db_mode
        should_add_texts = (
            self.vector_db_mode == "never_reuse" or
            not collection_exists or
            (self.vector_db_mode == "reuse_on_match" and collection_count != len(self.texts))
        )
        
        if should_add_texts:
            self._add_texts_in_batches(self.texts, embeddings, self.metadatas)

    def _validate_embeddings_match_texts(
        self, texts: List[str], embeddings: List[List[float]]
    ) -> None:
        """Validate that embeddings match texts count."""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

    def _validate_metadatas_match_texts(
        self, texts: List[str], metadatas: List[dict]
    ) -> None:
        """Validate that metadatas match texts count."""
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadatas")

    def _add_texts_in_batches(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Add texts to vector store in batches with progress tracking."""
        logger.info(
            f"Adding {len(texts)} documents to {self.__class__.__name__} collection '{self.name}'"
        )

        for idx in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Adding {self.content_type} to {self.__class__.__name__.replace('SimilaritySearcher', '')}",
        ):
            text_batch = texts[idx : idx + batch_size]
            metadata_batch = metadatas[idx : idx + batch_size] if metadatas else None

            if embeddings:
                embedding_batch = embeddings[idx : idx + batch_size]
                self._add_texts_with_embeddings(
                    text_batch, embedding_batch, metadata_batch, idx
                )
            else:
                embedding_batch = create_documents_embedding(text_batch)
                self._add_texts_with_embeddings(
                    text_batch, embedding_batch, metadata_batch, idx
                )

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

        # Prepare data for ChromaDB
        ids = [str(start_idx + i) for i in range(len(texts))]
        documents = texts

        # Add to ChromaDB collection
        if metadatas:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,  # type: ignore
                metadatas=metadatas,  # type: ignore
            )
        else:
            self.collection.add(
                ids=ids, documents=documents, embeddings=embeddings  # type: ignore
            )

    @similarity_search_retry
    async def asearch_with_score(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        where: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query.
        """
        logger.debug(
            f"Performing similarity search in collection '{self.name}' with k={k}"
        )

        # Get query embedding
        try:
            query_embedding = await _create_embeddings_with_cache([query])
            query_vector = query_embedding[0]
        except Exception as e:
            logger.error(f"Error creating embedding for query '{query}': {repr(e)}")
            return []
        # Perform search using ChromaDB
        return await self.asearch_by_vector(query_vector=query_vector, k=k, where=where)

    @similarity_search_retry
    async def asearch_by_vector(
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query vector.
        """
        logger.debug(
            f"Performing vector similarity search in collection '{self.name}' with k={k}"
        )

        try:
            # Perform search using ChromaDB
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_vector],
                n_results=k,
                where=where,
                include=["documents", "distances"],
            )

            # Convert ChromaDB results to our format
            search_results = []
            if results and results.get("documents") and results.get("distances"):
                documents = results["documents"][0] if results["documents"] else []
                distances = results["distances"][0] if results["distances"] else []

                for doc, distance in zip(documents, distances):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance
                    search_results.append((doc, score))

            return search_results

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
        metadatas: Optional[List[dict]] = None,
        vector_db_mode: str = "reuse_on_match",
        vector_size: int = 3072,
    ) -> None:
        """
        Initialize the QdrantSimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            metadatas: Optional metadata list corresponding to each text.
            vector_db_mode: Vector DB reuse mode ("always_reuse", "never_reuse", "reuse_on_match")
            vector_size: Size of the vector embeddings.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant dependencies not available. Install with: pip install qdrant-client"
            )

        super().__init__(name, texts, embeddings, metadatas, vector_db_mode)
        self.vector_size = vector_size
        self.content_type = determine_content_type(name)

        # Initialize Qdrant clients
        self._init_clients()

        # Initialize collection
        self._init_collection()

        # Add texts based on vector_db_mode
        should_add_texts = (
            self.vector_db_mode == "never_reuse" or 
            (self.vector_db_mode == "reuse_on_match" and self._get_collection_count() != len(texts))
        )
        if should_add_texts:
            self._add_texts_to_collection(embeddings)

    def _init_clients(self):
        """Initialize Qdrant clients with connection pooling."""
        QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
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

            if self.vector_db_mode == "always_reuse":
                logger.debug(
                    f"Using cached {self.content_type} from Qdrant collection '{self.name}' ({collection_count} items) - always_reuse mode"
                )
            elif self.vector_db_mode == "never_reuse":
                logger.info(
                    f"Recreating Qdrant collection '{self.name}' - never_reuse mode"
                )
                self.qdrant_client.delete_collection(self.name)
                self._create_collection()
            elif collection_count != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection_count} documents instead of {len(self.texts)}. Recreating collection."
                )
                self.qdrant_client.delete_collection(self.name)
                self._create_collection()
            else:
                logger.info(
                    f"Using cached {self.content_type} from Qdrant collection '{self.name}' ({collection_count} items)"
                )

        except Exception:
            logger.info(f"Creating new Qdrant collection: '{self.name}'")
            self._create_collection()

    def _create_collection(self):
        """Create a new Qdrant collection."""
        self.qdrant_client.create_collection(
            collection_name=self.name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )

    def _get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.qdrant_client.count(collection_name=self.name).count
        except:
            return 0

    def _add_texts_to_collection(self, embeddings: Optional[List[List[float]]] = None):
        """Add texts and embeddings to the Qdrant collection."""
        logger.info(
            f"Adding {len(self.texts)} documents to Qdrant collection '{self.name}'"
        )

        # Calculate adaptive batch size based on content
        avg_text_length = sum(len(text) for text in self.texts[:100]) / min(
            100, len(self.texts)
        )
        estimated_point_size = avg_text_length + self.vector_size * 4 + 1000
        max_payload_size = 30 * 1024 * 1024
        batch_size = max(1, min(300, int(max_payload_size // estimated_point_size)))

        logger.info(
            f"Using adaptive batch size: {batch_size} (avg text length: {avg_text_length:.0f} chars)"
        )

        total_batches = (len(self.texts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(total_batches), desc=f"Adding {self.content_type} to Qdrant"
        ):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.texts))

            text_batch = self.texts[start_idx:end_idx]
            metadata_batch = (
                self.metadatas[start_idx:end_idx] if self.metadatas else None
            )

            if embeddings:
                embedding_batch = embeddings[start_idx:end_idx]
            else:
                embedding_batch = create_documents_embedding(text_batch)

            self._add_batch_to_qdrant(
                text_batch, embedding_batch, metadata_batch, start_idx
            )
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def _add_batch_to_qdrant(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ):
        """Add a batch of texts with embeddings to Qdrant."""
        points = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {"content": text}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            points.append(
                PointStruct(id=start_idx + i, vector=embedding, payload=payload)
            )

        try:
            self.qdrant_client.upsert(collection_name=self.name, points=points)
        except Exception as e:
            if "larger than allowed" in str(e) and len(points) > 1:
                logger.warning(
                    f"Batch too large ({len(points)} points), splitting into smaller batches"
                )
                mid = len(points) // 2

                first_texts = texts[:mid]
                first_embeddings = embeddings[:mid]
                first_metadatas = metadatas[:mid] if metadatas else None
                self._add_batch_to_qdrant(
                    first_texts, first_embeddings, first_metadatas, start_idx
                )

                second_texts = texts[mid:]
                second_embeddings = embeddings[mid:]
                second_metadatas = metadatas[mid:] if metadatas else None
                self._add_batch_to_qdrant(
                    second_texts, second_embeddings, second_metadatas, start_idx + mid
                )
            else:
                logger.error(f"Failed to add batch to Qdrant: {str(e)}")
                raise

    @similarity_search_retry
    async def asearch_with_score(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        where: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query.
        """
        logger.info(
            f"Performing similarity search in collection '{self.name}' with k={k}"
        )

        try:
            # Get query embedding
            query_embedding = await _create_embeddings_with_cache([query])
            query_vector = query_embedding[0]

            # Build filter if provided
            query_filter = None
            if where:
                logger.debug(f"Applying filter: {where}")
                query_filter = self._build_filter(where)

            # Perform search
            search_result = await self.async_qdrant_client.search(
                collection_name=self.name,
                query_vector=query_vector,
                limit=k,
                score_threshold=threshold,
                query_filter=query_filter,
            )

            # Format results
            results = []
            for point in search_result:
                content = point.payload.get("content", "") if point.payload else ""
                score = point.score
                results.append((content, score))

            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {repr(e)}")
            # Return empty results on connection failure instead of crashing
            if "ConnectError" in str(e) or "ConnectTimeout" in str(e):
                logger.warning(
                    f"Connection error in search, returning empty results: {e}"
                )
                return []
            raise

    @similarity_search_retry
    async def asearch_by_vector(
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents to the query vector.
        """
        logger.debug(
            f"Performing vector similarity search in collection '{self.name}' with k={k}"
        )

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
                query_filter=query_filter,
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
                logger.warning(
                    f"Connection error in search, returning empty results: {e}"
                )
                return []
            raise

    def _build_filter(self, where: dict) -> Filter:
        """Build Qdrant filter from dictionary conditions."""
        conditions = []

        for key, value in where.items():
            if isinstance(value, (str, int, bool)):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None


def get_embedding_dimension() -> int:
    """Get the dimension of the current embedding model."""
    test_embedding = run_coroutine_sync(_create_embeddings_with_cache(["test"]))
    return len(test_embedding[0])


def get_similarity_searcher(
    provider: str,
    name: str,
    texts: List[str],
    embeddings: Optional[List[List[float]]] = None,
    metadatas: Optional[List[dict]] = None,
    vector_db_mode: str = "reuse_on_match",
) -> SimilaritySearcher:
    """
    Factory function to create similarity searcher instances based on provider.

    Args:
        provider: The vector database provider to use (e.g., "chroma", "qdrant")
        name: Name of the collection.
        texts: List of texts to be indexed.
        embeddings: Optional precomputed embeddings corresponding to the texts.
        metadatas: Optional metadata list corresponding to each text.
        vector_db_mode: Vector DB reuse mode ("always_reuse", "never_reuse", "reuse_on_match")

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
            metadatas=metadatas,
            vector_db_mode=vector_db_mode,
        )
    elif provider.lower() == "qdrant":
        # Get the actual embedding dimension
        vector_size = get_embedding_dimension()
        # Include dimension in collection name to avoid conflicts
        dimension_aware_name = f"{name}_{vector_size}d"
        return QdrantSimilaritySearcher(
            name=dimension_aware_name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            vector_db_mode=vector_db_mode,
            vector_size=vector_size,
        )
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")
