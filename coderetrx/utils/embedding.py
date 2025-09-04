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
import logging
from pathlib import Path
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

load_dotenv()


logger = logging.getLogger(__name__)

cache_path = get_cache_dir()


class EmbeddingSettings(BaseSettings):
    """Configuration settings for text embedding operations.

    This class manages all configuration parameters for embedding generation,
    including API settings, performance tuning, and caching behavior.
    Settings can be loaded from environment variables or .env files.

    Attributes:
        model_id: The embedding model identifier (e.g., 'text-embedding-3-large').
        base_url: Base URL for the embedding API endpoint.
        api_key: Authentication key for the embedding service.
        batch_size: Number of texts to process in a single API request.
        max_concurrency: Maximum number of concurrent embedding requests.
        max_trunc_chars: Maximum characters per text before truncation (-1 to disable).
        proxy: Optional proxy URL for HTTP requests (supports socks5 and http(s)).
        embedding_dimension: Dimensionality of the generated embedding vectors.

    Example:
        >>> settings = EmbeddingSettings(
        ...     model_id="text-embedding-3-small",
        ...     batch_size=50,
        ...     max_concurrency=10
        ... )
        >>> set_embedding_settings(settings)
    """

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
    embedding_dimension: int = Field(
        default=1024,
        description="Dimensionality of the embedding vectors",
        alias="EMBEDDING_DIMENSION",
    )

    def get_httpx_client(self) -> AsyncClient:
        """Create an async HTTP client with optional proxy configuration.

        Returns:
            AsyncClient: Configured HTTP client for making API requests.

        Example:
            >>> settings = EmbeddingSettings(proxy="http://proxy.example.com:8080")
            >>> client = settings.get_httpx_client()
        """
        if self.proxy:
            return AsyncClient(proxy=self.proxy)
        return AsyncClient()


# Global embedding settings
embedding_settings = EmbeddingSettings()


def get_embedding_settings() -> EmbeddingSettings:
    """Get the global embedding settings instance.

    Returns:
        EmbeddingSettings: The current global embedding configuration.

    Example:
        >>> settings = get_embedding_settings()
        >>> print(settings.model_id)
        'text-embedding-3-large'
    """
    return embedding_settings


def set_embedding_settings(settings: EmbeddingSettings) -> None:
    """Set the global embedding settings.

    Args:
        settings: EmbeddingSettings instance with new configuration.

    Example:
        >>> new_settings = EmbeddingSettings(batch_size=50)
        >>> set_embedding_settings(new_settings)
    """
    global embedding_settings
    embedding_settings = settings
    logger.info(f"Embedding settings updated: {embedding_settings.model_dump()}")


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to a maximum number of characters.

    Args:
        text: The input text to truncate.
        max_chars: Maximum number of characters to keep. Use -1 to disable truncation.

    Returns:
        str: Truncated text or original text if max_chars is -1.

    Example:
        >>> truncate_text("Hello world", 5)
        'Hello'
        >>> truncate_text("Hello world", -1)
        'Hello world'
    """
    if max_chars == -1:
        return text
    return text[:max_chars]


async def create_embeddings_with_cache(
    texts: List[str], settings: Optional[EmbeddingSettings] = None
) -> List[List[float]]:
    """Create embeddings for texts using OpenAI API with individual text caching.

    This function implements intelligent caching at the individual text level,
    checking cache for each text separately and only making API calls for
    uncached texts. This maximizes cache efficiency and minimizes API costs.

    Args:
        texts: List of texts to embed.
        settings: Optional embedding settings, defaults to global settings.

    Returns:
        List[List[float]]: List of embedding vectors corresponding to input texts.

    Raises:
        Exception: If API call fails after retries or if there are authentication issues.

    Example:
        >>> texts = ["Hello world", "How are you?"]
        >>> embeddings = await _create_embeddings_with_cache(texts)
        >>> len(embeddings)
        2
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
            import json

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


def similarity_search_retry(func):
    """Decorator for retry logic on similarity search operations.

    Applies exponential backoff retry strategy to handle transient failures
    in similarity search operations, such as network timeouts or temporary
    database unavailability.

    Args:
        func: The async function to decorate with retry logic.

    Returns:
        Decorated function with retry capabilities.

    Example:
        >>> @similarity_search_retry
        ... async def search_function():
        ...     # Function that might fail transiently
        ...     pass
    """

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=30, max=600),
        retry=retry_if_exception_type(Exception),
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(
        min=30, max=600
    ),  # Exponential backoff with a minimum wait of 30s and max of 10min
    retry=retry_if_exception_type(Exception),  # Retry only if an exception is raised
)
async def embed_batch_with_retry(batch: List[str]) -> List[List[float]]:
    """Embed a batch of documents with retry logic.

    This function processes a batch of texts through the embedding pipeline
    with automatic retry on failures. It's designed to be used in concurrent
    processing scenarios where individual batches might fail due to transient issues.

    Args:
        batch: List of text strings to embed.

    Returns:
        List[List[float]]: List of embedding vectors for the input batch.

    Raises:
        Exception: If all retry attempts are exhausted.

    Example:
        >>> batch = ["text1", "text2", "text3"]
        >>> embeddings = await embed_batch_with_retry(batch)
        >>> len(embeddings) == len(batch)
        True
    """
    try:
        logger.debug(f"Embedding batch of {len(batch)} documents")
        return await create_embeddings_with_cache(batch)
    except Exception as e:
        logger.warning(
            f"Embedding batch failed, will retry: {str(e)}, batch is {batch}"
        )
        raise  # Re-raise to trigger retry


def create_documents_embedding(
    docs: List[str], batch_size: int = 100, max_concurrency: int = 5
) -> List[List[float]]:
    """Create embeddings for a list of documents with batching and concurrency.

    This function efficiently processes large document collections by:
    1. Splitting documents into manageable batches
    2. Processing batches concurrently with configurable limits
    3. Applying retry logic for robust operation
    4. Flattening results into a single list

    Args:
        docs: List of document strings to embed.
        batch_size: Number of documents to process in each batch.
        max_concurrency: Maximum number of concurrent batch operations.

    Returns:
        List[List[float]]: List of embedding vectors, one per input document.

    Raises:
        Exception: If embedding process fails after retries.

    Example:
        >>> docs = ["doc1", "doc2", "doc3", "doc4"]
        >>> embeddings = create_documents_embedding(docs, batch_size=2, max_concurrency=2)
        >>> len(embeddings) == len(docs)
        True
    """
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

