
from httpx import AsyncClient
from pydantic_settings import BaseSettings
from coderetrx.utils.path import get_cache_dir
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
from coderetrx.utils.embedding import (
    create_embeddings_with_cache,
    create_documents_embedding,
    get_embedding_settings,
    similarity_search_retry,
)

from coderetrx.utils.concurrency import abatch_func_call, run_coroutine_sync
from abc import ABC, abstractmethod
from functools import wraps
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
import httpx
from dotenv import load_dotenv

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
        HnswConfigDiff,
        PayloadSchemaType,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
try:

    import chromadb #type:ignore

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
logger = logging.getLogger(__name__)
cache_path = get_cache_dir()

class SimilaritySearcher(ABC):
    """Abstract base class for similarity search implementations.

    Defines the interface for vector database similarity search operations.
    Concrete implementations handle specific database backends like ChromaDB
    or Qdrant while providing a consistent API.

    This class supports both synchronous and asynchronous search operations,
    with automatic text embedding generation and flexible filtering capabilities.

    Attributes:
        texts: List of texts indexed in the collection.
        name: Collection name identifier.
        vector_db_mode: Database reuse strategy.
        metadatas: Optional metadata associated with texts.
    """

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        indexed_metadata_fields: List[str] = [],
        vector_db_mode: str = "reuse_on_match",
        hnsw_m: Optional[int] = None,
    ) -> None:
        """Initialize the SimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            metadatas: Optional metadata list corresponding to each text.
            indexed_metadata_fields: Metadata fields to index (ignored for ChromaDB).
            vector_db_mode: Vector DB reuse mode ("always_reuse", "never_reuse", "reuse_on_match").
            hnsw_m: HNSW algorithm parameter for index construction.

        Example:
            >>> searcher = ConcreteSearcher(
            ...     name="my_docs",
            ...     texts=["doc1", "doc2"],
            ...     vector_db_mode="reuse_on_match"
            ... )
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
        """Search for the most similar documents to the query.

        Args:
            query: Query string to search for.
            k: Number of top results to return.
            threshold: Minimum similarity score threshold.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.

        Example:
            >>> results = await searcher.asearch_with_score("python code", k=5)
            >>> len(results) <= 5
            True
        """
        pass

    @abstractmethod
    async def asearch_by_vector(
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """Search for the most similar documents to the query vector.

        Args:
            query_vector: Pre-computed embedding vector for the query.
            k: Number of top results to return.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.

        Example:
            >>> vector = [0.1, 0.2, 0.3, ...]  # 1024-dimensional vector
            >>> results = await searcher.asearch_by_vector(vector, k=3)
            >>> len(results) <= 3
            True
        """
        pass

    def search_with_score(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        where: Optional[dict] = None,
    ):
        """Synchronous wrapper for asearch_with_score.

        Args:
            query: Query string to search for.
            k: Number of top results to return.
            threshold: Minimum similarity score threshold.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.

        Example:
            >>> results = searcher.search_with_score("machine learning", k=10)
        """
        from coderetrx.utils.concurrency import run_coroutine_sync

        return run_coroutine_sync(self.asearch_with_score(query, k, threshold, where))

    def search_by_vector(
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
    ):
        """Synchronous wrapper for asearch_by_vector.

        Args:
            query_vector: Pre-computed embedding vector for the query.
            k: Number of top results to return.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.

        Example:
            >>> vector = get_query_embedding("search term")
            >>> results = searcher.search_by_vector(vector, k=5)
        """
        from coderetrx.utils.concurrency import run_coroutine_sync

        return run_coroutine_sync(self.asearch_by_vector(query_vector, k, where))


class ChromaSimilaritySearcher(SimilaritySearcher):
    """ChromaDB-based implementation of SimilaritySearcher.

    Provides similarity search capabilities using ChromaDB as the vector database
    backend. ChromaDB is an open-source embedding database that's easy to set up
    and suitable for development and small to medium-scale production workloads.

    Features:
        - Persistent storage with automatic collection management
        - Cosine similarity search with HNSW indexing
        - Metadata filtering support
        - Automatic batch processing for large document sets
        - Collection reuse strategies for efficient development workflows

    Note:
        ChromaDB does not support indexed metadata fields, so the
        indexed_metadata_fields parameter is ignored with a warning.
    """

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        indexed_metadata_fields: List[str] = [],
        vector_db_mode: str = "reuse_on_match",
        hnsw_m: Optional[int] = None,
    ) -> None:
        """Initialize ChromaDB similarity searcher.

        Args:
            name: Collection name identifier.
            texts: List of texts to index.
            embeddings: Optional precomputed embeddings.
            metadatas: Optional metadata for each text.
            indexed_metadata_fields: Metadata fields to index (ignored for ChromaDB).
            vector_db_mode: Collection reuse strategy.
            hnsw_m: HNSW parameter M for index construction.

        Raises:
            ImportError: If ChromaDB dependencies are not available.
        """
        if not hnsw_m:
            hnsw_m = 1024
        super().__init__(name, texts, embeddings, metadatas,indexed_metadata_fields, vector_db_mode)
        if indexed_metadata_fields:
            logger.warning(
                "ChromaDB does not support indexed metadata fields, ignoring."
            )
        self._initialize_vector_store(embeddings, hnsw_m)

    def _initialize_vector_store(
        self, embeddings: Optional[List[List[float]]], hnsw_m: int
    ) -> None:
        """Initialize the ChromaDB vector store with collection management.

        Args:
            embeddings: Optional precomputed embeddings.
            hnsw_m: HNSW parameter for index construction.

        Raises:
            ImportError: If ChromaDB is not available.
        """
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
                    f"Using cached {self.name} from ChromaDB collection '{self.name}' ({collection_count} items) - always_reuse mode"
                )
            elif self.vector_db_mode == "never_reuse":
                logger.info(
                    f"Recreating ChromaDB collection '{self.name}' - never_reuse mode"
                )
                self.chromadb_client.delete_collection(self.name)
                self.collection = self.chromadb_client.create_collection(
                    name=self.name, metadata={"hnsw:space": "cosine", "hnsw:M": hnsw_m}
                )
                collection_exists = False
            elif collection_count != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection_count} documents instead of {len(self.texts)}. Recreating collection."
                )
                self.chromadb_client.delete_collection(self.name)
                self.collection = self.chromadb_client.create_collection(
                    name=self.name, metadata={"hnsw:space": "cosine", "hnsw:M": hnsw_m}
                )
                collection_exists = False
            else:
                logger.debug(
                    f"Using cached {self.name} from ChromaDB collection '{self.name}' ({collection_count} items)"
                )
        except Exception:
            logger.info(f"Creating new ChromaDB collection: '{self.name}'")
            self.collection = self.chromadb_client.create_collection(
                name=self.name, metadata={"hnsw:space": "cosine", "hnsw:M": hnsw_m}
            )
            collection_exists = False

        # Add texts and embeddings to collection based on vector_db_mode
        should_add_texts = (
            self.vector_db_mode == "never_reuse"
            or not collection_exists
            or (
                self.vector_db_mode == "reuse_on_match"
                and collection_count != len(self.texts)
            )
        )

        if should_add_texts:
            self._add_texts_in_batches(self.texts, embeddings, self.metadatas)

    def _validate_embeddings_match_texts(
        self, texts: List[str], embeddings: List[List[float]]
    ) -> None:
        """Validate that embeddings count matches texts count.

        Args:
            texts: List of text strings.
            embeddings: List of embedding vectors.

        Raises:
            ValueError: If counts don't match.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

    def _validate_metadatas_match_texts(
        self, texts: List[str], metadatas: List[dict]
    ) -> None:
        """Validate that metadata count matches texts count.

        Args:
            texts: List of text strings.
            metadatas: List of metadata dictionaries.

        Raises:
            ValueError: If counts don't match.
        """
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadatas")

    def _add_texts_in_batches(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Add texts to vector store in batches with progress tracking.

        Args:
            texts: List of texts to add.
            embeddings: Optional precomputed embeddings.
            metadatas: Optional metadata for each text.
            batch_size: Number of texts to process per batch.
        """
        logger.info(
            f"Adding {len(texts)} documents to {self.__class__.__name__} collection '{self.name}'"
        )

        for idx in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Adding {self.name} to {self.__class__.__name__.replace('SimilaritySearcher', '')}",
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
        """Add texts with precomputed embeddings to the collection.

        Args:
            texts: List of texts to add.
            embeddings: List of embedding vectors.
            metadatas: Optional metadata for each text.
            start_idx: Starting index for document IDs.
        """
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
        """Search for the most similar documents to the query.

        Args:
            query: Query string to search for.
            k: Number of top results to return.
            threshold: Minimum similarity score threshold.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.
        """
        logger.debug(
            f"Performing similarity search in collection '{self.name}' with k={k}"
        )

        # Get query embedding
        try:
            query_embedding = await create_embeddings_with_cache([query])
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
        """Search for the most similar documents to the query vector.

        Args:
            query_vector: Pre-computed embedding vector for the query.
            k: Number of top results to return.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.
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
    """High-performance Qdrant-based implementation using direct Qdrant API.

    Provides similarity search capabilities using Qdrant as the vector database
    backend. Qdrant is a high-performance vector database designed for production
    workloads with advanced filtering and indexing capabilities.

    Features:
        - High-performance vector search with HNSW indexing
        - Advanced metadata filtering with indexed fields
        - Adaptive batch sizing for optimal performance
        - Connection pooling and retry logic
        - Automatic collection management
        - Payload schema optimization
    """

    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        indexed_metadata_fields: List[str] = [],
        vector_db_mode: str = "reuse_on_match",
        hnsw_m: Optional[int] = None,
    ) -> None:
        """Initialize the QdrantSimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            metadatas: Optional metadata list corresponding to each text.
            indexed_metadata_fields: List of metadata fields to create indexes for.
            vector_db_mode: Vector DB reuse mode ("always_reuse", "never_reuse", "reuse_on_match").
            hnsw_m: HNSW parameter M for index construction.

        Raises:
            ImportError: If Qdrant dependencies are not available.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant dependencies not available. Install with: pip install qdrant-client"
            )
        if not hnsw_m:
            hnsw_m = 16
        super().__init__(name, texts, embeddings, metadatas,indexed_metadata_fields, vector_db_mode)
        self.vector_size = get_embedding_settings().embedding_dimension

        # Initialize Qdrant clients
        self._init_clients()

        # Initialize collection
        self._init_collection(hnsw_m)

        # Add texts based on vector_db_mode
        should_add_texts = self.vector_db_mode == "never_reuse" or (
            self.vector_db_mode == "reuse_on_match"
            and self._get_collection_count() != len(texts)
        )
        if should_add_texts:
            if metadatas and indexed_metadata_fields:
                self._create_metadata_indexes(metadatas, indexed_metadata_fields)
            self._add_texts_to_collection(embeddings)

    def _create_metadata_indexes(
        self, metadatas: List[dict], indexed_metadata_fields: List[str]
    ) -> None:
        """Create indexes for specified metadata fields.

        Args:
            metadatas: List of metadata dictionaries.
            indexed_metadata_fields: List of field names to index.
        """
        for field in indexed_metadata_fields:
            if metadatas is None or len(metadatas) == 0 or field not in metadatas[0]:
                logger.warning(
                    f"Cannot create index for field '{field}' - field not found in metadata"
                )
                continue
            field_sample = metadatas[0][field]
            field_type = ""
            if type(field_sample) == list and len(field_sample) > 0:
                field_sample = field_sample[0]
            if isinstance(field_sample, float):
                field_type = PayloadSchemaType.FLOAT
            elif isinstance(field_sample, int):
                field_type = PayloadSchemaType.INTEGER
            elif isinstance(field_sample, bool):
                field_type = PayloadSchemaType.BOOL
            elif isinstance(field_sample, str):
                field_type = PayloadSchemaType.KEYWORD
            else:
                logger.warning(
                    f"Cannot create index for field '{field}' - unsupported type {type(field_sample)}"
                )
                continue

            self.qdrant_client.create_payload_index(
                collection_name=self.name, field_name=field, field_schema=field_type
            )

    def _init_clients(self) -> None:
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

    def _init_collection(self, hnsw_m: int) -> None:
        """Initialize or create Qdrant collection.

        Args:
            hnsw_m: HNSW parameter for index construction.
        """
        try:
            collection_count = self.qdrant_client.count(collection_name=self.name).count

            if self.vector_db_mode == "always_reuse":
                logger.debug(
                    f"Using cached {self.name} from Qdrant collection '{self.name}' ({collection_count} items) - always_reuse mode"
                )
            elif self.vector_db_mode == "never_reuse":
                logger.info(
                    f"Recreating Qdrant collection '{self.name}' - never_reuse mode"
                )
                self.qdrant_client.delete_collection(self.name)
                self._create_collection(hnsw_m)
            elif collection_count != len(self.texts):
                logger.info(
                    f"Collection '{self.name}' exists but has {collection_count} documents instead of {len(self.texts)}. Recreating collection."
                )
                self.qdrant_client.delete_collection(self.name)
                self._create_collection(hnsw_m)
            else:
                logger.info(
                    f"Using cached {self.name} from Qdrant collection '{self.name}' ({collection_count} items)"
                )

        except Exception:
            logger.info(f"Creating new Qdrant collection: '{self.name}'")
            self._create_collection(hnsw_m)

    def _create_collection(self, hnsw_m: int) -> None:
        """Create a new Qdrant collection.

        Args:
            hnsw_m: HNSW parameter for index construction.
        """
        self.qdrant_client.create_collection(
            collection_name=self.name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=hnsw_m,
            ),
        )

    def _get_collection_count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            int: Number of documents in the collection.
        """
        try:
            return self.qdrant_client.count(collection_name=self.name).count
        except:
            return 0

    def _add_texts_to_collection(
        self, embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """Add texts and embeddings to the Qdrant collection.

        Args:
            embeddings: Optional precomputed embeddings.
        """
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
            range(total_batches), desc=f"Adding {self.name} to Qdrant"
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
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=30, max=600),
        retry=retry_if_exception_type(Exception),
    )
    def _add_batch_to_qdrant(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        start_idx: int = 0,
    ) -> None:
        """Add a batch of texts with embeddings to Qdrant.

        Args:
            texts: List of texts to add.
            embeddings: List of embedding vectors.
            metadatas: Optional metadata for each text.
            start_idx: Starting index for point IDs.
        """
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
                logger.warning(f"Retrying due to {e}...")
                raise

    @similarity_search_retry
    async def asearch_with_score(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        where: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """Search for the most similar documents to the query.

        Args:
            query: Query string to search for.
            k: Number of top results to return.
            threshold: Minimum similarity score threshold.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.
        """
        logger.info(
            f"Performing similarity search in collection '{self.name}' with k={k}"
        )

        try:
            # Get query embedding
            query_embedding = await create_embeddings_with_cache([query])
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
        """Search for the most similar documents to the query vector.

        Args:
            query_vector: Pre-computed embedding vector for the query.
            k: Number of top results to return.
            where: Optional filter conditions as key-value pairs.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples sorted by relevance.
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

                content = point.payload.get("content", "") #type: ignore

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
        """Build Qdrant filter from dictionary conditions.

        Args:
            where: Dictionary of filter conditions.

        Returns:
            Filter: Qdrant filter object.
        """
        conditions = []

        for key, value in where.items():
            if isinstance(value, (str, int, bool)):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None


def get_similarity_searcher(
    provider: str,
    name: str,
    texts: List[str],
    embeddings: Optional[List[List[float]]] = None,
    metadatas: Optional[List[dict]] = None,
    indexed_metadata_fields: List[str] = [],
    vector_db_mode: str = "reuse_on_match",
    hnsw_m: Optional[int] = None,
) -> SimilaritySearcher:
    """Factory function to create similarity searcher instances based on provider.

    Args:
        provider: The vector database provider to use (e.g., "chroma", "qdrant").
        name: Name of the collection.
        texts: List of texts to be indexed.
        embeddings: Optional precomputed embeddings corresponding to the texts.
        metadatas: Optional metadata list corresponding to each text.
        indexed_metadata_fields: List of metadata fields to create indexes for.
        vector_db_mode: Vector DB reuse mode ("always_reuse", "never_reuse", "reuse_on_match").
        hnsw_m: HNSW parameter for index construction.

    Returns:
        SimilaritySearcher: A SimilaritySearcher instance based on the specified provider.

    Raises:
        ValueError: If the provider is not supported.

    Example:
        >>> searcher = get_similarity_searcher(
        ...     provider="qdrant",
        ...     name="my_docs",
        ...     texts=["doc1", "doc2"],
        ...     vector_db_mode="reuse_on_match"
        ... )
    """
    if provider.lower() == "chroma":
        return ChromaSimilaritySearcher(
            name=name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            indexed_metadata_fields=indexed_metadata_fields,
            vector_db_mode=vector_db_mode,
            hnsw_m=hnsw_m,
        )
    elif provider.lower() == "qdrant":
        return QdrantSimilaritySearcher(
            name=name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            indexed_metadata_fields=indexed_metadata_fields,
            vector_db_mode=vector_db_mode,
            hnsw_m=hnsw_m,
        )
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")