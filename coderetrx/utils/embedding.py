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
from typing import List
from pydantic import SecretStr
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from coderetrx.utils.concurrency import abatch_func_call, run_coroutine_sync
import chromadb
import logging


logger = logging.getLogger(__name__)

cache_path = Path(__file__).parent.parent.parent / ".cache"
embedder_cache_store = LocalFileStore(cache_path / "embeddings")

chromadb_client = chromadb.PersistentClient(path=str(cache_path / "chroma"))

import json

underlying_embeddings = OpenAIEmbeddings(
    model=os.environ["EMBEDDING_MODEL_ID"],
    base_url=os.environ["EMBEDDING_BASE_URL"],
    api_key=SecretStr(os.environ["EMBEDDING_API_KEY"]),
    openai_proxy=None,
)  # type:ignore

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, embedder_cache_store, namespace=underlying_embeddings.model
)


# Define a retry-decorated function for embedding documents
@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(
        min=30, max=600
    ),  # Exponential backoff with a minimum wait of 30s and max of 10min
    retry=retry_if_exception_type(Exception),  # Retry only if an exception is raised
)
async def embed_batch_with_retry(batch):
    """Embed a batch of documents with retry logic."""
    try:
        return await cached_embedder.aembed_documents(batch)
    except Exception as e:
        logger.warning(f"Embedding batch failed, will retry: {str(e)}")
        raise  # Re-raise to trigger retry


def create_documents_embedding(docs: List[str], batch_size=100, max_concurrency=5):
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
        logger.info(
            f"Successfully created embeddings for {len(docs)} documents with {max_concurrency} concurrent workers"
        )
        return result
    except Exception as e:
        logger.error(f"Failed to embed documents: {str(e)}", exc_info=True)
        raise  # Re-raise the exception for the caller to handle


from typing import List, Optional


class SimilaritySearcher:
    def __init__(
        self,
        name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        use_cache: bool = True,
    ) -> None:
        """
        Initialize the SimilaritySearcher.

        Args:
            name: Name of the collection.
            texts: List of texts to be indexed.
            embeddings: Optional precomputed embeddings corresponding to the texts.
            use_cache: Whether to use cached collection if available.
        """
        self.texts = texts
        self.name = name
        self.use_cache = use_cache

        # Check or create Chroma collection

        try:
            collection = chromadb_client.get_collection(name)
            if collection.count() != len(texts):
                logger.info(
                    f"Collection '{name}' exists but has {collection.count()} documents instead of {len(texts)}. Recreating collection."
                )
                use_cache = False
            else:
                logger.info(
                    f"Using existing ChromaDB collection '{name}' with {collection.count()} documents"
                )

            if not use_cache:
                chromadb_client.delete_collection(name)
                collection = chromadb_client.get_collection(name)
        except:
            logger.info(f"Creating new ChromaDB collection: '{name}'")
            collection = chromadb_client.create_collection(
                name, metadata={"hnsw:space": "cosine"}
            )

        self.collection = collection

        # Add texts and embeddings to collection if not using cache
        if not use_cache:
            logger.info(
                f"Adding {len(texts)} documents to ChromaDB collection '{name}'"
            )
            batch_size = 1000
            for idx in tqdm(range(0, len(texts), batch_size), desc="Adding documents to ChromaDB"):
                text_batch = texts[idx : idx + batch_size]
                if embeddings:
                    embedding_batch = embeddings[idx : idx + batch_size]
                    self._add_texts_with_embeddings(text_batch, embedding_batch)
                else:
                    self.collection.add(
                        documents=text_batch,
                        embeddings=create_documents_embedding(text_batch),
                        ids=[str(idx + i) for i in range(len(text_batch))],
                    )

    def _add_texts_with_embeddings(
        self, texts: List[str], embeddings: List[List[float]]
    ):
        """Add texts with precomputed embeddings to the collection."""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        self.collection.add(
            documents=texts,
            embeddings=embeddings,  # type: ignore
            ids=[str(i) for i in range(len(texts))],
        )

    def add_texts(
        self, texts: List[str], embeddings: Optional[List[List[float]]] = None
    ):
        """Add texts to the collection, optionally with precomputed embeddings."""
        if embeddings:
            self._add_texts_with_embeddings(texts, embeddings)
        else:
            self.collection.add(
                documents=texts,
                embeddings=create_documents_embedding(texts),
                ids=[str(i) for i in range(len(texts))],
            )

    def search(self, query: str, k: int = 10):
        """
        Search for the most similar documents to the query.

        Args:
            query: Query string.
            k: Number of top results to return.

        Returns:
            A list of tuples (document, normalized_score).
        """

        def normalize_score(raw_score, metric="cosine", max_distance=1.0):
            """
            Normalize raw similarity or distance score to [0, 1].

            Args:
                raw_score: The raw similarity or distance score.
                metric: The metric used ('cosine' or 'euclidean').
                max_distance: Maximum distance for euclidean normalization.

            Returns:
                Normalized score in [0, 1].
            """
            if metric == "cosine":
                # Normalize cosine similarity from [-1, 1] to [0, 1]
                return (raw_score + 1) / 2
            elif metric == "euclidean":
                # Normalize euclidean distance to [0, 1]
                return max(0, 1 - (raw_score / max_distance))
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        logger.info(
            f"Performing similarity search in collection '{self.name}' with k={k}"
        )
        query_embedding = create_documents_embedding([query])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        documents = results["documents"][0]  # type: ignore
        distances = results["distances"][0]  # type: ignore
        assert documents is not None
        assert distances is not None

        # Normalize distances or scores to [0, 1]
        normalized_scores = [
            normalize_score(-distance, metric="cosine") for distance in distances
        ]  # Use cosine similarity normalization

        result = list(zip(documents, normalized_scores))
        logger.info(
            f"Found {len(result)} matching documents in collection '{self.name}'"
        )
        return result
