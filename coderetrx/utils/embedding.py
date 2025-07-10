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
from typing import List
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


logger = logging.getLogger(__name__)

cache_path = get_cache_dir() 
embedder_cache_store = LocalFileStore(cache_path / "embeddings")

chromadb_client = chromadb.PersistentClient(path=str(cache_path / "chroma"))
langchain_chroma_client = Chroma(client=chromadb_client)
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
        logger.info(f"Embedding batch of {len(batch)} documents")
        return await cached_embedder.aembed_documents(batch)
    except Exception as e:
        logger.warning(f"Embedding batch failed, will retry: {str(e)}, batch is {batch}")
        raise  # Re-raise to trigger retry


def create_documents_embedding(docs: List[str], batch_size=100, max_concurrency=15):
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
        if "symbol_names" in name:
            content_type = "symbol names"
        elif "symbol_contents" in name:
            content_type = "symbol contents"
        elif "symbol_codelines" in name:
            content_type = "code lines"
        elif "keywords" in name:
            content_type = "keywords"
        elif "symbol_codelines" in name:
            content_type = "symbol code lines"
        else:
            content_type = "documents"


        # Check or create Chroma collection

        try:
            collection = chromadb_client.get_collection(name)
            if collection.count() != len(texts):
                logger.info(
                    f"Collection '{name}' exists but has {collection.count()} documents instead of {len(texts)}. Recreating collection."
                )
                use_cache = False
                chromadb_client.delete_collection(name)
                collection = chromadb_client.create_collection(name, metadata={"hnsw:space": "cosine"})
            else:
                logger.info(f"Using cached {content_type} from ChromaDB collection '{name}' ({collection.count()} items)")

        except Exception as e:
            logger.info(f"Creating new ChromaDB collection: '{name}'")
            collection = chromadb_client.create_collection(
                name, metadata={"hnsw:space": "cosine"}
            )
            use_cache = False

        self.langchian_chroma_client = Chroma(
            client=chromadb_client,
            collection_name=name,
            embedding_function=cached_embedder,
            persist_directory=str(cache_path / "chroma"),
        )

        # Add texts and embeddings to collection if not using cache
        if not use_cache:
            logger.info(
                f"Adding {len(texts)} documents to ChromaDB collection '{name}'"
            )
            batch_size = 1000
            
            for idx in tqdm(range(0, len(texts), batch_size), desc=f"Adding {content_type} to ChromaDB"):
                text_batch = texts[idx : idx + batch_size]
                metadata_batch = metadatas[idx : idx + batch_size] if metadatas else None
                if embeddings:
                    embedding_batch = embeddings[idx : idx + batch_size]
                    self._add_texts_with_embeddings(text_batch, embedding_batch, metadata_batch)
                else:
                    add_kwargs = {
                        "texts": text_batch,
                        "embeddings": create_documents_embedding(text_batch),
                        "ids": [str(idx + i) for i in range(len(text_batch))],
                    }
                    if metadata_batch:
                        add_kwargs["metadatas"] = metadata_batch
                    self.langchian_chroma_client.add_texts(**add_kwargs)

    def _add_texts_with_embeddings(
        self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None
    ):
        """Add texts with precomputed embeddings to the collection."""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        if metadatas and len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadatas")

        add_kwargs = {
            "texts": texts,
            "embeddings": embeddings,
            "ids": [str(i) for i in range(len(texts))],
        }
        if metadatas:
            add_kwargs["metadatas"] = metadatas
        
        self.langchian_chroma_client.add_texts(**add_kwargs)
    @retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(
        min=30, max=600
    ),  # Exponential backoff with a minimum wait of 30s and max of 10min
    retry=retry_if_exception_type(Exception),  # Retry only if an exception is raised
    )
    async def asearch_with_score(self, query: str,  k: int = 10, threshold = 0.0 ,where: Optional[dict]=None):
        """
        Search for the most similar documents to the query.

        Args:
            query: Query string.
            k: Number of top results to return.

        Returns:
            A list of tuples (document, normalized_score).
        """

        logger.info(
            f"Performing similarity search in collection '{self.name}' with k={k}"
        )
        if where:
            logger.info(f"Applying filter: {where}")
            results = await self.langchian_chroma_client.asimilarity_search_with_relevance_scores(
                query = query, filter=where, k=k, score_threshold=threshold
            )
        else:
            results = await self.langchian_chroma_client.asimilarity_search_with_relevance_scores(query=query, k=k, score_threshold=threshold)
        results = list(map(lambda x: (x[0].page_content, x[1]), results))
        return results
    @retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(
        min=30, max=600
    ),  # Exponential backoff with a minimum wait of 30s and max of 10min
    retry=retry_if_exception_type(Exception),  # Retry only if an exception is raised
    )
    async def asearch_by_vector(
        self, query_vector: List[float], k: int = 10, where: Optional[dict] = None
    ):
        """
        Search for the most similar documents to the query vector.

        Args:
            query_vector: Query vector.
            k: Number of top results to return.

        Returns:
            A list of tuples (document, normalized_score).
        """
        logger.info(
            f"Performing vector similarity search in collection '{self.name}' with k={k}"
        )
        try: 
            if where:
                logger.info(f"Applying filter: {where}")
                docs = await self.langchian_chroma_client.asimilarity_search_by_vector(
                    embedding=query_vector, filter=where, k=k
                )
            else:
                docs = await self.langchian_chroma_client.asimilarity_search_by_vector(embedding=query_vector, k=k, score_threshold=threshold)
        except Exception as e:
            logger.error(f"Error during similarity search: {repr(e)}")
            raise
        

        docs= [doc.page_content for doc in docs]
        return docs