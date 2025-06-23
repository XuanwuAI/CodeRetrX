from attrs import define
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from codelib.utils.embedding import SimilaritySearcher
from codelib.static.codebase import Codebase
from .smart_codebase import SmartCodebase

import logging

logger = logging.getLogger(__name__)


class SmartCodebaseSettings(BaseSettings):
    """Configuration settings for SmartCodebase with environment variable support."""
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )

    # LLM Configuration
    llm_mapfilter_model_id: str = Field(
        default="mistralai/devstral-small",
        description="Primary model ID for LLM map/filter operations",
    )
    llm_function_call_model_id: str = Field(
        default="mistralai/devstral-small",
        description="Primary model ID for function call operations",
    )
    llm_fallback_model_id: str = Field(
        default="anthropic/claude-3.7-sonnet",
        description="Fallback model ID for LLM operations",
    )

    # Batch Configuration
    llm_mapfilter_max_batch_length: int = Field(
        default=10000, description="Maximum character length for LLM batch processing"
    )
    llm_mapfilter_max_batch_size: int = Field(
        default=100, description="Maximum number of elements in a batch"
    )

    # Concurrency Configuration
    llm_max_concurrent_requests: int = Field(
        default=5, description="Maximum number of concurrent LLM requests"
    )

    # Similarity Search Configuration
    similarity_search_threshold: float = Field(
        default=0.1, description="Threshold for similarity search results"
    )

    # Embedding flags
    symbol_name_embedding: bool = Field(
        default=False,
        description="Enable symbol name embeddings",
        alias="SYMBOL_NAME_EMBEDDING",
    )
    symbol_content_embedding: bool = Field(
        default=False,
        description="Enable symbol content embeddings",
        alias="SYMBOL_CONTENT_EMBEDDING",
    )
    keyword_embedding: bool = Field(
        default=False, description="Enable keyword embeddings", alias="KEYWORD_EMBEDDING"
    )

@define
class CodebaseFactory:
    @classmethod
    def from_json(cls, data: dict) -> SmartCodebase:
        assert isinstance(data, dict)
        codebase = Codebase.from_json(data)
        settings = SmartCodebaseSettings()
        smart_codebase = SmartCodebase(
            id=codebase.id,
            dir=codebase.dir,
            source_files=codebase.source_files,
            dependency_files=codebase.dependency_files,
            symbols=codebase.symbols,
            keywords=codebase.keywords,
            dependencies=codebase.dependencies,
            settings=settings,
        )
        smart_codebase.init_all()
        cls._initialize_similarity_searchers(smart_codebase, settings)
        return smart_codebase

    @classmethod
    def new(cls, id: str, dir: Path) -> SmartCodebase:
        settings = SmartCodebaseSettings()
        smart_codebase = SmartCodebase.new(id, dir, settings=settings)
        smart_codebase.init_all()
        cls._initialize_similarity_searchers(smart_codebase, settings)
        return smart_codebase

    @classmethod
    def _initialize_similarity_searchers(
        cls, codebase: SmartCodebase, settings: SmartCodebaseSettings
    ) -> None:
        """
        Initialize SimilaritySearcher instances for the codebase if embeddings are available.

        Args:
            codebase: The codebase to initialize searchers for
        """

        # Initialize symbol name searcher if embeddings are available
        if settings.symbol_name_embedding:
            try:
                symbol_names = list(set([symbol.name for symbol in codebase.symbols]))
                logger.info(
                    f"Initializing symbol name similarity searcher with {len(symbol_names)} unique symbols"
                )
                codebase.symbol_name_searcher = SimilaritySearcher(
                    f"{codebase.id}_symbol_names",
                    symbol_names,
                )
                logger.info("Symbol name similarity searcher initialized successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize symbol name similarity searcher: {e}"
                )
        else:
            logger.info(
                "Symbol name embeddings feature is not enabled (SYMBOL_NAME_EMBEDDING not set), symbol name searcher not initialized"
            )

        # Initialize symbol content searcher if embeddings are available

        if settings.symbol_content_embedding:
            try:
                symbol_contents = [
                    symbol.chunk.code() for symbol in codebase.symbols if symbol.chunk
                ]
                logger.info(
                    f"Initializing symbol content similarity searcher with {len(symbol_contents)} symbol contents"
                )
                codebase.symbol_content_searcher = SimilaritySearcher(
                    f"{codebase.id}_symbol_contents",
                    symbol_contents,
                )
                logger.info(
                    "Symbol content similarity searcher initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize symbol content similarity searcher: {e}"
                )
        else:
            logger.info(
                "Symbol content embeddings feature is not enabled (SYMBOL_CONTENT_EMBEDDING not set), symbol content searcher not initialized"
            )

        # Initialize keyword searcher if embeddings are available
        if settings.keyword_embedding:
            try:
                keyword_contents = [keyword.content for keyword in codebase.keywords]
                logger.info(
                    f"Initializing keyword similarity searcher with {len(keyword_contents)} unique keywords"
                )
                codebase.keyword_searcher = SimilaritySearcher(
                    f"{codebase.id}_keywords",
                    keyword_contents,
                )
                logger.info("Keyword similarity searcher initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize keyword similarity searcher: {e}")
        else:
            logger.info(
                "Keyword embeddings feature is not enabled (KEYWORD_EMBEDDING not set), keyword searcher not initialized"
            )
