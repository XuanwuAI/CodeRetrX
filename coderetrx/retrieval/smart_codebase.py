from abc import ABC, abstractmethod
from coderetrx.static import Codebase, Keyword, Symbol, File
from typing import Literal, List, Tuple, Any, Union, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from coderetrx.static.codebase.codebase import CodeLine


class SmartCodebaseSettings(BaseSettings):
    """Configuration settings for SmartCodebase with environment variable support."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )

    # LLM Configuration
    default_model_id: str = Field(
        default="openai/gpt-4.1-mini",
        description="Default model ID for all LLM operations",
    )
    llm_mapfilter_model_id: Optional[str] = Field(
        default=None,
        description="Primary model ID for LLM map/filter operations",
    )
    llm_mapfilter_special_model_id: Optional[str] = Field(
        default=None,
        description="Special model ID for LLM map/filter operations",
    )
    llm_fallback_model_id: Optional[str] = Field(
        default=None,
        description="Fallback model ID for LLM operations",
    )
    llm_topic_extraction_model_id: Optional[str] = Field(
        default=None,
        description="Model ID for topic extraction operations",
        alias="LLM_TOPIC_EXTRACTION_MODEL_ID",
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
        default=True,
        description="Enable symbol name embeddings",
        alias="SYMBOL_NAME_EMBEDDING",
    )
    symbol_content_embedding: bool = Field(
        default=False,
        description="Enable symbol content embeddings",
        alias="SYMBOL_CONTENT_EMBEDDING",
    )

    keyword_embedding: bool = Field(
        default=False,
        description="Enable keyword embeddings",
        alias="KEYWORD_EMBEDDING",
    )
    symbol_codeline_embedding: bool = Field(
        default=True,
        description="Enable symbol codeline embeddings",
        alias="SYMBOL_CODELINE_EMBEDDING",
    )
    symbol_codeline_embedding_maxchars: int = Field(
        default=100,
        description="Maximum number of characters to embed for symbol codelines. Set to 0 to embed 1 line at a time.",
        alias="SYMBOL_CODELINE_EMBEDDING_MAXCHARS",
    )

    vector_db_provider: Literal["chroma", "qdrant"] = Field(
        default="qdrant",
        description="Provider of vector database to use for embeddings",
        alias="VECTOR_DB_PROVIDER",
    )
    
    # Codebase Processing Configuration
    max_chunks_one_file: int = Field(
        default=500,
        description="Maximum number of chunks allowed in one file",
        alias="MAX_CHUNKS_ONE_FILE",
    )
    keyword_sentence_extraction: bool = Field(
        default=False,
        description="Enable sentence-based keyword extraction",
        alias="KEYWORD_SENTENCE_EXTRACTION",
    )
      
    vector_db_mode: Literal["always_reuse", "never_reuse", "reuse_on_match"] = Field(
        default="reuse_on_match",
        description="Vector DB reuse mode: always_reuse (always use existing), never_reuse (always recreate), reuse_on_match (reuse only if collection count matches)",
        alias="VECTOR_DB_MODE",
    )


LLMMapFilterTargetType = Literal[
    "file_name",
    "file_content",
    "symbol_name",
    "symbol_content",
    "root_symbol_name",
    "root_symbol_content",
    "leaf_symbol_name",
    "leaf_symbol_content",
    "class_name",
    "class_content",
    "function_name",
    "function_content",
    "dependency_name",
    "dependency_reference",
    "dependency",
    "keyword",
]

SimilaritySearchTargetType = Literal[
    "symbol_name", "symbol_content", "keyword", "symbol_codeline"
]
LLMCallMode = Literal["traditional", "function_call"]


class CodeMapFilterResult(BaseModel):
    index: int
    reason: str
    result: Any
    is_extended_match: bool = False


class KeywordExtractorResult(BaseModel):
    reason: str
    result: str


class SmartCodebase(Codebase, ABC):
    @abstractmethod
    async def llm_filter(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: Optional[List[str]] = None,
        additional_code_elements: Optional[List[Union[Keyword, Symbol, File]]] = None,
        llm_call_mode: LLMCallMode = "traditional",
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[CodeMapFilterResult]]:
        pass

    @abstractmethod
    async def llm_map(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: Optional[List[str]] = None,
        additional_code_elements: Optional[List[Union[Keyword, Symbol, File]]] = None,
        llm_call_mode: LLMCallMode = "traditional",
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[CodeMapFilterResult]]:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        target_types: List[SimilaritySearchTargetType],
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 100,
    ) -> List[Symbol | Keyword]:
        pass

    @abstractmethod
    async def similarity_search_lines_per_symbol(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 10,
        scope: Literal[
            "root_symbol", "leaf_symbol", "symbol", "class", "function"
        ] = "symbol",
        subdirs_or_files: Optional[List[str]] = None,
    ) -> List[CodeLine]:
        """
        Search for similar lines within a specific symbol using metadata filtering.
        """
        pass

    @abstractmethod
    async def similarity_search_lines_per_file(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 10,
        subdirs_or_files: Optional[List[str]] = None,
    ) -> List[CodeLine]:
        """
        Search for similar lines grouped by file using metadata filtering.
        """
        pass
