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
    llm_mapfilter_model_id: str = Field(
        default="mistralai/devstral-small",
        description="Primary model ID for LLM map/filter operations",
    )
    llm_function_call_model_id: str = Field(
        default="mistralai/devstral-small",
        description="Primary model ID for function call operations",
    )
    llm_mapfilter_special_model_id: str = Field(
        default="openai/gpt-4.1-mini",
        description="Special model ID for LLM map/filter operations",
    )
    llm_fallback_model_id: str = Field(
        default="anthropic/claude-sonnet-4",
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
    symbol_codeline_embedding: bool = Field(
        default=False,
        description="Enable symbol codeline embeddings",
        alias="SYMBOL_CODELINE_EMBEDDING",
    )

LLMMapFilterTargetType = Literal[
    "file_name",
    "file_content",
    "symbol_name",
    "symbol_content",
    "class_name",
    "class_content",
    "function_name",
    "function_content",
    "dependency_name",
    "dependency_reference",
    "dependency",
    "keyword",
]
SimilaritySearchTargetType = Literal["symbol_name", "symbol_content", "keyword"]
LLMCallMode = Literal["traditional", "function_call"]

class CodeMapFilterResult(BaseModel):
    index: int
    reason: str
    result: Any


class KeywordExtractorResult(BaseModel):
    reason: str
    result: str


class SmartCodebase(Codebase, ABC):
    @abstractmethod
    async def llm_filter(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: List[str] = [],
        additional_code_elements: List[Union[Keyword, Symbol, File]] = [],
        llm_call_mode: LLMCallMode = "traditional",
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[CodeMapFilterResult]]:
        pass

    @abstractmethod
    async def llm_map(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: List[str] = [],
        additional_code_elements: List[Union[Keyword, Symbol, File]] = [],
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
    ) -> List[Symbol | Keyword ]:
        pass

    @abstractmethod
    async def similarity_search_lines_per_symbol(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 10,
    ) -> List[CodeLine]:
        """
        Search for similar lines within a specific symbol using metadata filtering.
        """
        pass
