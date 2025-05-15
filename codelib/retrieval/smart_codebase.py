from abc import ABC, abstractmethod
from codelib.static import Codebase, Keyword, Symbol, File
from typing import Literal, List, Tuple, Any, Union, Optional
from pydantic import BaseModel

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
    ) -> Tuple[List[Any], List[CodeMapFilterResult]]:
        pass

    @abstractmethod
    async def llm_map(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: List[str] = [],
        additional_code_elements: List[Union[Keyword, Symbol, File]] = [],
    ) -> Tuple[List[Any], List[CodeMapFilterResult]]:
        pass

    @abstractmethod
    def similarity_search(
        self,
        target_types: List[SimilaritySearchTargetType],
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 100,
    ) -> List[Symbol | Keyword]:
        pass
