"""
Strategy for adaptive filtering of symbols using vector similarity search followed by LLM refinement.
"""

from typing import List, Union, Optional, override, Literal, Any
from .base import (
    AdaptiveFilterByVectorAndLLMStrategy,
    FilterByVectorAndLLMStrategy,
    StrategyExecuteResult,
)
from ..smart_codebase import (
    SmartCodebase as Codebase,
    LLMMapFilterTargetType,
    SimilaritySearchTargetType,
)
from coderetrx.static import Keyword, Symbol, File


class AdaptiveFilterSymbolContentByVectorAndLLMStrategy(AdaptiveFilterByVectorAndLLMStrategy):
    """Strategy to filter symbols using adaptive vector similarity search followed by LLM refinement."""

    name: str = "ADAPTIVE_FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["symbol_content"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "symbol_content"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

    @override
    def filter_elements(
        self,
        elements: List[Any],
        target_type: LLMMapFilterTargetType = "symbol_content",
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        filtered_symbols: List[Union[Keyword, Symbol, File]] = []
        for element in elements:
            if not isinstance(element, Symbol):
                continue
                # If subdirs_or_files is provided and codebase is available, filter by subdirs
            if subdirs_or_files and codebase:
                # Get the relative path from the codebase directory
                rpath = str(element.file.path)
                if any(rpath.startswith(subdir) for subdir in subdirs_or_files):
                    filtered_symbols.append(element)
            else:
                filtered_symbols.append(element)
        if target_type == "class_content":
            # If the target type is class_content, filter symbols that are classes
            filtered_symbols = [
                elem for elem in filtered_symbols if elem.type == "class"
            ]
        elif target_type == "function_content":
            # If the target type is function_content, filter symbols that are functions
            filtered_symbols = [
                elem for elem in filtered_symbols if elem.type == "function"
            ]
        return filtered_symbols

    @override
    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        file_paths = []
        for symbol in filtered_elements:
            if isinstance(symbol, Symbol):
                file_path = str(symbol.file.path)
                if file_path.startswith(tuple(subdirs_or_files)):
                    file_paths.append(file_path)
        return file_paths

    @override
    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
        target_type: str = "symbol_content",
    ) -> StrategyExecuteResult:
        prompt = f"""
        requirement: A code chunk with this name is highly likely to meet the following criteria:
        <content_criteria>
        {prompt}
        </content_criteria>
        <note>
        The objective of this requirement is to preliminarily identify code chunks that are likely to meet specific content criteria based on their names. 
        Code chunks with matching names will proceed to deeper analysis in the content filter (content_criteria) at a later stage (not in this run). 
        </note>
        """
        return await super().execute(codebase, prompt, subdirs_or_files, target_type)
