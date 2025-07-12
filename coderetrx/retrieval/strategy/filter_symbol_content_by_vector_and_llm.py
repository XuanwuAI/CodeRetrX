"""
Strategy for filtering symbols using vector similarity search followed by LLM refinement.
"""

from typing import List, Union, Optional, override
from .base import FilterByVectorAndLLMStrategy
from ..smart_codebase import (
    SmartCodebase as Codebase,
    LLMMapFilterTargetType,
    SimilaritySearchTargetType,
)
from coderetrx.static import Keyword, Symbol, File


class FilterSymbolContentByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter symbols using vector similarity search followed by LLM refinement."""

    @override
    def get_strategy_name(self) -> str:
        return "FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM"

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
        elements: List[Symbol],
        target_type: LLMMapFilterTargetType = "symbol_content",
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        """Filter and convert elements to the expected type for additional_code_elements."""
        filtered_symbol = [elem for elem in elements if isinstance(elem, Symbol)]
        if target_type == "class_content":
            # If the target type is class_content, filter symbols that are classes
            filtered_symbol = [
                elem for elem in filtered_symbol if elem.type == "class"
            ]
        elif target_type == "function_content":
            # If the target type is function_content, filter symbols that are functions
            filtered_symbol = [
                elem for elem in filtered_symbol if elem.type == "function"
            ]
        return filtered_symbol

    @override
    def collect_file_paths(
        self,
        filtered_elements: List[Symbol],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        """Collect file paths from the filtered symbols."""
        file_paths = set()
        for symbol in filtered_elements:
            if isinstance(symbol, Symbol):
                file_path = str(symbol.file.path)
                if not subdirs_or_files or any(
                    file_path.startswith(subdir) for subdir in subdirs_or_files
                ):
                    file_paths.add(file_path)
        return list(file_paths)
