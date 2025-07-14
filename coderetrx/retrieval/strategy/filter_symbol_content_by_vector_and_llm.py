"""
Strategy for filtering symbols using vector similarity search followed by LLM refinement.
"""

from collections import defaultdict
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
        filtered_symbols: List[Symbol] = []
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
        elif target_type == "leaf_symbol_content":
            # If the target type is leaf_symbol_content, filter symbols that are leaves

            parent_of_symbol = {symbol.id: symbol.chunk.parent.id for symbol in codebase.symbols if symbol.chunk.parent}
            childs_of_symbol = defaultdict(list)
            for child, parent in parent_of_symbol.items():
                childs_of_symbol[parent].append(child)
            filtered_symbols = [
                elem for elem in filtered_symbols if not childs_of_symbol[elem.id] 
            ]
        elif target_type == "root_symbol_content":
            parent_of_symbol = {symbol.id: symbol.chunk.parent.id for symbol in codebase.symbols if symbol.chunk.parent}
            filtered_symbols = [
                elem for elem in filtered_symbols if not parent_of_symbol[elem.id]
            ]
        return filtered_symbols

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
