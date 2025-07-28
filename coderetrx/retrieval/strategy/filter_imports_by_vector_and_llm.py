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
from coderetrx.static import Keyword, Symbol, File, CodeChunk
from logging import getLogger

logger = getLogger(__name__)


class FilterImportsByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter symbols using vector similarity search followed by LLM refinement."""

    @override
    def get_strategy_name(self) -> str:
        return "FILTER_IMPORTS_BY_VECTOR_AND_LLM"

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["import"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "symbol_content"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

    @override
    def filter_elements(
        self,
        codebase: Codebase,
        elements: List[CodeChunk],
        target_type: LLMMapFilterTargetType = "dependency",
        subdirs_or_files: List[str] = [],
    ) -> List[Union[Keyword, Symbol, File, CodeChunk]]:
        filtered_symbols: List[CodeChunk] = []
        logger.info(f"Filtering {len(elements)} elements")
        for element in elements:
            if not isinstance(element, CodeChunk):
                logger.info(f"Skipping {element} because it is not a CodeChunk")
                continue
                # If subdirs_or_files is provided and codebase is available, filter by subdirs
            if subdirs_or_files and codebase:
                # Get the relative path from the codebase directory
                rpath = str(element.src.path)
                if any(rpath.startswith(subdir) for subdir in subdirs_or_files):
                    filtered_symbols.append(element)
            else:
                filtered_symbols.append(element)
        logger.info(f"Filtered to {len(filtered_symbols)} symbols")
        filtered_symbols = [
            elem for elem in filtered_symbols if elem.type == "import"
        ]
        logger.info(f"->>Filtered to {len(filtered_symbols)} symbols")
        return filtered_symbols

    @override
    def collect_file_paths(
        self,
        filtered_elements: List[CodeChunk],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        """Collect file paths from the filtered symbols."""
        file_paths = set()
        for symbol in filtered_elements:
            if isinstance(symbol, CodeChunk):
                file_path = str(symbol.src.path)
                if not subdirs_or_files or any(
                    file_path.startswith(subdir) for subdir in subdirs_or_files
                ):
                    file_paths.add(file_path)
        return list(file_paths)
