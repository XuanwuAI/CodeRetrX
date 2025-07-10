"""
Strategy for filtering symbols using vector similarity search.
"""

from typing import List, override
from .base import FilterByVectorStrategy
from ..smart_codebase import SmartCodebase as Codebase, SimilaritySearchTargetType
from coderetrx.static import Symbol


class FilterSymbolByVectorStrategy(FilterByVectorStrategy[Symbol]):
    """Strategy to filter symbols using vector similarity search."""

    name: str = "FILTER_SYMBOL_BY_VECTOR"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["symbol_name"]

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

    @override
    def extract_file_paths(
        self, elements: List[Symbol], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[str]:
        file_paths = []
        for symbol in elements:
            if isinstance(symbol, Symbol):
                file_path = str(symbol.file.path)
                if file_path.startswith(tuple(subdirs_or_files)):
                    file_paths.append(file_path)
        return list(set(file_paths))
