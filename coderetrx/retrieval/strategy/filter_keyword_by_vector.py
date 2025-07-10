"""
Strategy for filtering keywords using vector similarity search.
"""

from typing import List, override, Any
from .base import FilterByVectorStrategy
from ..smart_codebase import SmartCodebase as Codebase, SimilaritySearchTargetType
from coderetrx.static import Keyword


class FilterKeywordByVectorStrategy(FilterByVectorStrategy[Keyword]):
    """Strategy to filter keywords using vector similarity search."""

    name: str = "FILTER_KEYWORD_BY_VECTOR"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["keyword"]

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.keywords)

    @override
    def extract_file_paths(
        self, elements: List[Keyword], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[str]:
        referenced_paths = set()
        for item in elements:
            if isinstance(item, Keyword) and item.referenced_by:
                for ref_file in item.referenced_by:
                    if str(ref_file.path).startswith(tuple(subdirs_or_files)):
                        referenced_paths.add(str(ref_file.path))
        return list(referenced_paths)
