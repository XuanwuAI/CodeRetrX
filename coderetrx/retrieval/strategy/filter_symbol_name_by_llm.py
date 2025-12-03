"""
Strategy for filtering symbols using LLM.
"""

from typing import List
from .base import FilterByLLMStrategy
from ..smart_codebase import SmartCodebase as Codebase, LLMMapFilterTargetType
from coderetrx.static import Symbol


class FilterSymbolNameByLLMStrategy(FilterByLLMStrategy[Symbol]):
    """Strategy to filter symbols using LLM."""

    name: str = "FILTER_SYMBOL_NAME_BY_LLM"

    
    def get_strategy_name(self) -> str:
        return self.name

    
    def get_target_type(self) -> LLMMapFilterTargetType:
        return "symbol_name"

    
    def extract_file_paths(
        self, elements: List[Symbol], codebase: Codebase
    ) -> List[str]:
        return [str(symbol.file.path) for symbol in elements]
