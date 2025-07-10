"""
Strategy package for code retrieval strategies.

This package contains individual strategy implementations split from the original strategies.py file.
Each strategy is now in its own module for better organization and maintainability.
"""

from .base import (
    RecallStrategy,
    StrategyExecuteResult,
    RecallStrategyExecutor,
    FilterByLLMStrategy,
    FilterByVectorStrategy,
    FilterByVectorAndLLMStrategy,
    AdaptiveFilterByVectorAndLLMStrategy,
    deduplicate_elements,
    rel_path,
)
from .factory import StrategyFactory

from .filter_filename_by_llm import FilterFilenameByLLMStrategy
from .filter_symbol_by_llm import FilterSymbolByLLMStrategy
from .filter_dependency_by_llm import FilterDependencyByLLMStrategy
from .filter_keyword_by_vector import FilterKeywordByVectorStrategy
from .filter_symbol_by_vector import FilterSymbolByVectorStrategy
from .filter_keyword_by_vector_and_llm import FilterKeywordByVectorAndLLMStrategy
from .filter_symbol_by_vector_and_llm import FilterSymbolByVectorAndLLMStrategy
from .adaptive_filter_keyword_by_vector_and_llm import AdaptiveFilterKeywordByVectorAndLLMStrategy
from .adaptive_filter_symbol_by_vector_and_llm import AdaptiveFilterSymbolByVectorAndLLMStrategy
from .filter_topk_line_by_vector_and_llm import FilterTopkLineByVectorAndLLMStrategy

__all__ = [
    # Enums and Models
    "RecallStrategy",
    "StrategyExecuteResult",
    
    # Base Classes
    "RecallStrategyExecutor",
    "FilterByLLMStrategy",
    "FilterByVectorStrategy", 
    "FilterByVectorAndLLMStrategy",
    "AdaptiveFilterByVectorAndLLMStrategy",
    
    # Concrete Strategy Implementations
    "FilterFilenameByLLMStrategy",
    "FilterSymbolByLLMStrategy",
    "FilterDependencyByLLMStrategy",
    "FilterKeywordByVectorStrategy",
    "FilterSymbolByVectorStrategy",
    "FilterKeywordByVectorAndLLMStrategy",
    "FilterSymbolByVectorAndLLMStrategy",
    "AdaptiveFilterKeywordByVectorAndLLMStrategy",
    "AdaptiveFilterSymbolByVectorAndLLMStrategy",
    "FilterTopkLineByVectorAndLLMStrategy",
    
    # Factory and Utilities
    "StrategyFactory",
    "deduplicate_elements",
    "rel_path",
]
