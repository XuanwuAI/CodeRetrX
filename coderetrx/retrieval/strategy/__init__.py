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
from .filter_symbol_name_by_llm import FilterSymbolNameByLLMStrategy
from .filter_dependency_by_llm import FilterDependencyByLLMStrategy
from .filter_keyword_by_vector import FilterKeywordByVectorStrategy
from .filter_symbol_content_by_vector import FilterSymbolContentByVectorStrategy
from .filter_keyword_by_vector_and_llm import FilterKeywordByVectorAndLLMStrategy
from .filter_symbol_content_by_vector_and_llm import FilterSymbolContentByVectorAndLLMStrategy
from .adaptive_filter_keyword_by_vector_and_llm import (
    AdaptiveFilterKeywordByVectorAndLLMStrategy,
)
from .adaptive_filter_symbol_content_by_vector_and_llm import (
    AdaptiveFilterSymbolContentByVectorAndLLMStrategy,
)
from .filter_line_per_symbol_by_vector_and_llm import FilterLinePerSymbolByVectorAndLLMStrategy

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
    "FilterSymbolNameByLLMStrategy",
    "FilterDependencyByLLMStrategy",
    "FilterKeywordByVectorStrategy",
    "FilterSymbolContentByVectorStrategy",
    "FilterKeywordByVectorAndLLMStrategy",
    "FilterSymbolContentByVectorAndLLMStrategy",
    "AdaptiveFilterKeywordByVectorAndLLMStrategy",
    "AdaptiveFilterSymbolContentByVectorAndLLMStrategy",
    "FilterLinePerSymbolByVectorAndLLMStrategy",
    # Factory and Utilities
    "StrategyFactory",
    "deduplicate_elements",
    "rel_path",
]
