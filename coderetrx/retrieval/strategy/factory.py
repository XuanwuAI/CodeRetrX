from coderetrx.retrieval.strategy.base import RecallStrategy
from typing import Optional, Union, List

from coderetrx.retrieval.smart_codebase import LLMCallMode

from coderetrx.retrieval.topic_extractor import TopicExtractor
from coderetrx.retrieval.strategy.base import RecallStrategyExecutor
from coderetrx.retrieval.strategy.filter_filename_by_llm import (
    FilterFilenameByLLMStrategy,
)
from coderetrx.retrieval.strategy.filter_keyword_by_vector import (
    FilterKeywordByVectorStrategy,
)
from coderetrx.retrieval.strategy.filter_symbol_by_vector import (
    FilterSymbolByVectorStrategy,
)
from coderetrx.retrieval.strategy.filter_symbol_by_llm import FilterSymbolByLLMStrategy
from coderetrx.retrieval.strategy.filter_dependency_by_llm import (
    FilterDependencyByLLMStrategy,
)
from coderetrx.retrieval.strategy.filter_keyword_by_vector_and_llm import (
    FilterKeywordByVectorAndLLMStrategy,
)
from coderetrx.retrieval.strategy.filter_symbol_by_vector_and_llm import (
    FilterSymbolByVectorAndLLMStrategy,
)
from coderetrx.retrieval.strategy.adaptive_filter_symbol_by_vector_and_llm import (
    AdaptiveFilterSymbolByVectorAndLLMStrategy,
)
from coderetrx.retrieval.strategy.adaptive_filter_keyword_by_vector_and_llm import (
    AdaptiveFilterKeywordByVectorAndLLMStrategy,
)
from coderetrx.retrieval.strategy.filter_topk_line_by_vector_and_llm import (
    FilterTopkLineByVectorAndLLMStrategy,
)


class StrategyFactory:
    """Factory for creating strategy executors."""

    def __init__(
        self,
        topic_extractor: Optional[TopicExtractor] = None,
        llm_call_mode: LLMCallMode = "traditional",
    ):
        self.topic_extractor = topic_extractor
        self.llm_call_mode = llm_call_mode

    def create_strategy(self, strategy: RecallStrategy) -> RecallStrategyExecutor:
        """Create a strategy executor based on the strategy enum."""
        strategy_map = {
            RecallStrategy.FILTER_FILENAME_BY_LLM: FilterFilenameByLLMStrategy,
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR: FilterKeywordByVectorStrategy,
            RecallStrategy.FILTER_SYMBOL_BY_VECTOR: FilterSymbolByVectorStrategy,
            RecallStrategy.FILTER_SYMBOL_BY_LLM: FilterSymbolByLLMStrategy,
            RecallStrategy.FILTER_DEPENDENCY_BY_LLM: FilterDependencyByLLMStrategy,
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR_AND_LLM: FilterKeywordByVectorAndLLMStrategy,
            RecallStrategy.FILTER_SYMBOL_BY_VECTOR_AND_LLM: FilterSymbolByVectorAndLLMStrategy,
            RecallStrategy.ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM: AdaptiveFilterKeywordByVectorAndLLMStrategy,
            RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM: AdaptiveFilterSymbolByVectorAndLLMStrategy,
            RecallStrategy.FILTER_TOPK_LINE_BY_VECTOR_AND_LLM: FilterTopkLineByVectorAndLLMStrategy,
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategy_map[strategy](
            topic_extractor=self.topic_extractor, llm_call_mode=self.llm_call_mode
        )
