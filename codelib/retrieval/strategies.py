from enum import Enum
from typing import (
    Callable,
    Coroutine,
    List,
    Literal,
    Any,
    Tuple,
    Protocol,
    Dict,
    Set,
    Union,
    TypeVar,
    Generic,
    Optional,
    override,
)
import logging
import os
from abc import ABC, abstractmethod
from .smart_codebase import (
    LLMMapFilterTargetType,
    SimilaritySearchTargetType,
    SmartCodebase as Codebase,
    CodeMapFilterResult, LLMCallMode,
)
import random
from .topic_extractor import TopicExtractor
from codelib.static import Symbol, Keyword, File, CodeElement
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from os import PathLike
from attrs import define

logger = logging.getLogger(__name__)

class RecallStrategy(Enum):
    FILTER_FILENAME_BY_LLM = "filter_filename_by_llm"
    FILTER_KEYWORD_BY_VECTOR = "filter_keywords_by_vector"
    FILTER_SYMBOL_BY_VECTOR = "filter_symbol_by_vector"
    FILTER_SYMBOL_BY_LLM = "filter_symbol_by_llm"
    FILTER_KEYWORD_BY_VECTOR_AND_LLM = "filter_keyword_by_vector_and_llm"
    FILTER_SYMBOL_BY_VECTOR_AND_LLM = "filter_symbol_by_vector_and_llm"
    ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM = "adaptive_filter_keyword_by_vector_and_llm"
    ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM = "adaptive_filter_symbol_by_vector_and_llm"
    INTELLIGENT_FILTER = "intelligent_filter"


class CodeRecallSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )
    llm_secondary_recall_model_id: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model ID for secondary recall",
    )
    llm_primary_recall_model_id: Optional[str] = Field(
        default=None,
        description="Model ID for primary recall. If not provided will use the SmartCodebase default",
    )
    llm_selector_strategy_model_id: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model ID for determining the best recall strategy based on the prompt when llm_selector_strategy is enabled",
    )
    llm_call_mode: LLMCallMode = Field(
        default="function_call",
        description="Mode for LLM calls - 'traditional' or 'function_call'",
    )


async def _determine_strategy_by_llm(
    prompt: str,
    model_id: Optional[str] = None,
) -> List[RecallStrategy]:
    """
    Use LLM to determine the best recall strategies based on the prompt.
    
    Args:
        prompt: The user's prompt/query
        model_id: The model ID to use for strategy determination
        
    Returns:
        List of selected RecallStrategy
    """
    from codelib.utils.llm import call_llm_with_function_call
    
    # Define the function for strategy determination
    function_definition = {
        "name": "determine_search_strategy",
        "description": "Determine the best code search strategies based on the user query",
        "parameters": {
            "type": "object",
            "properties": {
                "element_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["FILENAME", "SYMBOL", "LINE"]
                    },
                    "description": "The types of code elements to search for (can select one or multiple)",
                    "minItems": 1,
                    "maxItems": 3
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for why these element types were chosen"
                }
            },
            "required": ["element_types", "reason"]
        }
    }
    
    system_prompt = """Analyze this code search prompt and determine the most effective search method:

Consider what type of code element would best match the search criteria. PREFER TO SELECT ONLY ONE TYPE unless you are genuinely uncertain about which single approach would work best.

FILENAME: Use when the search criteria can be identified primarily by file characteristics:
- Specific file extensions or patterns
- File naming conventions
- Configuration files
- Documentation files
- Files in specific directories
- Files with particular path patterns
- etc.

SYMBOL: Use when the search criteria involve specific named code constructs:
- Function names or method names
- Class names or interface names
- Module names or dependency names
- When you know the exact name of what you're looking for
- Shallow search based on identifiers
- etc.

LINE: Use when the search criteria require understanding file content and behavior:
- Complex patterns or logic flows
- Code that performs specific operations
- Algorithm implementations or patterns
- Business logic analysis
- Error handling patterns
- Content-based search within files
- When function/class names alone are not sufficient
- Deep content analysis
- Line-level vector recall and LLM judgment
- etc.

SELECTION GUIDELINES:
- Try to identify the SINGLE MOST APPROPRIATE type first
- Only select multiple types if the search criteria genuinely spans multiple categories and you cannot determine which single approach would be most effective
- When in doubt between two types, consider which one would capture the most relevant results for the specific query

Search prompt to analyze:
{prompt}"""

    user_prompt = f"""Analyze the following code search query and determine what types of code elements to search for:

<query>
{prompt}
</query>

Call the determine_search_strategy function with your analysis. 
IMPORTANT: Try to select only ONE element type if possible. Only select multiple types if you genuinely cannot determine which single approach would be most effective."""

    try:
        # Use the same model list pattern as other function calls in the codebase
        settings = CodeRecallSettings()
        effective_model_id = model_id or settings.llm_selector_strategy_model_id
        model_ids = [effective_model_id, "anthropic/claude-3.7-sonnet"]
        
        function_args = await call_llm_with_function_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            function_definition=function_definition,
            model_ids=model_ids,
        )
        
        element_types = function_args.get("element_types", [])
        reason = function_args.get("reason", "")
        
        # Ensure all element types are uppercase
        element_types = [et.upper() for et in element_types if isinstance(et, str)]
        
        logger.info(f"LLM determined element types: {element_types}. Reason: {reason}")
        
        # Map the response to RecallStrategy enum list
        strategies = []
        for element_type in element_types:
            if element_type == "FILENAME":
                strategies.append(RecallStrategy.FILTER_FILENAME_BY_LLM)
            elif element_type == "LINE":
                strategies.append(RecallStrategy.INTELLIGENT_FILTER)
            elif element_type == "SYMBOL":
                strategies.append(RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM)
            else:
                logger.warning(f"LLM returned invalid element type: {element_type}. Skipping.")
        
        if not strategies:
            logger.warning(f"No valid strategies determined. Defaulting to SYMBOL recall")
            return [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM]
        
        return strategies
            
    except Exception as e:
        logger.error(f"Error determining strategy by LLM: {e}. Defaulting to SYMBOL recall")
        return [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM]


def rel_path(a: PathLike, b: PathLike) -> str:
    return str(Path(a).relative_to(Path(b)))


class RecallStrategyExecutor(ABC):
    """Base class for recall strategy executors."""

    def __init__(self, topic_extractor: Optional[TopicExtractor] = None, llm_call_mode: LLMCallMode = "traditional"):
        self.topic_extractor = topic_extractor
        self.llm_call_mode = llm_call_mode

    def filter_elements_by_subdirs(
        self, elements: List[Any], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[Any]:
        """Filter elements to only include those from the specified subdirectories."""
        if not subdirs_or_files:
            return elements  # No filtering if no subdirs specified

        filtered_elements = []
        for element in elements:
            if isinstance(element, Symbol):
                file_path = str(element.file.path)
                if any(file_path.startswith(subdir) for subdir in subdirs_or_files):
                    filtered_elements.append(element)
            elif isinstance(element, Keyword):
                for ref_file in element.referenced_by:
                    relative_path = str(ref_file.path)
                    if any(
                        relative_path.startswith(subdir) for subdir in subdirs_or_files
                    ):
                        filtered_elements.append(element)
                        break
            elif isinstance(element, File):
                file_path = str(element.path)
                if any(file_path.startswith(subdir) for subdir in subdirs_or_files):
                    filtered_elements.append(element)

        return filtered_elements

    @abstractmethod
    async def execute(
        self, codebase: Any, prompt: str, subdirs_or_files: List[str]
    ) -> Tuple[List[str], List[Any]]:
        """
        Execute the recall strategy.

        Args:
            codebase: The codebase to search in
            prompt: The prompt for filtering or mapping
            subdirs_or_files: List of subdirectories or files to process

        Returns:
            Tuple of (file_paths, llm_results)
        """
        pass


class FilterByLLMStrategy(RecallStrategyExecutor, Generic[CodeElement], ABC):
    """Base strategy for filtering elements using LLM."""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy for logging purposes."""
        pass

    @abstractmethod
    def get_target_type(self) -> LLMMapFilterTargetType:
        """Return the target type for LLM filtering."""
        pass

    @abstractmethod
    def extract_file_paths(
        self, elements: List[CodeElement], codebase: Codebase
    ) -> List[str]:
        """Extract file paths from the filtered elements."""
        pass

    async def execute(
        self, codebase: Codebase, prompt: str, subdirs_or_files: List[str]
    ) -> Tuple[List[str], List[Any]]:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        try:
            elements, llm_results = await codebase.llm_filter(
                prompt, self.get_target_type(), subdirs_or_files, llm_call_mode=self.llm_call_mode
            )
            elements = self.filter_elements_by_subdirs(
                elements, codebase, subdirs_or_files
            )
            file_paths = self.extract_file_paths(elements, codebase)
            return list(set(file_paths)), llm_results
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class FilterByVectorStrategy(RecallStrategyExecutor, Generic[CodeElement], ABC):
    """Base strategy for filtering elements using vector similarity search."""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy for logging purposes."""
        pass

    @abstractmethod
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        """Return the target types for similarity search."""
        pass

    @abstractmethod
    def get_collection_size(self, codebase: Codebase) -> int:
        """Return the size of the collection to search in."""
        pass

    @abstractmethod
    def extract_file_paths(
        self, elements: List[Any], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[str]:
        """Extract file paths from the filtered elements."""
        pass

    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = (
                await self.topic_extractor.extract_topic(input_text=prompt, llm_call_mode=self.llm_call_mode)
                if self.topic_extractor
                else prompt
            )

            if not topic:
                logger.warning(
                    "Topic extraction failed, using original prompt for vector similarity search"
                )
                topic = prompt
            else:
                logger.info(
                    f"Using extracted topic '{topic}' for vector similarity search"
                )

            # Perform vector similarity search with the extracted topic
            elements = await codebase.similarity_search(
                target_types=self.get_target_types_for_vector(),
                query=topic,
                threshold=0,
                top_k=self.get_collection_size(codebase) // 10,
            )
            logger.info(
                f"Found {len(elements)} elements by similarity search query '{topic}'"
            )

            # Filter elements by subdirectories
            if subdirs_or_files:
                filtered_elements = self.filter_elements_by_subdirs(
                    elements, codebase, subdirs_or_files
                )
                logger.info(
                    f"Filtered to {len(filtered_elements)} elements from specified subdirectories"
                )
                elements = filtered_elements

            file_paths = self.extract_file_paths(elements, codebase, subdirs_or_files)
            return file_paths, []
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class FilterByVectorAndLLMStrategy(RecallStrategyExecutor, ABC):
    """Base strategy to filter code elements using vector similarity search followed by LLM refinement."""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy for logging purposes."""
        pass

    @abstractmethod
    def get_target_types_for_vector(
        self,
    ) -> List[Literal["symbol_name", "symbol_content", "keyword"]]:
        """Return the target types for similarity search."""
        pass

    @abstractmethod
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        """Return the target type for LLM filtering."""
        pass

    @abstractmethod
    def get_collection_size(self, codebase: Codebase) -> int:
        """Return the size of the collection to search in."""
        pass

    @abstractmethod
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        """
        Filter and convert elements to the expected type for additional_code_elements.

        Args:
            elements: List of elements to filter
            subdirs_or_files: List of subdirectories or files to filter by
            codebase: The codebase instance for resolving paths

        Returns:
            Filtered list of elements
        """
        pass

    @abstractmethod
    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        """Collect file paths from the filtered elements."""
        pass

    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = (
                await self.topic_extractor.extract_topic(input_text=prompt, llm_call_mode=self.llm_call_mode)
                if self.topic_extractor
                else prompt
            )

            if not topic:
                logger.warning(
                    "Topic extraction failed, using original prompt for vector similarity search"
                )
                topic = prompt
            else:
                logger.info(
                    f"Using extracted topic '{topic}' for vector similarity search"
                )

            # Step 1: Perform initial filtering using vector-based method with the extracted topic
            elements = await codebase.similarity_search(
                target_types=self.get_target_types_for_vector(),
                query=topic,
                threshold=0,
                top_k=self.get_collection_size(codebase) // 10,
            )
            logger.info(
                f"Found {len(elements)} elements by similarity search query '{topic}'"
            )

            # Filter elements by subdirectories and types
            elements = self.filter_elements_by_subdirs(
                elements, codebase, subdirs_or_files
            )
            filtered_elements = self.filter_elements(
                elements, subdirs_or_files, codebase
            )
            logger.info(
                f"Filtered to {len(elements)} elements from specified subdirectories"
            )

            # Step 2: Pass the filtered results to the LLM for further refinement
            if not elements:
                logger.info(
                    f"No elements found by vector search, returning empty result"
                )
                return [], []

            # Use LLM to filter the elements
            refined_elements, llm_results = await codebase.llm_filter(
                prompt=prompt,
                target_type=self.get_target_type_for_llm(),
                subdirs_or_files=[],  # Empty list as we're providing elements directly
                additional_code_elements=filtered_elements,
                llm_call_mode=self.llm_call_mode,
            )

            logger.info(f"LLM refined results to {len(refined_elements)} elements")

            # Collect file paths from the filtered elements
            file_paths = self.collect_file_paths(
                refined_elements, codebase, subdirs_or_files
            )

            return list(set(file_paths)), llm_results
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class AdaptiveFilterByVectorAndLLMStrategy(RecallStrategyExecutor, ABC):
    """Base strategy to filter code elements using adaptive vector similarity search followed by LLM refinement."""

    def __init__(self, initial_limit: int = 50, 
                 threshold: float = 0.3, multiplier: float = 1.8, exit_probability: float = 0.9, 
                 tail_analysis_size: int = 10, max_iterations: int = 5, 
                 topic_extractor: Optional[TopicExtractor] = None, llm_call_mode: LLMCallMode = "traditional"):
        """
        Initialize the adaptive filtering strategy with configurable parameters.

        Args:
            initial_limit: Initial number of elements to retrieve in first iteration (default: 50)
            threshold: Base propotion of successful tail results (default: 0.3)
            multiplier: Factor to increase threshold between iterations (default: 1.8)
            exit_probability: Probability threshold for early exit (default: 0.9)
            tail_analysis_size: Number of results to analyze for early exit decision (default: 10)
            max_iterations: Maximum number of search iterations to perform (default: 5)
            topic_extractor: Optional TopicExtractor instance for vector-based strategies
            llm_call_mode: Mode for LLM calls - "traditional", "function_call"
        """
        super().__init__(topic_extractor=topic_extractor, llm_call_mode=llm_call_mode)
        self.initial_limit = initial_limit
        self.threshold = threshold
        self.multiplier = multiplier
        self.exit_probability = exit_probability
        self.tail_analysis_size = tail_analysis_size
        self.max_iterations = max_iterations

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy for logging purposes."""
        pass

    @abstractmethod
    def get_target_types_for_vector(
        self,
    ) -> List[Literal["symbol_name", "symbol_content", "keyword"]]:
        """Return the target types for similarity search."""
        pass

    @abstractmethod
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        """Return the target type for LLM filtering."""
        pass

    @abstractmethod
    def get_collection_size(self, codebase: Codebase) -> int:
        """Return the size of the collection to search in."""
        pass

    @abstractmethod
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        """
        Filter and convert elements to the expected type for additional_code_elements.

        Args:
            elements: List of elements to filter
            subdirs_or_files: List of subdirectories or files to filter by
            codebase: The codebase instance for resolving paths

        Returns:
            Filtered list of elements
        """
        pass

    @abstractmethod
    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        """Collect file paths from the filtered elements."""
        pass

    async def adaptive_retrieval(self, codebase: Codebase, prompt: str, query_embedding_text: str, subdirs_or_files: List[str]) -> Tuple[List[str], List[Any]]:
        """
        Adaptive retrieval algorithm with reduced LLM calls and smarter early exit.

        Args:
            codebase: The codebase to search in
            prompt: Original prompt for LLM filtering
            query_embedding_text: Text to use for vector similarity search
            subdirs_or_files: List of subdirectories or files to filter by

        Returns:
            Tuple of (file_paths, refined_elements)
        """
        total_available_chunks = self.get_collection_size(codebase)
        running_limit = min(self.initial_limit, total_available_chunks)

        logger.info(f"Starting optimized adaptive retrieval with initial limit: {running_limit}, total available: {total_available_chunks}")

        accumulated_file_paths = set()
        previous_limit = 0
        iterations = 0
        consecutive_low_quality = 0
        last_success_ratio = 0.5

        vector_cache = {}

        while running_limit <= total_available_chunks and iterations < self.max_iterations:
            print("iterations, running_limit: ", iterations, running_limit)
            iterations += 1
            logger.info(f"Adaptive retrieval iteration {iterations}: searching with limit {running_limit}")

            cache_key = running_limit
            vector_results = await codebase.similarity_search(
                target_types=self.get_target_types_for_vector(),
                query=query_embedding_text,
                threshold=0,
                top_k=running_limit,
            )
            vector_cache[cache_key] = vector_results
            new_vector_results = vector_results[previous_limit:]

            logger.info(f"Processing {len(new_vector_results)} new vector results (total: {len(vector_results)})")

            if not new_vector_results:
                logger.info(f"No new results found, stopping adaptive retrieval")
                break

            new_elements = self.filter_elements_by_subdirs(new_vector_results, codebase, subdirs_or_files)
            new_filtered_elements = self.filter_elements(new_elements, subdirs_or_files, codebase)

            accumulated_file_paths.update(self.collect_file_paths(new_filtered_elements, codebase, subdirs_or_files))

            if not new_filtered_elements:
                logger.info(f"No filtered elements found, stopping adaptive retrieval")
                break

            # Doing tail analysis
            tail_start = max(0, len(new_filtered_elements) - self.tail_analysis_size)
            tail_elements, _ = await codebase.llm_filter(
                prompt=prompt,
                target_type=self.get_target_type_for_llm(),
                subdirs_or_files=[],
                additional_code_elements=new_filtered_elements[tail_start:],
                llm_call_mode=self.llm_call_mode,
            )

            if tail_elements:
                tail_success_ratio = len(tail_elements) / min(self.tail_analysis_size, len(new_filtered_elements))
                logger.info(f"Estimated tail success ratio: {tail_success_ratio:.3f}")
            else:
                tail_success_ratio = 0

            quality_trend = tail_success_ratio / last_success_ratio
            logger.info(f"Quality trend: {quality_trend:.3f} (current: {tail_success_ratio:.3f}, last: {last_success_ratio:.3f})")
            if tail_success_ratio >= self.threshold and quality_trend >= 0.3:
                consecutive_low_quality = 0
                previous_limit = running_limit
                old_limit = running_limit

                if tail_success_ratio > 0.7:
                    expansion_factor = self.multiplier * 1.5
                elif tail_success_ratio > 0.3:
                    expansion_factor = self.multiplier
                else:
                    expansion_factor = (self.multiplier - 1) * 0.5 + 1

                running_limit = min(int(running_limit * expansion_factor), total_available_chunks)

                logger.info(f"Expanding search from {old_limit} to {running_limit} (success ratio {tail_success_ratio:.3f} > threshold {self.threshold})")
            else:
                consecutive_low_quality += 1
                logger.info(f"Quality dropped (success ratio {tail_success_ratio:.3f}, last success ratio {last_success_ratio:.3f}), consecutive count: {consecutive_low_quality}")

                if consecutive_low_quality >= 2:
                    logger.info(f"Consecutive low quality count reached 2, stopping adaptive retrieval")
                    break

                rand = random.random()
                if (rand < self.exit_probability):
                    break
                else:
                    running_limit = min(int(running_limit * (self.multiplier - 1) * 0.2 + 1), total_available_chunks)
        print("accumulated_file_paths size: ", len(accumulated_file_paths))
        return list(accumulated_file_paths), []

    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = (
                await self.topic_extractor.extract_topic(input_text=prompt, llm_call_mode=self.llm_call_mode)
                if self.topic_extractor
                else prompt
            )

            if not topic:
                logger.warning(
                    "Topic extraction failed, using original prompt for vector similarity search"
                )
                topic = prompt
            else:
                logger.info(
                    f"Using extracted topic '{topic}' for vector similarity search"
                )

            # Perform adaptive retrieval (includes both vector search and LLM filtering)
            file_paths, llm_results = await self.adaptive_retrieval(codebase, prompt, topic, subdirs_or_files)

            return file_paths, llm_results
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class FilterFilenameByLLMStrategy(FilterByLLMStrategy[File]):
    """Strategy to filter filenames using LLM."""

    @override
    def get_strategy_name(self) -> str:
        return "FILTER_FILENAME_BY_LLM"

    @override
    def get_target_type(self) -> LLMMapFilterTargetType:
        return "file_name"

    @override
    def extract_file_paths(self, elements: List[File], codebase: Codebase) -> List[str]:
        return [str(file.path) for file in elements]

    @override
    async def execute(
        self, codebase: Codebase, prompt: str, subdirs_or_files: List[str]
    ) -> Tuple[List[str], List[Any]]:
        prompt = f"""
        A file with this path is highly likely to contain content that matches the following criteria:
        <content_criterias>
        {prompt}
        </content_criterias>
        <note>
        The objective of this requirement is to preliminarily identify files based on their paths that are likely to meet specific content criteria.
        Files with matching paths will proceed to a deeper analysis in the content filter (content_criterias) at a later stage (not in this run).  
        </note>
        """
        return await super().execute(codebase, prompt, subdirs_or_files)


class FilterSymbolByLLMStrategy(FilterByLLMStrategy[Symbol]):
    """Strategy to filter symbols using LLM."""

    name: str = "FILTER_SYMBOL_BY_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_type(self) -> LLMMapFilterTargetType:
        return "symbol_name"

    @override
    def extract_file_paths(
        self, elements: List[Symbol], codebase: Codebase
    ) -> List[str]:
        return [str(symbol.file.path) for symbol in elements]


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
        self, elements: List[Symbol], codebase:  Codebase, subdirs_or_files: List[str]
    ) -> List[str]:
        file_paths = []
        for symbol in elements:
            if isinstance(symbol, Symbol):
                file_path = str(symbol.file.path)
                if file_path.startswith(tuple(subdirs_or_files)):
                    file_paths.append(file_path)
        return list(set(file_paths))


class FilterKeywordByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter keywords using vector similarity search followed by LLM refinement."""

    name: str = "FILTER_KEYWORD_BY_VECTOR_AND_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["keyword"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "keyword"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.keywords)

    @override
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        keyword_elements: List[Union[Keyword, Symbol, File]] = []
        for element in elements:
            if isinstance(element, Keyword):
                # If subdirs_or_files is provided and codebase is available, filter by subdirs
                if subdirs_or_files and codebase:
                    for ref_file in element.referenced_by:
                        if any(
                            str(ref_file.path).startswith(subdir)
                            for subdir in subdirs_or_files
                        ):
                            keyword_elements.append(element)
                            break
                else:
                    keyword_elements.append(element)
        return keyword_elements

    @override
    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        referenced_paths = set()
        for keyword in filtered_elements:
            if isinstance(keyword, Keyword) and keyword.referenced_by:
                for ref_file in keyword.referenced_by:
                    if str(ref_file.path).startswith(tuple(subdirs_or_files)):
                        referenced_paths.add(str(ref_file.path))
        return list(referenced_paths)

    @override
    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        prompt = f"""
        A code chunk containing the specified keywords is highly likely to meet the following criteria:
        <content_criteria>
        {prompt}
        </content_criteria>
        <note>
        The objective of this requirement is to preliminarily filter files that are likely to meet specific content criteria based on the keywords they contain. 
        Files with matching keywords will proceed to deeper analysis in the content filter (content_criteria) at a later stage (not in this run). 
        </note>
        """
        return await super().execute(codebase, prompt, subdirs_or_files)


class AdaptiveFilterKeywordByVectorAndLLMStrategy(AdaptiveFilterByVectorAndLLMStrategy):
    """Strategy to filter keywords using adaptive vector similarity search followed by LLM refinement."""

    name: str = "ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["keyword"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "keyword"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.keywords)

    @override
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        keyword_elements: List[Union[Keyword, Symbol, File]] = []
        for element in elements:
            if isinstance(element, Keyword):
                if subdirs_or_files and codebase:
                    for ref_file in element.referenced_by:
                        if any(
                            str(ref_file.path).startswith(subdir)
                            for subdir in subdirs_or_files
                        ):
                            keyword_elements.append(element)
                            break
                else:
                    keyword_elements.append(element)
        return keyword_elements

    @override
    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        referenced_paths = set()
        for keyword in filtered_elements:
            if isinstance(keyword, Keyword) and keyword.referenced_by:
                for ref_file in keyword.referenced_by:
                    if str(ref_file.path).startswith(tuple(subdirs_or_files)):
                        referenced_paths.add(str(ref_file.path))
        return list(referenced_paths)

    @override
    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        prompt = f"""
        A code chunk containing the specified keywords is highly likely to meet the following criteria:
        <content_criteria>
        {prompt}
        </content_criteria>
        <note>
        The objective of this requirement is to preliminarily filter files that are likely to meet specific content criteria based on the keywords they contain. 
        Files with matching keywords will proceed to deeper analysis in the content filter (content_criteria) at a later stage (not in this run). 
        </note>
        """
        return await super().execute(codebase, prompt, subdirs_or_files)


class FilterSymbolByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter symbols using vector similarity search followed by LLM refinement."""

    name: str = "FILTER_SYMBOL_BY_VECTOR_AND_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["symbol_name"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "symbol_name"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

    @override
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        symbol_elements: List[Union[Keyword, Symbol, File]] = []
        for element in elements:
            if isinstance(element, Symbol):
                # If subdirs_or_files is provided and codebase is available, filter by subdirs
                if subdirs_or_files and codebase:
                    # Get the relative path from the codebase directory
                    rpath = str(element.file.path)
                    if any(rpath.startswith(subdir) for subdir in subdirs_or_files):
                        symbol_elements.append(element)
                else:
                    symbol_elements.append(element)
        return symbol_elements

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
    ) -> Tuple[List[str], List[Any]]:
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
        return await super().execute(codebase, prompt, subdirs_or_files)


class AdaptiveFilterSymbolByVectorAndLLMStrategy(AdaptiveFilterByVectorAndLLMStrategy):
    """Strategy to filter symbols using adaptive vector similarity search followed by LLM refinement."""

    name: str = "ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["symbol_name"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "symbol_name"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

    @override
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        symbol_elements: List[Union[Keyword, Symbol, File]] = []
        for element in elements:
            if isinstance(element, Symbol):
                # If subdirs_or_files is provided and codebase is available, filter by subdirs
                if subdirs_or_files and codebase:
                    # Get the relative path from the codebase directory
                    rpath = str(element.file.path)
                    if any(rpath.startswith(subdir) for subdir in subdirs_or_files):
                        symbol_elements.append(element)
                else:
                    symbol_elements.append(element)
        return symbol_elements

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
    ) -> Tuple[List[str], List[Any]]:
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
        return await super().execute(codebase, prompt, subdirs_or_files)


@define
class SymbolStruct:
    """Structure to represent a symbol and its metadata for intelligent filtering."""
    vectors: List[Any]       # Embeddings of the symbol lines
    symbol: Symbol           # The original symbol object
    lines: List[str]         # Individual lines of the symbol
    symbol_id: str           # Unique identifier for the symbol
    file_path: str           # File path for the symbol


class IntelligentFilterStrategy(RecallStrategyExecutor):
    """
    Intelligent filtering strategy that performs line-level vector recall within symbols,
    then uses LLM to judge and select the most relevant lines across all symbols.
    """
    
    def __init__(self, 
                 top_k_per_symbol: int = 5,
                 max_queries: int = 20,
                 topic_extractor: Optional[TopicExtractor] = None, 
                 llm_call_mode: LLMCallMode = "traditional"):
        """
        Initialize the intelligent filtering strategy.
        
        Args:
            top_k_per_symbol: Number of top lines to recall per symbol (default: 5)
            max_queries: Maximum number of LLM queries allowed (default: 100)
            topic_extractor: Optional TopicExtractor instance
            llm_call_mode: Mode for LLM calls
        """
        super().__init__(topic_extractor=topic_extractor, llm_call_mode=llm_call_mode)
        self.top_k_per_symbol = top_k_per_symbol
        self.max_queries = max_queries
    
    def get_strategy_name(self) -> str:
        return "INTELLIGENT_FILTER"
    
    async def _generate_line_embeddings(self, lines: List[str]) -> List[Any]:
        """Generate embeddings for individual lines using cached embedder."""
        from codelib.utils.embedding import cached_embedder
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        
        filtered_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 3]
        if not filtered_lines:
            return []
        
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(min=30, max=600),
            retry=retry_if_exception_type(Exception),
        )
        async def _embed_lines_with_retry():
            """Embed lines using cached embedder."""
            try:
                # Use cached embedder instead of direct create_documents_embedding
                embeddings = await cached_embedder.aembed_documents(filtered_lines)
                return embeddings
            except Exception as e:
                logger.warning(f"Line embedding batch failed, will retry: {str(e)}")
                raise
        
        try:
            embeddings = await _embed_lines_with_retry()
            logger.debug(f"Successfully generated {len(embeddings)} embeddings for {len(filtered_lines)} lines")
            return embeddings
        except Exception as e:
            logger.error(f"All retry attempts failed for line embeddings: {e}")
            return []
    
    async def _vector_recall_topk(self, func_struct: SymbolStruct, query: str, k: int) -> List[Tuple[str, int, float]]:
        """
        Perform vector-based top-k recall for lines within a function using in-memory computation.
        
        Args:
            func_struct: Function structure containing vectors and lines
            query: Query text for similarity search
            k: Number of top lines to return
            
        Returns:
            List of tuples (line_content, line_index, similarity_score)
        """
        if not func_struct.vectors or not func_struct.lines:
            return []
        
        return await self._manual_vector_recall(func_struct, query, k)
    
    async def _manual_vector_recall(self, func_struct: SymbolStruct, query: str, k: int) -> List[Tuple[str, int, float]]:
        """
        Fallback manual vector recall using numpy operations.
        """
        try:
            from codelib.utils.embedding import create_documents_embedding
            import numpy as np
            
            # Generate query embedding using robust utilities
            query_embedding = create_documents_embedding([query])
            if not query_embedding:
                return []
            
            query_vec = np.array(query_embedding[0])
            line_vecs = np.array(func_struct.vectors)
            
            # Calculate cosine similarities with proper normalization
            query_norm = query_vec / np.linalg.norm(query_vec)
            line_norms = line_vecs / np.linalg.norm(line_vecs, axis=1, keepdims=True)
            
            similarities = np.dot(line_norms, query_norm)
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            filtered_lines = [line.strip() for line in func_struct.lines if line.strip() and len(line.strip()) > 3]
            
            for idx in top_indices:
                if idx < len(filtered_lines):
                    # Find original line index
                    filtered_line = filtered_lines[idx]
                    original_idx = -1
                    for i, original_line in enumerate(func_struct.lines):
                        if original_line.strip() == filtered_line:
                            original_idx = i
                            break

                    if original_idx >= 0:
                        normalized_score = (similarities[idx] + 1) / 2
                        results.append((filtered_line, original_idx, normalized_score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in manual vector recall for symbol {func_struct.symbol_id}: {e}")
            return []
    
    async def _prepare_symbol_structures(self, symbols: List[Symbol], subdirs_or_files: List[str]) -> List[SymbolStruct]:
        """
        Prepare symbol structures with line-level embeddings.
        
        Args:
            symbols: List of symbols to process
            subdirs_or_files: List of subdirectories or files to filter by
            
        Returns:
            List of SymbolStruct objects with embeddings
        """
        from tqdm import tqdm
        
        symbol_structs = []
        
        # Progress bar for symbol processing
        with tqdm(total=len(symbols), desc="Processing symbols for intelligent filtering", unit="symbol") as pbar:
            for symbol in symbols:
                try:
                    # Filter by subdirectories if specified
                    file_path = str(symbol.file.path)
                    if subdirs_or_files and not any(file_path.startswith(subdir) for subdir in subdirs_or_files):
                        pbar.update(1)
                        continue
                    
                    # Get symbol lines
                    lines = symbol.chunk.lines()
                    if not lines:
                        pbar.update(1)
                        continue
                    
                    # Generate embeddings for lines using existing utilities
                    vectors = await self._generate_line_embeddings(lines)
                    if not vectors:
                        pbar.update(1)
                        continue
                    
                    # Create function structure
                    symbol_struct = SymbolStruct(
                        vectors=vectors,
                        symbol=symbol,
                        lines=lines,
                        symbol_id=symbol.id or f"{symbol.name}_{symbol.file.path}",
                        file_path=file_path
                    )
                    
                    symbol_structs.append(symbol_struct)
                    
                except Exception as e:
                    logger.error(f"Error preparing symbol structure for {symbol.name}: {e}")
                finally:
                    pbar.update(1)
        
        logger.info(f"Prepared {len(symbol_structs)} symbol structures for intelligent filtering")
        return symbol_structs
    
    async def _llm_recall_judgment(self, line_candidates: List[Tuple[str, str, str]], query: str) -> List[str]:
        """
        Use LLM to judge and select the most relevant lines from candidates.

        Args:
            line_candidates: List of (line_content, symbol_name, file_path) tuples
            query: Original query for relevance judgment
            
        Returns:
            List of selected line contents
        """
        if not line_candidates:
            return []
        
        max_batch_size = 30
        all_selected_lines = []
        
        # Process in batches
        for i in range(0, len(line_candidates), max_batch_size):
            batch = line_candidates[i:i + max_batch_size]
            batch_selected = await self._process_candidate_batch(batch, query)
            all_selected_lines.extend(batch_selected)
            
            if len(line_candidates) > max_batch_size:
                logger.info(f"Processed batch {i//max_batch_size + 1}: Selected {len(batch_selected)} lines from {len(batch)} candidates")
        
        return all_selected_lines
    
    async def _process_candidate_batch(self, line_candidates: List[Tuple[str, str, str]], query: str) -> List[str]:
        """Process a single batch of line candidates."""
        try:
            from codelib.utils.llm import call_llm_with_function_call
            
            # Prepare candidates text
            candidates_text = []
            for i, (line, symbol_name, file_path) in enumerate(line_candidates):
                candidates_text.append(f"{i+1}. [{symbol_name} in {file_path}] {line.strip()}")
            
            candidates_str = "\n".join(candidates_text)
            
            # Define function for line selection
            function_definition = {
                "name": "select_relevant_lines",
                "description": "Select the most relevant code lines based on the query criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of indices (1-based) of the most relevant lines"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation for the selection"
                        }
                    },
                    "required": ["selected_indices", "reasoning"]
                }
            }
            
            system_prompt = f"""You are analyzing code lines to find the most relevant ones for a specific query.

Query: {query}

Select the most relevant lines from the candidates below. Focus on:
1. Direct relevance to the query requirements
2. Functional significance and completeness
3. Quality and clarity of the code

Be selective and choose only the lines that truly match the query criteria."""
            
            user_prompt = f"""Here are the candidate code lines:

{candidates_str}

Select the most relevant lines for the query: "{query}"

Call the select_relevant_lines function with your analysis."""
            
            function_args = await call_llm_with_function_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_definition=function_definition,
                model_ids=["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4"],
            )
            
            selected_indices = function_args.get("selected_indices", [])
            reasoning = function_args.get("reasoning", "")
            
            logger.info(f"LLM selected {len(selected_indices)} lines from batch of {len(line_candidates)}. Reasoning: {reasoning}")
            
            # Extract selected lines
            selected_lines = []
            for idx in selected_indices:
                if 1 <= idx <= len(line_candidates):
                    selected_lines.append(line_candidates[idx-1][0])
            
            return selected_lines
            
        except Exception as e:
            logger.error(f"Error in LLM recall judgment batch: {e}")
            # Fallback: return first few candidates from this batch
            return [candidate[0] for candidate in line_candidates[:3]]
    
    async def execute(self, codebase: Codebase, prompt: str, subdirs_or_files: List[str]) -> Tuple[List[str], List[Any]]:
        """
        Execute the intelligent filter strategy with optimized batch processing.
        
        Args:
            codebase: The codebase to search in
            prompt: The prompt for filtering
            subdirs_or_files: List of subdirectories or files to process
            
        Returns:
            Tuple of (file_paths, llm_results)
        """
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        
        try:
            # Extract topic for vector search
            topic = (
                await self.topic_extractor.extract_topic(input_text=prompt, llm_call_mode=self.llm_call_mode)
                if self.topic_extractor
                else prompt
            )
            
            if not topic:
                logger.warning("Topic extraction failed, using original prompt")
                topic = prompt
            else:
                logger.info(f"Using extracted topic '{topic}' for intelligent filtering")
            
            # Step 1: Get all symbols and prepare structures with embeddings
            print("Entering Step 1 - Preparing symbol structures")
            all_symbols = codebase.symbols
            filtered_symbols = self.filter_elements_by_subdirs(all_symbols, codebase, subdirs_or_files)

            logger.info(f"Processing {len(filtered_symbols)} symbols for intelligent filtering")
            
            symbol_structs = await self._prepare_symbol_structures(filtered_symbols, subdirs_or_files)
            
            if not symbol_structs:
                logger.info("No symbols found for intelligent filtering")
                return [], []
            
            # Step 2: Perform vector-based recall for each symbol and sort by similarity
            all_recalled_lines = []
            print("Entering Step 2 - Vector recall for symbols")
            for symbol_struct in symbol_structs:
                recalled_lines = await self._vector_recall_topk(
                    symbol_struct, topic, self.top_k_per_symbol
                )
                
                if recalled_lines:
                    # Store line candidates with metadata
                    for line_content, line_idx, score in recalled_lines:
                        all_recalled_lines.append((
                            line_content, 
                            symbol_struct.symbol.name,
                            symbol_struct.file_path,
                            symbol_struct.symbol_id,
                            score
                        ))
            
            logger.info(f"Collected {len(all_recalled_lines)} line candidates from vector recall")
            
            if not all_recalled_lines:
                logger.info("No lines recalled from vector search")
                return [], []
            
            # Sort by vector similarity score (descending)
            all_recalled_lines.sort(key=lambda x: x[4], reverse=True)
            logger.info("Sorted line candidates by vector similarity score")
            
            # Step 3: Optimized LLM processing with dynamic batch evaluation
            print("Entering Step 3 - Dynamic batch processing")
            selected_file_paths = set()
            recalled_symbols = set()
            batch_size = 120
            
            # Process candidates in dynamic batches
            current_index = 0
            batch_count = 0
            query_count = 0
            
            # Progress bar for LLM queries
            from tqdm import tqdm
            with tqdm(total=self.max_queries, desc="LLM batch queries", unit="query") as pbar:
                while current_index < len(all_recalled_lines) and query_count < self.max_queries:
                    # Collect next batch of valid candidates
                    batch_candidates = []
                    
                    while len(batch_candidates) < batch_size and current_index < len(all_recalled_lines):
                        line_content, symbol_name, file_path, symbol_id, score = all_recalled_lines[current_index]
                        current_index += 1
                        
                        # Skip if symbol already recalled
                        if symbol_id in recalled_symbols:
                            continue
                        
                        # Skip if file path already selected
                        if file_path in selected_file_paths:
                            continue
                        
                        batch_candidates.append((line_content, symbol_name, file_path, symbol_id, score))
                    
                    if not batch_candidates:
                        break
                    
                    batch_count += 1
                    
                    # Prepare batch data for LLM judgment
                    batch_data = [(line_content, symbol_name, file_path) for line_content, symbol_name, file_path, symbol_id, score in batch_candidates]
                    
                    # Get LLM judgment on this batch
                    selected_lines = await self._llm_recall_judgment(batch_data, prompt)
                    query_count += 1
                    pbar.update(1)  # Update progress bar
                    
                    # Mark selected symbols as recalled
                    for selected_line in selected_lines:
                        for line_content, symbol_name, file_path, symbol_id, score in batch_candidates:
                            if line_content == selected_line and symbol_id not in recalled_symbols:
                                recalled_symbols.add(symbol_id)
                                selected_file_paths.add(file_path)
                                logger.info(f"Selected symbol {symbol_name} from {file_path} (score: {score:.3f})")
                                break
                    
                    logger.info(f"Processed batch {batch_count}: Selected {len(selected_lines)} symbols from {len(batch_candidates)} candidates")
            
            file_paths = list(selected_file_paths)
            print(f"Total selected file paths: {len(file_paths)}")
            logger.info(f"Intelligent filtering completed. Selected {len(file_paths)} files from {len(recalled_symbols)} symbols.")
            
            return file_paths, []
            
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class StrategyFactory:
    """Factory for creating strategy executors."""

    def __init__(self, topic_extractor: Optional[TopicExtractor] = None, llm_call_mode: LLMCallMode = "traditional"):
        self.topic_extractor = topic_extractor
        self.llm_call_mode = llm_call_mode

    def create_strategy(self, strategy: RecallStrategy) -> RecallStrategyExecutor:
        """Create a strategy executor based on the strategy enum."""
        strategy_map = {
            RecallStrategy.FILTER_FILENAME_BY_LLM: FilterFilenameByLLMStrategy,
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR: FilterKeywordByVectorStrategy,
            RecallStrategy.FILTER_SYMBOL_BY_VECTOR: FilterSymbolByVectorStrategy,
            RecallStrategy.FILTER_SYMBOL_BY_LLM: FilterSymbolByLLMStrategy,
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR_AND_LLM: FilterKeywordByVectorAndLLMStrategy,
            RecallStrategy.FILTER_SYMBOL_BY_VECTOR_AND_LLM: FilterSymbolByVectorAndLLMStrategy,
            RecallStrategy.ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM: AdaptiveFilterKeywordByVectorAndLLMStrategy,
            RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM: AdaptiveFilterSymbolByVectorAndLLMStrategy,
            RecallStrategy.INTELLIGENT_FILTER: IntelligentFilterStrategy,
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategy_map[strategy](topic_extractor=self.topic_extractor, llm_call_mode=self.llm_call_mode)