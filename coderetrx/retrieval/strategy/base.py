"""
Base classes and utilities for code retrieval strategies.
"""

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
import asyncio
import logging
from abc import ABC, abstractmethod
from coderetrx.retrieval.smart_codebase import (
    SimilaritySearchTargetType,
    SmartCodebase as Codebase,
    CodeMapFilterResult,
    LLMCallMode,
    LLMMapFilterTargetType,
)
from coderetrx.static.codebase.codebase import CodeLine
import random
from ..topic_extractor import TopicExtractor
from coderetrx.static import (
    Symbol,
    Keyword,
    File,
    CodeElementTypeVar,
    Dependency,
    CodeElement,
)
from pathlib import Path
from pydantic import Field
from os import PathLike
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RecallStrategy(Enum):
    FILTER_FILENAME_BY_LLM = "filter_filename_by_llm"
    FILTER_KEYWORD_BY_VECTOR = "filter_keywords_by_vector"
    FILTER_SYMBOL_CONTENT_BY_VECTOR = "filter_symbol_by_vector"
    FILTER_SYMBOL_NAME_BY_LLM = "filter_symbol_by_llm"
    FILTER_DEPENDENCY_BY_LLM = "filter_dependency_by_llm"
    FILTER_KEYWORD_BY_VECTOR_AND_LLM = "filter_keyword_by_vector_and_llm"
    FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM = "filter_symbol_by_vector_and_llm"
    ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM = (
        "adaptive_filter_keyword_by_vector_and_llm"
    )
    ADAPTIVE_FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM = (
        "adaptive_filter_symbol_by_vector_and_llm"
    )
    FILTER_LINE_PER_SYMBOL_BY_VECTOR_AND_LLM = (
        "filter_line_per_symbol_by_vector_and_llm"
    )
    FILTER_LINE_PER_FILE_BY_VECTOR_AND_LLM = (
        "filter_line_per_file_by_vector_and_llm"
    )


class StrategyExecuteResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    file_paths: List[str] = Field(
        description="List of file paths returned by the strategy"
    )
    elements: List[CodeElement] = Field(
        description="List of code elements returned by the strategy"
    )
    llm_results: List[CodeMapFilterResult] = Field(
        description="List of LLM results returned by the strategy"
    )


def rel_path(a: PathLike, b: PathLike) -> str:
    return str(Path(a).relative_to(Path(b)))


def deduplicate_elements(elements: List[CodeElement]) -> List[CodeElement]:
    """
    Deduplicate elements by their id attribute.

    Args:
        elements: List of code elements to deduplicate

    Returns:
        List of deduplicated code elements, preserving order of first occurrence
    """
    if not elements:
        return elements

    seen_ids: Set[str] = set()
    deduplicated_elements: List[CodeElement] = []

    for element in elements:
        # Get element id, fallback to a generated id if not available
        element_id = getattr(element, "id", None)
        if element_id is None:
            # Generate fallback id based on element type and attributes
            if hasattr(element, "name") and hasattr(element, "file"):
                element_id = f"{element.name}_{element.file.path}"
            elif hasattr(element, "path"):
                element_id = str(element.path)
            else:
                # Last resort: use object id
                element_id = str(id(element))

        if element_id not in seen_ids:
            seen_ids.add(element_id)
            deduplicated_elements.append(element)
        else:
            logger.debug(f"Deduplicated element with id: {element_id}")

    if len(deduplicated_elements) < len(elements):
        logger.info(
            f"Deduplicated {len(elements) - len(deduplicated_elements)} elements, "
            f"keeping {len(deduplicated_elements)} unique elements"
        )

    return deduplicated_elements


class RecallStrategyExecutor(ABC):
    llm_call_mode: LLMCallMode

    def __init__(
        self,
        topic_extractor: Optional[TopicExtractor] = None,
        llm_call_mode: LLMCallMode = "traditional",
    ):
        self.topic_extractor = topic_extractor
        self.llm_call_mode = llm_call_mode

    def filter_elements_by_subdirs(
        self,
        codebase: Codebase,
        elements: List[CodeElement],
        subdirs_or_files: List[str],
    ) -> List[CodeElement]:
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
            elif isinstance(element, Dependency):
                if element.imported_by:
                    for imported_file in element.imported_by:
                        relative_path = str(imported_file.path)
                        if any(
                            relative_path.startswith(subdir)
                            for subdir in subdirs_or_files
                        ):
                            filtered_elements.append(element)
                            break

        return filtered_elements

    @abstractmethod
    async def execute(
        self,
        codebase: Any,
        prompt: str,
        subdirs_or_files: List[str],
        target_type: str = "symbol_content",
    ) -> StrategyExecuteResult:
        """
        Execute the recall strategy.

        Args:
            codebase: The codebase to search in
            prompt: The prompt for filtering or mapping
            subdirs_or_files: List of subdirectories or files to process
            target_type: The target_type level for retrieval (default: "symbol_content")

        Returns:
            StrategyExecuteResult containing file_paths, elements, and llm_results
        """
        pass


class FilterByLLMStrategy(RecallStrategyExecutor, Generic[CodeElementTypeVar], ABC):
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
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
        target_type: str = "symbol_content",
    ) -> StrategyExecuteResult:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy with target_type: {target_type}")
        try:
            elements, llm_results = await codebase.llm_filter(
                prompt,
                self.get_target_type(),
                subdirs_or_files,
                llm_call_mode=self.llm_call_mode,
            )
            elements = self.filter_elements_by_subdirs(
                codebase, elements, subdirs_or_files
            )
            file_paths = self.extract_file_paths(elements, codebase)
            return StrategyExecuteResult(
                file_paths=list(set(file_paths)),
                elements=deduplicate_elements(elements),
                llm_results=llm_results,
            )
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class FilterByVectorStrategy(RecallStrategyExecutor, Generic[CodeElementTypeVar], ABC):
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
        target_type: str = "symbol_content",
    ) -> StrategyExecuteResult:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy with target_type: {target_type}")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = (
                await self.topic_extractor.extract_topic(
                    input_text=prompt, llm_call_mode=self.llm_call_mode
                )
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
                    codebase, elements, subdirs_or_files
                )
                logger.info(
                    f"Filtered to {len(filtered_elements)} elements from specified subdirectories"
                )
                elements = filtered_elements

            file_paths = self.extract_file_paths(elements, codebase, subdirs_or_files)
            return StrategyExecuteResult(
                file_paths=file_paths,
                elements=deduplicate_elements(elements),
                llm_results=[],
            )
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
    ) -> List[SimilaritySearchTargetType]:
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
        codebase: Codebase,
        elements: List[Any],
        target_type: LLMMapFilterTargetType = "symbol_content",
        subdirs_or_files: List[str] = [],
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
        target_type: LLMMapFilterTargetType = "symbol_content",
    ) -> StrategyExecuteResult:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy with target_type: {target_type}")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = (
                await self.topic_extractor.extract_topic(
                    input_text=prompt, llm_call_mode=self.llm_call_mode
                )
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
                codebase, elements, subdirs_or_files
            )
            filtered_elements = self.filter_elements(
                codebase, elements, target_type, subdirs_or_files
            )
            logger.info(
                f"Filtered to {len(elements)} elements from specified subdirectories"
            )

            # Step 2: Pass the filtered results to the LLM for further refinement
            if not elements:
                logger.info(
                    f"No elements found by vector search, returning empty result"
                )
                return StrategyExecuteResult(file_paths=[], elements=[], llm_results=[])

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

            return StrategyExecuteResult(
                file_paths=list(set(file_paths)),
                elements=deduplicate_elements(refined_elements),
                llm_results=llm_results,
            )
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e

            # Extract topic from input text


class AdaptiveFilterByVectorAndLLMStrategy(RecallStrategyExecutor, ABC):
    """Base strategy to filter code elements using adaptive vector similarity search followed by LLM refinement."""

    def __init__(
        self,
        initial_limit: int = 50,
        threshold: float = 0.3,
        multiplier: float = 1.8,
        exit_probability: float = 0.9,
        tail_analysis_size: int = 10,
        max_iterations: int = 5,
        topic_extractor: Optional[TopicExtractor] = None,
        llm_call_mode: LLMCallMode = "traditional",
    ):
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
    ) -> SimilaritySearchTargetType:
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
        codebase: Codebase,
        elements: List[Any],
        target_type: LLMMapFilterTargetType = "symbol_content",
        subdirs_or_files: List[str] = [],
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

    async def adaptive_retrieval(
        self,
        codebase: Codebase,
        prompt: str,
        query_embedding_text: str,
        subdirs_or_files: List[str],
        target_type: LLMMapFilterTargetType = "symbol_content",
    ) -> Tuple[List[str], List[CodeElement], List[Any]]:
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

        logger.info(
            f"Starting optimized adaptive retrieval with initial limit: {running_limit}, total available: {total_available_chunks}"
        )

        accumulated_file_paths = set()
        previous_limit = 0
        iterations = 0
        consecutive_low_quality = 0
        last_success_ratio = 0.5

        vector_cache = {}
        accumulated_elements = []

        while (
            running_limit <= total_available_chunks and iterations < self.max_iterations
        ):
            print("iterations, running_limit: ", iterations, running_limit)
            iterations += 1
            logger.info(
                f"Adaptive retrieval iteration {iterations}: searching with limit {running_limit}"
            )

            cache_key = running_limit
            vector_results = await codebase.similarity_search(
                target_types=self.get_target_types_for_vector(),
                query=query_embedding_text,
                threshold=0,
                top_k=running_limit,
            )
            vector_cache[cache_key] = vector_results
            new_vector_results = vector_results[previous_limit:]

            logger.info(
                f"Processing {len(new_vector_results)} new vector results (total: {len(vector_results)})"
            )

            if not new_vector_results:
                logger.info(f"No new results found, stopping adaptive retrieval")
                break

            new_elements = self.filter_elements_by_subdirs(
                codebase, new_vector_results, subdirs_or_files
            )
            new_filtered_elements = self.filter_elements(
                codebase, new_elements, target_type, subdirs_or_files
            )
            accumulated_elements.extend(new_filtered_elements)

            accumulated_file_paths.update(
                self.collect_file_paths(
                    new_filtered_elements, codebase, subdirs_or_files
                )
            )

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
                tail_success_ratio = len(tail_elements) / min(
                    self.tail_analysis_size, len(new_filtered_elements)
                )
                logger.info(f"Estimated tail success ratio: {tail_success_ratio:.3f}")
            else:
                tail_success_ratio = 0

            quality_trend = tail_success_ratio / last_success_ratio
            logger.info(
                f"Quality trend: {quality_trend:.3f} (current: {tail_success_ratio:.3f}, last: {last_success_ratio:.3f})"
            )
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

                running_limit = min(
                    int(running_limit * expansion_factor), total_available_chunks
                )

                logger.info(
                    f"Expanding search from {old_limit} to {running_limit} (success ratio {tail_success_ratio:.3f} > threshold {self.threshold})"
                )
            else:
                consecutive_low_quality += 1
                logger.info(
                    f"Quality dropped (success ratio {tail_success_ratio:.3f}, last success ratio {last_success_ratio:.3f}), consecutive count: {consecutive_low_quality}"
                )

                if consecutive_low_quality >= 2:
                    logger.info(
                        f"Consecutive low quality count reached 2, stopping adaptive retrieval"
                    )
                    break

                rand = random.random()
                if rand < self.exit_probability:
                    break
                else:
                    running_limit = min(
                        int(running_limit * (self.multiplier - 1) * 0.2 + 1),
                        total_available_chunks,
                    )
        print("accumulated_file_paths size: ", len(accumulated_file_paths))
        return (
            list(accumulated_file_paths),
            deduplicate_elements(accumulated_elements),
            [],
        )

    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
        target_type: LLMMapFilterTargetType = "symbol_content",
    ) -> StrategyExecuteResult:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy with target_type: {target_type}")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = (
                await self.topic_extractor.extract_topic(
                    input_text=prompt, llm_call_mode=self.llm_call_mode
                )
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
            file_paths, elements, llm_results = await self.adaptive_retrieval(
                codebase, prompt, topic, subdirs_or_files, target_type
            )

            return StrategyExecuteResult(
                file_paths=file_paths,
                elements=elements,
                llm_results=llm_results,
            )
        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e


class BaseLLMBatchProcessor(ABC):
    """Base class for LLM batch processing with configurable settings."""
    
    def __init__(self, codebase: "Codebase"):
        self.codebase = codebase
        self.max_batch_size = codebase.settings.llm_mapfilter_max_batch_size
        self.max_batch_length = codebase.settings.llm_mapfilter_max_batch_length
    
    @abstractmethod
    def format_candidate(self, index: int, codeline: CodeLine) -> str:
        """Format a single candidate for display in the batch."""
        pass
    
    @abstractmethod
    def get_system_prompt(self, query: str) -> str:
        """Get the system prompt for the LLM."""
        pass
    
    @abstractmethod
    def get_user_prompt(self, candidates_str: str, query: str) -> str:
        """Get the user prompt for the LLM."""
        pass
    
    def get_function_definition(self) -> dict:
        """Get the function definition for LLM function calling."""
        return {
            "name": "select_relevant_lines",
            "description": "Select the relevant code lines based on the query criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of indices (1-based) of the relevant lines",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation for the selection",
                    },
                },
                "required": ["selected_indices", "reasoning"],
            },
        }
    
    def prepare_batch_text(self, line_candidates: List[CodeLine]) -> Tuple[List[str], int]:
        """
        Prepare batch text with length constraints.
        
        Returns:
            Tuple of (candidates_text_list, total_length)
        """
        candidates_text = []
        total_length = 0
        
        for i, codeline in enumerate(line_candidates):
            candidate_text = self.format_candidate(i, codeline)
            
            # Check if adding this candidate would exceed max_batch_length
            if total_length + len(candidate_text) > self.max_batch_length and candidates_text:
                break
                
            candidates_text.append(candidate_text)
            total_length += len(candidate_text) + 1  # +1 for newline
        
        return candidates_text, total_length
    
    async def process_candidate_batch(
        self, line_candidates: List[CodeLine], query: str, retry_cnt: int = 0
    ) -> List[CodeLine]:
        """Process a single batch of line candidates."""
        try:
            from coderetrx.utils.llm import call_llm_with_function_call
            
            candidates_text, _ = self.prepare_batch_text(line_candidates)
            candidates_str = "\n".join(candidates_text)
            
            # Get prompts
            system_prompt = self.get_system_prompt(query)
            user_prompt = self.get_user_prompt(candidates_str, query)
            function_definition = self.get_function_definition()
            
            function_args = await call_llm_with_function_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_definition=function_definition,
                model_ids=["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4"],
            )
            
            selected_indices = function_args.get("selected_indices", [])
            reasoning = function_args.get("reasoning", "")
            
            logger.debug(
                f"LLM selected {len(selected_indices)} lines from batch of {len(line_candidates)}. Reasoning: {reasoning}"
            )
            
            # Extract selected lines
            selected_lines = []
            for idx in selected_indices:
                if 1 <= idx <= len(line_candidates):
                    selected_lines.append(line_candidates[idx - 1])
            
            return selected_lines
            
        except Exception as e:
            logger.error(f"Error in LLM recall judgment batch: {e}")
            # Fallback: return first few candidates from this batch
            return [candidate for candidate in line_candidates[:10]]
    
    async def llm_recall_judgment(
        self, line_candidates: List[CodeLine], query: str
    ) -> List[CodeLine]:
        """
        Use LLM to judge and select the relevant lines from candidates.
        
        Args:
            line_candidates: List of CodeLine objects
            query: Original query for relevance judgment
            
        Returns:
            List of selected CodeLine objects
        """
        if not line_candidates:
            return []
        
        all_selected_lines = []
        
        # Create tasks for all batches
        tasks = []
        for i in range(0, len(line_candidates), self.max_batch_size):
            batch = line_candidates[i : i + self.max_batch_size]
            task = self.process_candidate_batch(batch, query)
            tasks.append(task)
        
        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results
        for batch_selected in batch_results:
            all_selected_lines.extend(batch_selected)
        
        if len(line_candidates) > self.max_batch_size:
            logger.debug(
                f"Processed {len(tasks)} batches concurrently: Selected {len(all_selected_lines)} total lines"
            )
        
        return all_selected_lines
