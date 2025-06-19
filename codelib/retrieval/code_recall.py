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
from os import PathLike

logger = logging.getLogger(__name__)


async def _perform_secondary_recall(
    codebase: Codebase,
    prompt: str,
    elements: List[Any],
    llm_results: List[Any],
    granularity: LLMMapFilterTargetType,
    llm_method: Callable,
    llm_call_mode: LLMCallMode = "traditional",
) -> Tuple[List[Any], List[Any]]:
    """
    Perform secondary recall using a more powerful model for refined filtering.
    
    Args:
        codebase: The codebase instance
        prompt: Original prompt
        elements: Initial filtered elements
        llm_results: Initial LLM results
        granularity: Target type for filtering
        llm_method: LLM method to use (filter or map)
        llm_call_mode: LLM call mode
        
    Returns:
        Tuple of (refined_elements, refined_llm_results)
    """
    if not elements:
        logger.info("No elements to perform secondary recall on")
        return elements, llm_results
    
    logger.info(f"Starting secondary recall on {len(elements)} elements")
    
    # Use a more powerful model for secondary recall
    secondary_model_id = os.environ.get("SECONDARY_MODEL_ID", "anthropic/claude-3.5-sonnet")
    
    # Store original environment variables
    original_mapfilter_model = os.environ.get("LLM_MAPFILTER_MODEL_ID")
    original_function_call_model = os.environ.get("LLM_FUNCTION_CALL_MODEL_ID")
    
    try:
        # Temporarily override environment variables for secondary recall
        os.environ["LLM_MAPFILTER_MODEL_ID"] = secondary_model_id
        os.environ["LLM_FUNCTION_CALL_MODEL_ID"] = secondary_model_id
        
        refined_prompt = f"""
Please perform a more precise analysis of the following code elements based on this requirement:

{prompt}

The elements below have already passed an initial filtering stage. Now, please apply stricter criteria to identify only the most relevant elements that truly match the requirement.

Focus on:
1. Exact semantic relevance to the requirement
2. Functional alignment with the specified criteria  
3. Quality and completeness of the match

Be more selective and only include elements that have high confidence of relevance.
"""
        
        file_paths = []
        if elements:
            for element in elements:
                if hasattr(element, 'file') and hasattr(element.file, 'path'):
                    file_paths.append(str(element.file.path))
                elif hasattr(element, 'path'):
                    file_paths.append(str(element.path))
                elif isinstance(element, str):
                    file_paths.append(element)
        
        unique_file_paths = list(dict.fromkeys(file_paths))
        
        if not unique_file_paths:
            logger.warning("No file paths extracted from elements for secondary recall")
            return elements, llm_results
        
        logger.info(f"Performing secondary recall on {len(unique_file_paths)} unique files")
        
        secondary_elements, secondary_llm_results = await llm_method(
            prompt=refined_prompt,
            target_type=granularity,
            subdirs_or_files=unique_file_paths,
            llm_call_mode=llm_call_mode,
        )
        
        logger.info(f"Secondary recall refined results from {len(elements)} to {len(secondary_elements)} elements")
        
        return secondary_elements, secondary_llm_results
        
    except Exception as e:
        logger.error(f"Secondary recall failed: {e}")
        return elements, llm_results
    finally:
        if original_mapfilter_model is not None:
            os.environ["LLM_MAPFILTER_MODEL_ID"] = original_mapfilter_model
        elif "LLM_MAPFILTER_MODEL_ID" in os.environ:
            del os.environ["LLM_MAPFILTER_MODEL_ID"]
            
        if original_function_call_model is not None:
            os.environ["LLM_FUNCTION_CALL_MODEL_ID"] = original_function_call_model
        elif "LLM_FUNCTION_CALL_MODEL_ID" in os.environ:
            del os.environ["LLM_FUNCTION_CALL_MODEL_ID"]


class RecallStrategy(Enum):
    FILTER_FILENAME_BY_LLM = "filter_filename_by_llm"
    FILTER_KEYWORD_BY_VECTOR = "filter_keywords_by_vector"
    FILTER_SYMBOL_BY_VECTOR = "filter_symbol_by_vector"
    FILTER_SYMBOL_BY_LLM = "filter_symbol_by_llm"
    FILTER_KEYWORD_BY_VECTOR_AND_LLM = "filter_keyword_by_vector_and_llm"
    FILTER_SYMBOL_BY_VECTOR_AND_LLM = "filter_symbol_by_vector_and_llm"
    ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM = "adaptive_filter_keyword_by_vector_and_llm"
    ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM = "adaptive_filter_symbol_by_vector_and_llm"


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


# Define generic types for code elements


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
        self, elements: List[Symbol], codebase: Codebase, subdirs_or_files: List[str]
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
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategy_map[strategy](topic_extractor=self.topic_extractor, llm_call_mode=self.llm_call_mode)


async def _multi_strategy_code_recall(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: str,
    llm_method: Callable,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    llm_call_mode: LLMCallMode = "traditional",
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    """
    Process code elements based on the specified prompt and mode.

    Args:
        prompt: The prompt for filtering or mapping
        subdirs_or_files: List of subdirectories or files to process
        granularity: The granularity level for code analysis
        mode: The search mode to use:
            - "fast": Uses FILTER_FILENAME_BY_LLM only
            - "balance": Uses FILTER_KEYWORD_BY_VECTOR + FILTER_FILENAME_BY_LLM
            - "precise": Uses full LLM filtering/mapping (default behavior)
            - "custom": Uses the provided custom_strategies
        custom_strategies: List of strategies to run in custom mode
        llm_method: The LLM method to call (codebase.llm_filter or codebase.llm_map)
        topic_extractor: Optional TopicExtractor instance for vector-based strategies
        llm_call_mode: Mode for LLM calls - "traditional", "function_call"

    Returns:
        Tuple of (elements, llm_results)
    """
    subdirs_or_files = [str(subdir).lstrip("/") for subdir in subdirs_or_files]
    elements = []
    llm_results = []
    extended_subdirs_or_files = set()
    all_llm_results = []
    elements = []
    llm_results = []
    extended_subdirs_or_files = set()
    all_llm_results = []

    # Determine which strategies to run based on the mode
    strategies_to_run: List[RecallStrategy] = []

    if mode == "fast":
        strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]
    elif mode == "balance":
        strategies_to_run = [
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR_AND_LLM,
            RecallStrategy.FILTER_FILENAME_BY_LLM,
        ]
    elif mode == "precise":
        # Precise mode: Use full LLM filtering/mapping (default behavior)
        logger.info(f"Using precise mode with full LLM processing")
        # In precise mode, we process all symbols directly
        filtered_symbols = codebase.symbols

        # Add file paths from symbols to extended_subdirs_or_files
        for symbol in filtered_symbols:
            for subdir in subdirs_or_files:
                if str(symbol.file.path).startswith(subdir):
                    extended_subdirs_or_files.add(str(symbol.file.path))
                    break
    elif mode == "custom" and custom_strategies:
        strategies_to_run = custom_strategies
    else:
        if mode == "custom" and not custom_strategies:
            logger.warning(
                "Custom mode specified but no custom strategies provided. Defaulting to fast mode."
            )
            strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]
        else:
            logger.warning(f"Unknown mode: {mode}. Defaulting to fast mode.")
            strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]

    # Execute each strategy in sequence
    if mode != "precise":
        strategy_factory = StrategyFactory(topic_extractor=topic_extractor, llm_call_mode=llm_call_mode)
        for strategy in strategies_to_run:
            try:
                strategy_executor = strategy_factory.create_strategy(strategy)
                file_paths, strategy_llm_results = await strategy_executor.execute(
                    codebase, prompt, subdirs_or_files
                )

                # Add file paths to the set of extended_subdirs_or_files
                extended_subdirs_or_files.update(file_paths)

                # Add LLM results if any
                if strategy_llm_results:
                    all_llm_results.extend(strategy_llm_results)

            except Exception as e:
                logger.error(f"Error executing strategy {strategy}: {e}")
                raise e

    # Convert set to list
    extended_subdirs_or_files = list(extended_subdirs_or_files)

    logger.debug(f"extended_subdirs_or_files size: {len(extended_subdirs_or_files)}")

    if not extended_subdirs_or_files:
        logger.info(f"No files found for '{prompt}'")
        return [], []

    logger.info(
        f"Processing {len(extended_subdirs_or_files)} files with {granularity} granularity"
    )

    # Check if the llm_method supports mode parameter (for SmartCodebase methods)
    logger.debug(f"Function call mode: {llm_call_mode}")
    elements, llm_results = await llm_method(
        prompt=prompt,
        target_type=granularity,
        subdirs_or_files=extended_subdirs_or_files,
        llm_call_mode=llm_call_mode,
    )

    # Secondary recall: further filter results if enabled and results exist
    if os.getenv("ENABLE_SECONDARY_RECALL", "false").lower() == "true":
        final_elements, final_llm_results = await _perform_secondary_recall(
            codebase=codebase,
            prompt=prompt,
            elements=elements,
            llm_results=llm_results,
            granularity=granularity,
            llm_method=llm_method,
            llm_call_mode=llm_call_mode,
        )
    else:
        final_elements = elements
        final_llm_results = llm_results
    
    return final_elements, final_llm_results


async def multi_strategy_code_mapping(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: Literal["fast", "balance", "precise", "custom"],
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    llm_call_mode: LLMCallMode = "traditional",
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        mode,
        codebase.llm_map,
        custom_strategies,
        topic_extractor,
        llm_call_mode,
    )


async def multi_strategy_code_filter(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: Literal["fast", "balance", "precise", "custom"],
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    llm_call_mode: LLMCallMode = "traditional",
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        mode,
        codebase.llm_filter,
        custom_strategies,
        topic_extractor,
        llm_call_mode,
    )
