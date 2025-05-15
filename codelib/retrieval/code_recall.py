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
)
import logging
import os
from abc import ABC, abstractmethod
from .smart_codebase import (
    LLMMapFilterTargetType,
    SimilaritySearchTargetType,
    SmartCodebase as Codebase,
    CodeMapFilterResult,
)
from .topic_extractor import TopicExtractor
from codelib.static import Symbol, Keyword, File, CodeElement
from pathlib import Path
from os import PathLike

logger = logging.getLogger(__name__)


class RecallStrategy(Enum):
    FILTER_FILENAME_BY_LLM = "filter_filename_by_llm"
    FILTER_KEYWORD_BY_VECTOR = "filter_keywords_by_vector"
    FILTER_SYMBOL_BY_VECTOR = "filter_symbol_by_vector"
    FILTER_SYMBOL_BY_LLM = "filter_symbol_by_llm"
    FILTER_KEYWORD_BY_VECTOR_AND_LLM = "filter_keyword_by_vector_and_llm"
    FILTER_SYMBOL_BY_VECTOR_AND_LLM = "filter_symbol_by_vector_and_llm"


def rel_path(a: PathLike, b: PathLike) -> str:
    return str(Path(a).relative_to(Path(b)))


class RecallStrategyExecutor(ABC):
    """Base class for recall strategy executors."""

    def filter_elements_by_subdirs(
        self, elements: List[Any], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[Any]:
        """Filter elements to only include those from the specified subdirectories."""
        if not subdirs_or_files:
            return elements  # No filtering if no subdirs specified

        filtered_elements = []
        for element in elements:
            if isinstance(element, Symbol):
                file_path = rel_path(element.file.path, codebase.dir)
                if any(file_path.startswith(subdir) for subdir in subdirs_or_files):
                    filtered_elements.append(element)
            elif isinstance(element, Keyword):
                for path in element.referenced_by:
                    rel_path = rel_path(path, codebase.dir)
                    if any(rel_path.startswith(subdir) for subdir in subdirs_or_files):
                        filtered_elements.append(element)
                        break
            elif isinstance(element, File):
                file_path = rel_path(element.path, codebase.dir)
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
                prompt, self.get_target_type(), subdirs_or_files
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
        topic_extractor: TopicExtractor,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = await topic_extractor.extract_topic(input_text=prompt)

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
            elements = codebase.similarity_search(
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
        topic_extractor: TopicExtractor,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> Tuple[List[str], List[Any]]:
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy")
        try:
            # Extract topic from input text before performing vector similarity search
            topic = await topic_extractor.extract_topic(input_text=prompt)

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
            elements = codebase.similarity_search(
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


class FilterFilenameByLLMStrategy(FilterByLLMStrategy[File]):
    """Strategy to filter filenames using LLM."""

    def get_strategy_name(self) -> str:
        return "FILTER_FILENAME_BY_LLM"

    def get_target_type(self) -> LLMMapFilterTargetType:
        return "file_name"

    def extract_file_paths(self, elements: List[File], codebase: Codebase) -> List[str]:
        return [rel_path(file.path, codebase.dir) for file in elements]

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

    def get_strategy_name(self) -> str:
        return self.name

    def get_target_type(self) -> LLMMapFilterTargetType:
        return "symbol_name"

    def extract_file_paths(
        self, elements: List[Symbol], codebase: Codebase
    ) -> List[str]:
        return [rel_path(symbol.file.path, codebase.dir) for symbol in elements]


class FilterKeywordByVectorStrategy(FilterByVectorStrategy[Keyword]):
    """Strategy to filter keywords using vector similarity search."""

    name: str = "FILTER_KEYWORD_BY_VECTOR"

    def get_strategy_name(self) -> str:
        return self.name

    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["keyword"]

    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.keywords)

    def extract_file_paths(
        self, elements: List[Keyword], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[str]:
        referenced_paths = set()
        for item in elements:
            if isinstance(item, Keyword) and item.referenced_by:
                for path in item.referenced_by:
                    if rel_path(path.path, codebase.dir).startswith(
                        tuple(subdirs_or_files)
                    ):
                        referenced_paths.add(str(path.path))
        return list(referenced_paths)


class FilterSymbolByVectorStrategy(FilterByVectorStrategy[Symbol]):
    """Strategy to filter symbols using vector similarity search."""

    name: str = "FILTER_SYMBOL_BY_VECTOR"

    def get_strategy_name(self) -> str:
        return self.name

    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["symbol_name"]

    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

    def extract_file_paths(
        self, elements: List[Symbol], codebase: Codebase, subdirs_or_files: List[str]
    ) -> List[str]:
        file_paths = []
        for symbol in elements:
            if isinstance(symbol, Symbol):
                file_path = rel_path(symbol.file.path, codebase.dir)
                if file_path.startswith(tuple(subdirs_or_files)):
                    file_paths.append(file_path)
        return list(set(file_paths))


class FilterKeywordByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter keywords using vector similarity search followed by LLM refinement."""

    name: str = "FILTER_KEYWORD_BY_VECTOR_AND_LLM"

    def get_strategy_name(self) -> str:
        return self.name

    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["keyword"]

    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "keyword"

    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.keywords)

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
                    for path in element.referenced_by:
                        # Get the relative path from the codebase directory
                        rpath = rel_path(path.path, codebase.dir)
                        if any(rpath.startswith(subdir) for subdir in subdirs_or_files):
                            keyword_elements.append(element)
                            break
                else:
                    keyword_elements.append(element)
        return keyword_elements

    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        referenced_paths = set()
        for keyword in filtered_elements:
            if isinstance(keyword, Keyword) and keyword.referenced_by:
                for path in keyword.referenced_by:
                    if rel_path(path.path, codebase.dir).startswith(
                        tuple(subdirs_or_files)
                    ):
                        referenced_paths.add(str(path.path))
        return list(referenced_paths)

    async def execute(
        self,
        codebase: Codebase,
        topic_extractor: TopicExtractor,
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
        return await super().execute(
            codebase, topic_extractor, prompt, subdirs_or_files
        )


class FilterSymbolByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter symbols using vector similarity search followed by LLM refinement."""

    name: str = "FILTER_SYMBOL_BY_VECTOR_AND_LLM"

    def get_strategy_name(self) -> str:
        return self.name

    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["symbol_name"]

    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "symbol_name"

    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.symbols)

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
                    rpath = rel_path(element.file.path, codebase.dir)
                    if any(rpath.startswith(subdir) for subdir in subdirs_or_files):
                        symbol_elements.append(element)
                else:
                    symbol_elements.append(element)
        return symbol_elements

    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        file_paths = []
        for symbol in filtered_elements:
            if isinstance(symbol, Symbol):
                file_path = rel_path(symbol.file.path, codebase.dir)
                if file_path.startswith(tuple(subdirs_or_files)):
                    file_paths.append(file_path)
        return file_paths

    async def execute(
        self,
        codebase: Codebase,
        topic_extractor: TopicExtractor,
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
        return await super().execute(
            codebase, topic_extractor, prompt, subdirs_or_files
        )


class StrategyFactory:
    """Factory for creating strategy executors."""

    @staticmethod
    def create_strategy(strategy: RecallStrategy) -> RecallStrategyExecutor:
        """Create a strategy executor based on the strategy enum."""
        strategy_map = {
            RecallStrategy.FILTER_FILENAME_BY_LLM: FilterFilenameByLLMStrategy(),
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR: FilterKeywordByVectorStrategy(),
            RecallStrategy.FILTER_SYMBOL_BY_VECTOR: FilterSymbolByVectorStrategy(),
            RecallStrategy.FILTER_SYMBOL_BY_LLM: FilterSymbolByLLMStrategy(),
            RecallStrategy.FILTER_KEYWORD_BY_VECTOR_AND_LLM: FilterKeywordByVectorAndLLMStrategy(),
            RecallStrategy.FILTER_SYMBOL_BY_VECTOR_AND_LLM: FilterSymbolByVectorAndLLMStrategy(),
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategy_map[strategy]


async def _multi_strategy_code_recall(
    codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: str,
    llm_method: Callable,
    custom_strategies: List[RecallStrategy] = [],
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
                if str(symbol.file).startswith(str(codebase.dir / subdir)):
                    extended_subdirs_or_files.add(
                        str(symbol.file.relative_to(codebase.dir))
                    )
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
        for strategy in strategies_to_run:
            try:
                strategy_executor = StrategyFactory.create_strategy(strategy)
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

    if not extended_subdirs_or_files:
        logger.info(f"No files found for '{prompt}'")
        return [], []

    logger.info(
        f"Processing {len(extended_subdirs_or_files)} files with {granularity} granularity"
    )
    elements, llm_results = await llm_method(
        prompt=prompt,
        target_type=granularity,
        subdirs_or_files=extended_subdirs_or_files,
    )

    return elements, llm_results


async def multi_strategy_code_mapping(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: Literal["fast", "balance", "precise", "custom"],
    custom_strategies: List[RecallStrategy] = [],
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        mode,
        codebase.llm_map,
        custom_strategies,
    )


async def multi_strategy_code_filter(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: Literal["fast", "balance", "precise", "custom"],
    custom_strategies: List[RecallStrategy] = [],
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        mode,
        codebase.llm_filter,
        custom_strategies,
    )
