from typing import (
    Callable,
    List,
    Literal,
    Any,
    Tuple,
    Optional,
)
import logging
from .smart_codebase import (
    LLMMapFilterTargetType,
    SmartCodebase as Codebase,
    CodeMapFilterResult, LLMCallMode,
)
from .topic_extractor import TopicExtractor
from coderetrx.static import Symbol, Keyword, File
from .strategies import (
    RecallStrategy,
    CodeRecallSettings,
    _determine_strategy_by_llm,
    StrategyFactory,
    StrategyExecuteResult,
)

logger = logging.getLogger(__name__)


async def _perform_secondary_recall(
    codebase: Codebase,
    prompt: str,
    elements: List[Any],
    llm_results: List[Any],
    granularity: LLMMapFilterTargetType,
    llm_method: Callable,
    llm_call_mode: LLMCallMode = "traditional",
    model_id: Optional[str] = None,
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
        model_id: Stronger model ID to use for secondary recall
        
    Returns:
        Tuple of (refined_elements, refined_llm_results)
    """
    if not elements:
        logger.info("No elements to perform secondary recall on")
        return elements, llm_results
    
    logger.info(f"Starting secondary recall on {len(elements)} elements")
    
    # Use a more powerful model for secondary recall
    secondary_model_id = model_id or "anthropic/claude-3.5-sonnet"
    
    try:
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
            prompt=prompt,
            target_type=granularity,
            subdirs_or_files=unique_file_paths,
            llm_call_mode=llm_call_mode,
            model_id=secondary_model_id,
        )
        
        logger.info(f"Secondary recall refined results from {len(elements)} to {len(secondary_elements)} elements")
        
        return secondary_elements, secondary_llm_results
        
    except Exception as e:
        logger.error(f"Secondary recall failed: {e}")
        return elements, llm_results


async def _multi_strategy_code_recall(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    mode: str,
    llm_method: Callable,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    """
    Process code elements based on the specified prompt and mode.

    Args:
        prompt: The prompt for filtering or mapping
        subdirs_or_files: List of subdirectories or files to process
        granularity: The granularity level for code analysis
        mode: The search mode to use:
            - "filename": Uses FILTER_FILENAME_BY_LLM only
            - "symbol": Uses ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM strategy
            - "line": Uses INTELLIGENT_FILTER strategy with line-level vector recall
            - "dependency": Uses FILTER_DEPENDENCY_BY_LLM strategy
            - "auto": Uses LLM to determine best strategy based on prompt (chooses from filename, symbol, line)
            - "precise": Uses full LLM filtering/mapping (default behavior)
            - "custom": Uses the provided custom_strategies
        custom_strategies: List of strategies to run in custom mode
        llm_method: The LLM method to call (codebase.llm_filter or codebase.llm_map)
        topic_extractor: Optional TopicExtractor instance for vector-based strategies
        settings: Optional CodeRecallSettings for configuration
        enable_secondary_recall: Whether to perform secondary recall after primary recall

    Returns:
        Tuple of (elements, llm_results)
    """
    if settings is None:
        settings = CodeRecallSettings()

    subdirs_or_files = [str(subdir).lstrip("/") for subdir in subdirs_or_files]
    elements = []
    llm_results = []
    extended_subdirs_or_files = set()
    all_llm_results = []
    # Determine which strategies to run based on the mode
    strategies_to_run: List[RecallStrategy] = []
    if mode == "filename":
        strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]
    elif mode == "symbol":
        strategies_to_run = [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM]
    elif mode == "line":
        strategies_to_run = [RecallStrategy.FILTER_TOPK_LINE_BY_VECTOR_AND_LLM]
    elif mode == "dependency":
        strategies_to_run = [RecallStrategy.FILTER_DEPENDENCY_BY_LLM]
    elif mode == "auto":
        strategies = await _determine_strategy_by_llm(
            prompt=prompt,
            model_id=settings.llm_selector_strategy_model_id,
        )
        strategy_names = [strategy.value for strategy in strategies]
        print(f"LLM smart strategies selected: {strategy_names}")
        strategies_to_run = strategies
    elif mode == "keyword":
        strategies_to_run = [RecallStrategy.ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM]
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
        strategy_factory = StrategyFactory(topic_extractor=topic_extractor, llm_call_mode=settings.llm_call_mode)
        for strategy in strategies_to_run:
            try:
                strategy_executor = strategy_factory.create_strategy(strategy)
                strategy_result = await strategy_executor.execute(
                    codebase, prompt, subdirs_or_files
                )

                # Add file paths to the set of extended_subdirs_or_files
                extended_subdirs_or_files.update(strategy_result.file_paths)

                # Add LLM results if any
                if strategy_result.llm_results:
                    all_llm_results.extend(strategy_result.llm_results)

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
    logger.debug(f"Function call mode: {settings.llm_call_mode}")
    
    # Use relaxed prompt for primary recall if secondary recall is enabled
    primary_prompt = prompt
    if enable_secondary_recall:
        primary_prompt = f"""Please be more lenient and inclusive in your search criteria. Include items that might be related even if they don't perfectly match.

{prompt}

Note: This is the primary filtering stage - we prefer to include potentially relevant items rather than miss them. A more precise filtering will be applied in the secondary stage."""
        logger.info("Using relaxed prompt for primary recall (secondary recall enabled)")
    
    elements, llm_results = await llm_method(
        prompt=primary_prompt,
        target_type=granularity,
        subdirs_or_files=extended_subdirs_or_files,
        llm_call_mode=settings.llm_call_mode,
        model_id=settings.llm_primary_recall_model_id,
    )

    # Secondary recall: further filter results if enabled and results exist
    if enable_secondary_recall:
        logger.info(f"Performing secondary recall on {len(elements)} elements")
        final_elements, final_llm_results = await _perform_secondary_recall(
            codebase=codebase,
            prompt=prompt,
            elements=elements,
            llm_results=llm_results,
            granularity=granularity,
            llm_method=llm_method,
            llm_call_mode=settings.llm_call_mode,
            model_id=settings.llm_secondary_recall_model_id,
        )
    else:
        final_elements = elements
        final_llm_results = llm_results
    
    return final_elements, final_llm_results


async def llm_traversal_mapping(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        "precise",
        codebase.llm_map,
        custom_strategies,
        topic_extractor,
        settings,
        enable_secondary_recall
    )

async def coderetrx_mapping(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    coarse_recall_strategy: Literal["filename", "symbol", "line", "auto", "custom", "dependency"],
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        coarse_recall_strategy,
        codebase.llm_map,
        custom_strategies,
        topic_extractor,
        settings,
        enable_secondary_recall
    )

async def llm_traversal_filter(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        "precise",
        codebase.llm_filter,
        custom_strategies,
        topic_extractor,
        settings,
        enable_secondary_recall
    )

async def coderetrx_filter(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    granularity: LLMMapFilterTargetType,
    coarse_recall_strategy: Literal["filename", "symbol", "line", "auto", "custom", "dependency"],
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        granularity,
        coarse_recall_strategy,
        codebase.llm_filter,
        custom_strategies,
        topic_extractor,
        settings,
        enable_secondary_recall
    )
