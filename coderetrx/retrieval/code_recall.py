from typing import (
    Callable,
    List,
    Literal,
    Any,
    Tuple,
    Optional,
)
import logging

from pydantic import Field

from .smart_codebase import (
    LLMMapFilterTargetType,
    SmartCodebase as Codebase,
    CodeMapFilterResult, LLMCallMode,
)
from .topic_extractor import TopicExtractor
from coderetrx.static import Symbol, Keyword, File
from .strategy import (
    RecallStrategy,
    StrategyExecuteResult,
    StrategyFactory
)

from pydantic_settings import BaseSettings, SettingsConfigDict
logger = logging.getLogger(__name__)

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
    from coderetrx.utils.llm import call_llm_with_function_call
    
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
                        "enum": ["FILENAME", "SYMBOL", "LINE", "DEPENDENCY"]
                    },
                    "description": "The types of code elements to search for (can select one or multiple)",
                    "minItems": 1,
                    "maxItems": 4
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

DEPENDENCY: Use when the search criteria focus on external libraries, packages, or modules:
- Finding code that uses specific libraries or frameworks
- Package imports and usage patterns
- Third-party dependency analysis
- Module relationships and dependencies
- Library version compatibility
- Framework-specific code patterns
- API usage from external packages
- Import statement analysis
- Dependencies listed in package managers
- etc.

SELECTION GUIDELINES:
- Try to identify the SINGLE MOST APPROPRIATE type first
- Only select multiple types if the search criteria genuinely spans multiple categories and you cannot determine which single approach would be most effective
- When in doubt between two types, consider which one would capture the most relevant results for the specific query
"""

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
                strategies.append(RecallStrategy.FILTER_TOPK_LINE_BY_VECTOR_AND_LLM)
            elif element_type == "SYMBOL":
                strategies.append(RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM)
            elif element_type == "DEPENDENCY":
                strategies.append(RecallStrategy.FILTER_DEPENDENCY_BY_LLM)
            else:
                logger.warning(f"LLM returned invalid element type: {element_type}. Skipping.")
        
        if not strategies:
            logger.warning(f"No valid strategies determined. Defaulting to SYMBOL recall")
            return [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM]
        
        return strategies
            
    except Exception as e:
        logger.error(f"Error determining strategy by LLM: {e}. Defaulting to SYMBOL recall")
        return [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM]



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
    secondary_model_id = model_id or "anthropic/claude-4-sonnet"
    
    try:
        
        
        logger.info(f"Performing secondary recall on {len(elements)} code elements")
        
        secondary_elements, secondary_llm_results = await llm_method(
            prompt=prompt,
            target_type=granularity,
            additional_code_elements=elements,
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
    use_coarse_recall_returned_elements = True,
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
    
    if use_coarse_recall_returned_elements:
        subdirs_or_files = []
        additional_code_elements = strategy_result.elements 
    elements, llm_results = await llm_method(
        prompt=primary_prompt,
        target_type=granularity,
        subdirs_or_files=extended_subdirs_or_files,
        additional_code_elements = additional_code_elements,
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
    use_coarse_recall_returned_elements: bool = True,
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
        enable_secondary_recall,
        use_coarse_recall_returned_elements
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
    use_coarse_recall_returned_elements: bool = True,
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
        enable_secondary_recall,
        use_coarse_recall_returned_elements
    )
