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
CoarseRecallStrategyType = Literal[
    "file_name",
    "symbol_name",
    "symbol_content",
    "line_per_symbol",
    "dependency",
    "auto",
    "custom",
]


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
                        "enum": ["FILE_NAME", "SYMBOL_NAME", "LINE_PER_SYMBOL", "DEPENDENCY"]
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

FILE_NAME: Use when the search criteria can be identified primarily by file characteristics:
- Specific file extensions or patterns
- File naming conventions
- Configuration files
- Documentation files
- Files in specific directories
- Files with particular path patterns
- etc.

SYMBOL_NAME: Use when the search criteria involve specific named code constructs:
- Function names or method names
- Class names or interface names
- Module names or dependency names
- When you know the exact name of what you're looking for
- Shallow search based on identifiers
- etc.

LINE_PER_SYMBOL: Use when the search criteria require understanding file content and behavior:
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
            if element_type == "FILE_NAME":
                strategies.append(RecallStrategy.FILTER_FILENAME_BY_LLM)
            elif element_type == "LINE_PER_SYMBOL":
                strategies.append(RecallStrategy.FILTER_LINE_PER_SYMBOL_BY_VECTOR_AND_LLM)
            elif element_type == "SYMBOL_NAME":
                strategies.append(RecallStrategy.FILTER_SYMBOL_NAME_BY_LLM)
            elif element_type == "DEPENDENCY":
                strategies.append(RecallStrategy.FILTER_DEPENDENCY_BY_LLM)
            else:
                logger.warning(f"LLM returned invalid element type: {element_type}. Skipping.")
        
        if not strategies:
            logger.warning(f"No valid strategies determined. Defaulting to SYMBOL recall")
            return [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM]
        
        return strategies
            
    except Exception as e:
        logger.error(f"Error determining strategy by LLM: {e}. Defaulting to SYMBOL recall")
        return [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM]



async def _perform_secondary_recall(
    codebase: Codebase,
    prompt: str,
    elements: List[Any],
    llm_results: List[Any],
    target_type: LLMMapFilterTargetType,
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
        target_type: Target type for filtering
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
            target_type=target_type,
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
    target_type: LLMMapFilterTargetType,
    coarse_recall_strategy: str,
    llm_method: Callable,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
    extend_coarse_recall_element_to_file = True,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    """
    Process code elements based on the specified prompt and mode.

    Args:
        prompt: The prompt for filtering or mapping
        subdirs_or_files: List of subdirectories or files to process
        target_type: The target_type level for code analysis
        mode: The search mode to use:
            - "file_name": Uses FILTER_FILENAME_BY_LLM only
            - "symbol_name": Uses ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM strategy
            - "line_per_symbol": Uses INTELLIGENT_FILTER strategy with line-level vector recall
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
    if coarse_recall_strategy == "file_name":
        strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]
    elif coarse_recall_strategy == "symbol_name":
        strategies_to_run = [RecallStrategy.FILTER_SYMBOL_NAME_BY_LLM]
    elif coarse_recall_strategy == "line_per_symbol":
        strategies_to_run = [RecallStrategy.FILTER_LINE_PER_SYMBOL_BY_VECTOR_AND_LLM]
    elif coarse_recall_strategy == "symbol_content":
        strategies_to_run = [RecallStrategy.ADAPTIVE_FILTER_SYMBOL_CONTENT_BY_VECTOR_AND_LLM]
    elif coarse_recall_strategy == "dependency":
        strategies_to_run = [RecallStrategy.FILTER_DEPENDENCY_BY_LLM]
    elif coarse_recall_strategy == "auto":
        strategies = await _determine_strategy_by_llm(
            prompt=prompt,
            model_id=settings.llm_selector_strategy_model_id,
        )
        strategy_names = [strategy.value for strategy in strategies]
        print(f"LLM smart strategies selected: {strategy_names}")
        strategies_to_run = strategies
    elif coarse_recall_strategy == "keyword":
        strategies_to_run = [RecallStrategy.ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM]
    elif coarse_recall_strategy == "precise":
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
    elif coarse_recall_strategy == "custom" and custom_strategies:
        strategies_to_run = custom_strategies
    else:
        if coarse_recall_strategy == "custom" and not custom_strategies:
            logger.warning(
                "Custom mode specified but no custom strategies provided. Defaulting to fast mode."
            )
            strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]
        else:
            logger.warning(f"Unknown mode: {coarse_recall_strategy}. Defaulting to fast mode.")
            strategies_to_run = [RecallStrategy.FILTER_FILENAME_BY_LLM]

    # Execute each strategy in sequence
    if coarse_recall_strategy != "precise":
        strategy_factory = StrategyFactory(topic_extractor=topic_extractor, llm_call_mode=settings.llm_call_mode)
        for strategy in strategies_to_run:
            try:
                strategy_executor = strategy_factory.create_strategy(strategy)
                strategy_result = await strategy_executor.execute(
                    codebase, prompt, subdirs_or_files, target_type
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
        f"Processing {len(extended_subdirs_or_files)} files with {target_type} target_type"
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
    
    additional_code_elements = []
    if not extend_coarse_recall_element_to_file and coarse_recall_strategy in ["line_per_symbol", "symbol_name"]:
        extended_subdirs_or_files = []
        additional_code_elements = strategy_result.elements 
    elements, llm_results = await llm_method(
        prompt=primary_prompt,
        target_type=target_type,
        subdirs_or_files=extended_subdirs_or_files,
        additional_code_elements=additional_code_elements,
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
            target_type=target_type,
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
    target_type: LLMMapFilterTargetType,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        target_type,
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
    target_type: Literal["symbol_content", "file_content", "function_content", "class_content"],
    coarse_recall_strategy: CoarseRecallStrategyType, 
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
    extend_coarse_recall_element_to_file: bool = True,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    if not target_type.endswith("_content"):
        raise ValueError(
            f"Target type '{target_type}' must end with '_content' for coderetrx_mapping. Use 'symbol_content', 'file_content', etc."
        )

    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        target_type,
        coarse_recall_strategy,
        codebase.llm_map,
        custom_strategies,
        topic_extractor,
        settings,
        enable_secondary_recall,
        extend_coarse_recall_element_to_file
    )

async def llm_traversal_filter(
    codebase: Codebase,
    prompt: str,
    subdirs_or_files: List[str],
    target_type: LLMMapFilterTargetType,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        target_type,
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
    target_type: Literal["symbol_content", "file_content", "function_content", "class_content"],
    coarse_recall_strategy: CoarseRecallStrategyType,
    custom_strategies: List[RecallStrategy] = [],
    topic_extractor: Optional[TopicExtractor] = None,
    settings: Optional[CodeRecallSettings] = None,
    enable_secondary_recall: bool = False,
    extend_coarse_recall_element_to_file: bool = True,
) -> Tuple[List[Symbol | File | Keyword], List[CodeMapFilterResult]]:
    if not target_type.endswith("_content"):
        raise ValueError(
            f"Target type '{target_type}' must end with '_content' for coderetrx_filter. Use 'symbol_content', 'file_content', etc."
        )
    return await _multi_strategy_code_recall(
        codebase,
        prompt,
        subdirs_or_files,
        target_type,
        coarse_recall_strategy,
        codebase.llm_filter,
        custom_strategies,
        topic_extractor,
        settings,
        enable_secondary_recall,
        extend_coarse_recall_element_to_file
    )
