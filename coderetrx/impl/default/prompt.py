# Compatibility import stub for backwards compatibility
# Prompt utilities have been moved to coderetrx.retrieval.prompt
# Will be removed in future versions

from coderetrx.retrieval.prompt import *
import logging
logger = logging.getLogger(__name__)
logger.warning("The 'coderetrx.impl.default.prompt' module is deprecated and will be removed in future versions, use coderetrx.retrieval.prompt instead.")

__all__ = [
    "llm_filter_prompt_template",
    "llm_mapping_prompt_template", 
    "topic_extraction_prompt_template",
    "KeywordExtractorResult",
    "llm_filter_function_call_system_prompt",
    "llm_mapping_function_call_system_prompt",
    "topic_extraction_function_call_system_prompt",
    "filter_and_mapping_function_call_user_prompt_template",
    "topic_extraction_function_call_user_prompt_template",
    "get_filter_function_definition",
    "get_mapping_function_definition", 
    "get_topic_extraction_function_definition",
]