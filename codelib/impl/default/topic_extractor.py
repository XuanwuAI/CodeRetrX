from typing import Dict, Any, List, Optional, Literal
import logging
import json
from pydantic import BaseModel

from codelib.impl.default.prompt import (
    KeywordExtractorResult,
    topic_extraction_prompt_template,
    topic_extraction_function_call_system_prompt,
    get_topic_extraction_function_definition, topic_extraction_function_call_user_prompt_template,
)
from codelib.utils.llm import call_llm_with_fallback, call_llm_with_function_call
from codelib.retrieval import (
    TopicExtractor as TopicExtractorBase,
    SimilaritySearchTargetType,
)
from codelib.retrieval import SmartCodebase
import os

logger = logging.getLogger(__name__)


class TopicExtractor(TopicExtractorBase):
    """
    A class to extract topics from input text using LLM before performing vector similarity searching.
    """
    async def extract_topic(self, input_text: str, llm_call_mode: Literal["traditional", "function_call"] = "traditional") -> Optional[str]:
        """
        Extract the core topic from the input text using LLM.

        Args:
            input_text: The input text to extract topic from
            llm_call_mode: Whether to use traditional prompt-based extraction or function call mode.

        Returns:
            The extracted topic as a string, or None if extraction fails
        """
        try:
            if llm_call_mode == "function_call":
                return await self._extract_topic_with_function_call(input_text)
            else:
                return await self._extract_topic_traditional(input_text)
        except Exception as e:
            logger.error(f"Error extracting topic: {str(e)}", exc_info=True)
            return None

    async def _extract_topic_traditional(self, input_text: str) -> Optional[str]:
        """Extract topic using traditional prompt-based approach."""
        try:
            # Prepare input data for the prompt template
            input_data = {"input": input_text}

            # Call LLM with the topic extraction prompt template
            result = await call_llm_with_fallback(
                response_model=KeywordExtractorResult,
                input_data=input_data,
                prompt_template=topic_extraction_prompt_template,
            )

            assert isinstance(result, KeywordExtractorResult)
            # Extract the topic from the result
            extracted_topic = result.result
            logger.info(f"Successfully extracted topic: '{extracted_topic}' from input using traditional mode")
            return extracted_topic

        except Exception as e:
            logger.error(f"Error in traditional topic extraction: {str(e)}", exc_info=True)
            return None

    async def _extract_topic_with_function_call(self, input_text: str) -> Optional[str]:
        """Extract topic using function call approach."""
        try:
            # Prepare prompts for function call
            system_prompt = topic_extraction_function_call_system_prompt
            user_prompt = topic_extraction_function_call_user_prompt_template.format(
                input=input_text,
            )
            function_definition = get_topic_extraction_function_definition()

            # Call LLM with function call
            model_ids = [os.environ.get("LLM_MAPFILTER_MODEL_ID", "openai/gpt-4.1-mini"), "anthropic/claude-3.7-sonnet"]
            function_args = await call_llm_with_function_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_definition=function_definition,
                model_ids=model_ids,
            )

            # Extract topic from function call result
            extracted_topic = function_args.get("result")
            reason = function_args.get("reason", "")

            if extracted_topic:
                logger.info(f"Successfully extracted topic: '{extracted_topic}' from input using function call mode. Reason: {reason}")
                return extracted_topic
            else:
                logger.warning("Function call returned empty result for topic extraction")
                return None

        except Exception as e:
            logger.error(f"Error in function call topic extraction: {str(e)}", exc_info=True)
            return None

    async def extract_and_search(
        self,
        codebase: SmartCodebase,
        input_text: str,
        target_types: List[SimilaritySearchTargetType],
        threshold: float = 0.1,
        top_k: int = 100,
        llm_call_mode: Literal["traditional", "function_call"] = "traditional",
    ) -> List[Any]:
        """
        Extract topic from input text and use it for vector similarity search.

        Args:
            codebase: The codebase to search in
            input_text: The input text to extract topic from
            target_types: List of target types for similarity search
            threshold: Similarity threshold
            top_k: Number of top results to return
            llm_call_mode: Whether to use traditional prompt-based extraction or function call mode.

        Returns:
            List of search results
        """
        # Extract topic from input text
        topic = await self.extract_topic(input_text, llm_call_mode)

        if not topic:
            logger.warning(
                "Using original input text for search as topic extraction failed"
            )
            topic = input_text

        logger.info(f"Performing similarity search with topic: '{topic}'")

        # Perform similarity search using the extracted topic
        results = await codebase.similarity_search(
            target_types=target_types, query=topic, threshold=threshold, top_k=top_k
        )

        logger.info(f"Found {len(results)} results for topic '{topic}'")
        return results
