from typing import Dict, Any, List, Optional
import logging
import json
from pydantic import BaseModel

from codelib.impl.default.prompt import (
    KeywordExtractorResult,
    topic_extraction_prompt_template,
)
from codelib.utils.llm import call_llm_with_fallback
from codelib.retrieval import (
    TopicExtractor as TopicExtractorBase,
    SimilaritySearchTargetType,
)
from codelib.retrieval import SmartCodebase

logger = logging.getLogger(__name__)


class TopicExtractor(TopicExtractorBase):
    """
    A class to extract topics from input text using LLM before performing vector similarity searching.
    """

    async def extract_topic(self, input_text: str) -> Optional[str]:
        """
        Extract the core topic from the input text using LLM.

        Args:
            input_text: The input text to extract topic from

        Returns:
            The extracted topic as a string, or None if extraction fails
        """
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
            logger.info(f"Successfully extracted topic: '{extracted_topic}' from input")
            return extracted_topic

        except Exception as e:
            logger.error(f"Error extracting topic: {str(e)}", exc_info=True)
            return None

    async def extract_and_search(
        self,
        codebase: SmartCodebase,
        input_text: str,
        target_types: List[SimilaritySearchTargetType],
        threshold: float = 0.1,
        top_k: int = 100,
    ) -> List[Any]:
        """
        Extract topic from input text and use it for vector similarity search.

        Args:
            codebase: The codebase to search in
            input_text: The input text to extract topic from
            subdirs_or_files: List of subdirectories or files to process
            target_types: List of target types for similarity search
            threshold: Similarity threshold
            top_k: Number of top results to return

        Returns:
            List of search results
        """
        # Extract topic from input text
        topic = await self.extract_topic(input_text)

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
