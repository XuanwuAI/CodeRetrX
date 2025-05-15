from abc import ABC, abstractmethod
from typing import Optional, List, Any
from .smart_codebase import SmartCodebase, SimilaritySearchTargetType
from logging import warning, info


class TopicExtractor(ABC):
    @abstractmethod
    async def extract_topic(self, input_text: str) -> Optional[str]:
        pass

    async def extract_and_search(
        self,
        codebase: SmartCodebase,
        input_text: str,
        target_types: List[SimilaritySearchTargetType],
        threshold: float = 0.1,
        top_k: int = 100,
    ) -> List[Any]:
        topic = await self.extract_topic(input_text)
        if not topic:
            warning("Using original input text for search as topic extraction failed")
            topic = input_text

        info(f"Performing similarity search with topic: '{topic}'")

        # Perform similarity search using the extracted topic
        results = codebase.similarity_search(
            target_types=target_types, query=topic, threshold=threshold, top_k=top_k
        )

        info(f"Found {len(results)} results for topic '{topic}'")
        return results
