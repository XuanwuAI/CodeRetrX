# Compatibility import stub for backwards compatibility  
# TopicExtractor has been moved to coderetrx.retrieval.topic_extractor
# Will be removed in future versions

from coderetrx.retrieval.topic_extractor import TopicExtractor
import logging
logger = logging.getLogger(__name__)
logger.warning("The 'coderetrx.impl.default.topic_extractor' module is deprecated and will be removed in future versions, use coderetrx.retrieval.topic_extractor instead.")


__all__ = ["TopicExtractor"]