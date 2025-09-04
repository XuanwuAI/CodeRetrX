# Compatibility import stubs for backwards compatibility
# All functionality has been moved to coderetrx.retrieval
# Will be removed in future versions

from coderetrx.retrieval import (
    CodebaseFactory,
    SmartCodebase,
    TopicExtractor,
)
import logging
logger = logging.getLogger(__name__)
logger.warning("The 'coderetrx.impl.default' module is deprecated and will be removed in future versions. use coderetrx.retrieval instead.")

# For backwards compatibility, re-export the main classes
__all__ = [
    "CodebaseFactory",
    "SmartCodebase", 
    "TopicExtractor",
]
