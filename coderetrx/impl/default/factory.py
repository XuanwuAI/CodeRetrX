# Compatibility import stub for backwards compatibility
# CodebaseFactory has been moved to coderetrx.retrieval.factory

# Will be removed in future versions
from coderetrx.retrieval.factory import CodebaseFactory
import logging
logger = logging.getLogger(__name__)
logger.warning("The 'coderetrx.impl.default.factory' module is deprecated and will be removed in future versions, use coderetrx.retrieval.factory instead.")

__all__ = ["CodebaseFactory"]