
# Compatibility import stub for backwards compatibility
# SmartCodebase has been moved to coderetrx.retrieval.smart_codebase
# Will be removed in future versions

from coderetrx.retrieval.smart_codebase import SmartCodebase, SmartCodebaseSettings
import logging
logger = logging.getLogger(__name__)
logger.warning("The 'coderetrx.impl.default.smart_codebase' module is deprecated and will be removed in future versions, use coderetrx.retrieval.smart_codebase instead.")

__all__ = ["SmartCodebase", "SmartCodebaseSettings"]