from typing import Optional, Union, Dict, Any, List
import logging
from pathlib import Path

from .base import CodebaseParser, UnsupportedLanguageError
from .treesitter import TreeSitterParser
from .codeql import CodeQLParser
from ..languages import IDXSupportedLanguage

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory for creating and managing codebase parsers.

    Supports auto-selection of parsers based on availability and preferences,
    with fallback mechanisms for robustness.
    """

    # Parser priority order for auto-selection
    PARSER_PRIORITY = ["treesitter", "codeql"]

    @classmethod
    def get_parser(cls, parser_type: str = "auto", **kwargs) -> CodebaseParser:
        """
        Get a parser instance.

        Args:
            parser_type: Type of parser to create:
                - "auto": Auto-select best available parser
                - "codeql": CodeQL parser
                - "treesitter": Tree-sitter parser
                - "hybrid": Use CodeQL where available, fallback to tree-sitter
            **kwargs: Parser-specific configuration options

        Returns:
            CodebaseParser instance

        Raises:
            ValueError: If parser_type is invalid
            RuntimeError: If no suitable parser is available
        """
        if parser_type == "auto":
            return cls._auto_select_parser(**kwargs)
        elif parser_type == "codeql":
            return cls._create_codeql_parser(**kwargs)
        elif parser_type == "treesitter":
            return cls._create_treesitter_parser(**kwargs)
        else:
            raise ValueError(f"Unknown parser type: {parser_type}")

    @classmethod
    def _auto_select_parser(cls, **kwargs) -> CodebaseParser:
        """
        Auto-select the best available parser.

        Tries parsers in priority order and returns the first working one.
        """
        for parser_name in cls.PARSER_PRIORITY:
            try:
                if parser_name == "codeql":
                    parser = cls._create_codeql_parser(**kwargs)
                    # Test if CodeQL is actually available
                    if cls._test_codeql_availability(parser):
                        logger.info("Auto-selected CodeQL parser")
                        return parser
                    else:
                        logger.info("CodeQL CLI not available, trying next parser")
                        continue
                elif parser_name == "treesitter":
                    parser = cls._create_treesitter_parser(**kwargs)
                    logger.info("Auto-selected Tree-sitter parser")
                    return parser
            except Exception as e:
                logger.debug(f"Failed to create {parser_name} parser: {e}")
                continue

        raise RuntimeError("No suitable parser available")

    @classmethod
    def _create_codeql_parser(cls, **kwargs) -> CodeQLParser:
        """Create a CodeQL parser instance."""
        return CodeQLParser(**kwargs)

    @classmethod
    def _create_treesitter_parser(cls, **kwargs) -> TreeSitterParser:
        """Create a Tree-sitter parser instance."""
        return TreeSitterParser(**kwargs)

    @classmethod
    def _test_codeql_availability(cls, parser: CodeQLParser) -> bool:
        """
        Test if CodeQL is actually available and working.

        Args:
            parser: CodeQL parser instance to test

        Returns:
            True if CodeQL is available, False otherwise
        """
        try:
            # Test basic CodeQL functionality
            supported_langs = parser.get_supported_languages()
            return len(supported_langs) > 0
        except Exception as e:
            logger.debug(f"CodeQL availability test failed: {e}")
            return False

    @classmethod
    def get_available_parsers(cls) -> Dict[str, bool]:
        """
        Get status of all available parsers.

        Returns:
            Dictionary mapping parser names to availability status
        """
        status = {}

        # Test Tree-sitter
        try:
            parser = cls._create_treesitter_parser()
            status["treesitter"] = len(parser.get_supported_languages()) > 0
        except Exception:
            status["treesitter"] = False

        # Test CodeQL
        try:
            parser = cls._create_codeql_parser()
            status["codeql"] = cls._test_codeql_availability(parser)
        except Exception:
            status["codeql"] = False

        return status

    @classmethod
    def recommend_parser(
        cls, languages: Optional[List[IDXSupportedLanguage]] = None
    ) -> str:
        """
        Recommend the best parser for given languages.

        Args:
            languages: List of languages to support (None for all)

        Returns:
            Recommended parser name
        """
        available = cls.get_available_parsers()

        if not languages:
            # If no specific languages, prefer CodeQL if available
            if available.get("codeql", False):
                return "codeql"
            elif available.get("treesitter", False):
                return "treesitter"
            else:
                return "auto"  # Let auto-selection handle the error

        # Check language support for each parser
        codeql_coverage = 0
        treesitter_coverage = 0

        if available.get("codeql", False):
            try:
                codeql_parser = cls._create_codeql_parser()
                codeql_supported = set(codeql_parser.get_supported_languages())
                codeql_coverage = len(set(languages).intersection(codeql_supported))
            except Exception:
                pass

        if available.get("treesitter", False):
            try:
                treesitter_parser = cls._create_treesitter_parser()
                treesitter_supported = set(treesitter_parser.get_supported_languages())
                treesitter_coverage = len(
                    set(languages).intersection(treesitter_supported)
                )
            except Exception:
                pass

        # Prefer CodeQL if it covers more languages, otherwise tree-sitter
        if codeql_coverage > treesitter_coverage:
            return "codeql"
        elif treesitter_coverage > 0:
            return "treesitter"
        elif codeql_coverage > 0:
            return "codeql"
        else:
            return "hybrid"  # Use hybrid for maximum coverage
