from typing import Dict, List, Optional
from pathlib import Path
import logging
from ...languages import IDXSupportedLanguage

logger = logging.getLogger(__name__)


class CodeQLQueryTemplates:
    """
    Provides CodeQL query templates for different languages and symbol types.

    Each template is loaded from .ql files in the queries/codeql directory,
    ensuring consistent results between parsers and maintainable query files.
    """

    # Supported query types for each language
    LANGUAGE_QUERY_TYPES = {
        "python": ["functions", "classes", "imports"],
        "javascript": ["functions", "classes", "imports"],
        "typescript": ["functions", "classes", "imports"],
        "java": ["functions", "classes", "imports"],
        "cpp": ["functions", "classes", "includes"],
        "c": ["functions", "classes", "includes"],
        "go": ["functions", "types", "imports"],
        "rust": ["functions", "structs"],
        "csharp": ["functions", "classes", "imports"],
    }

    @classmethod
    def get_supported_languages(cls) -> List[IDXSupportedLanguage]:
        """
        Get list of languages that have CodeQL query templates.

        Returns:
            List of supported languages
        """
        from typing import cast

        supported = []
        for language_str in cls.LANGUAGE_QUERY_TYPES.keys():
            language = cast(IDXSupportedLanguage, language_str)
            # Check if at least one query file exists for this language
            query_types = cls.LANGUAGE_QUERY_TYPES[language_str]
            for query_type in query_types:
                if cls._get_query_file_path(language, query_type).exists():
                    supported.append(language)
                    break

        return supported

    @classmethod
    def get_query(cls, language: IDXSupportedLanguage, query_type: str) -> Path:
        """
        Get a specific query for a language.

        Args:
            language: The language
            query_type: The type of query (e.g., 'functions', 'classes')

        Returns:
            Query text

        Raises:
            KeyError: If language or query type is not supported
            FileNotFoundError: If query file is not found
        """
        if language not in cls.LANGUAGE_QUERY_TYPES:
            raise KeyError(f"Language not supported: {language}")

        if query_type not in cls.LANGUAGE_QUERY_TYPES[language]:
            raise KeyError(
                f"Query type '{query_type}' not available for language: {language}"
            )

        query_file = cls._get_query_file_path(language, query_type)
        if not query_file.exists():
            raise FileNotFoundError(f"CodeQL query file not found: {query_file}")

        return query_file

    @classmethod
    def has_query(cls, language: IDXSupportedLanguage, query_type: str) -> bool:
        """
        Check if a specific query is available for a language.

        Args:
            language: The language
            query_type: The type of query

        Returns:
            True if query is available, False otherwise
        """
        if language not in cls.LANGUAGE_QUERY_TYPES:
            return False

        if query_type not in cls.LANGUAGE_QUERY_TYPES[language]:
            return False

        query_file = cls._get_query_file_path(language, query_type)
        return query_file.exists()

    @classmethod
    def get_available_queries(cls) -> Dict[str, List[str]]:
        """
        Get all available queries organized by language.

        Returns:
            Dictionary mapping language -> list of available query types
        """
        from typing import cast

        available = {}
        for language_str in cls.LANGUAGE_QUERY_TYPES:
            language = cast(IDXSupportedLanguage, language_str)
            query_types = []
            for query_type in cls.LANGUAGE_QUERY_TYPES[language_str]:
                if cls.has_query(language, query_type):
                    query_types.append(query_type)
            if query_types:
                available[language_str] = query_types

        return available

    @classmethod
    def _get_query_file_path(
        cls, language: IDXSupportedLanguage, query_type: str
    ) -> Path:
        """
        Get the file path for a specific query.

        Args:
            language: The language
            query_type: The type of query (e.g., 'functions', 'classes')

        Returns:
            Path to the query file
        """
        # Get the directory where this file is located
        current_file = Path(__file__)
        # Navigate to the queries directory: ../../../queries/codeql/
        queries_dir = current_file.parent.parent.parent / "queries" / "codeql"

        # Construct the path: queries/codeql/{language}/{query_type}.ql
        query_file_path = queries_dir / language / f"{query_type}.ql"

        return query_file_path
