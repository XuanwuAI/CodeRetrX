from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
from enum import Enum

from ..languages import IDXSupportedLanguage

if TYPE_CHECKING:
    from ..codebase import File, CodeChunk, Codebase


class ExtractionType(Enum):
    CHUNKS = "chunks"
    SYMBOLS = "symbols"
    KEYWORDS = "keywords"
    DEPENDENCIES = "dependencies"
    CALL_GRAPH = "call_graph"


class CodebaseParser(ABC):
    """
    Abstract base class for codebase parsers.

    Provides a unified interface for different parsing backends with direct
    population of codebase structures for optimal performance.
    """

    def __init__(self, **kwargs):
        """Initialize the parser with configuration options."""
        self.config = kwargs

    @abstractmethod
    def supports_language(self, language: IDXSupportedLanguage) -> bool:
        """
        Check if this parser supports the given language.

        Args:
            language: The language to check support for

        Returns:
            True if the language is supported, False otherwise
        """
        pass

    @abstractmethod
    def init_chunks(self, codebase: "Codebase") -> None:
        """
        Parse codebase and directly populate chunks into File objects.

        This method should populate:
        - codebase.all_chunks: List of all chunks across the codebase
        - file.chunks: List of chunks for each individual file

        Args:
            codebase: The codebase to parse and populate
        """
        pass

    @abstractmethod
    def init_symbols(self, codebase: "Codebase") -> None:
        """
        Parse codebase and directly populate symbols into Codebase object.

        This method should populate:
        - codebase.symbols: List of Symbol objects

        Args:
            codebase: The codebase to parse and populate
        """
        pass

    @abstractmethod
    def init_dependencies(self, codebase: "Codebase") -> None:
        """
        Parse codebase and directly populate dependencies into Codebase object.

        This method should populate:
        - codebase.dependencies: List of Dependency objects

        Args:
            codebase: The codebase to parse and populate
        """
        pass

    def init_keywords(self, codebase: "Codebase") -> None:
        """
        Parse codebase and directly populate keywords into Codebase object.

        This method provides a default implementation that extracts keywords
        from file content. It's parser-agnostic and can be used by all parsers.

        This method populates:
        - codebase.keywords: List of Keyword objects

        Args:
            codebase: The codebase to parse and populate
        """
        import re
        import logging
        from typing import Dict
        from tqdm import tqdm
        from ..codebase import Keyword

        logger = logging.getLogger(__name__)

        # Set to store unique keywords
        unique_keywords: Dict[str, Keyword] = {}

        # Check if we should use sentence-based extraction
        from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings

        settings = SmartCodebaseSettings()
        use_sentence_extraction = settings.keyword_sentence_extraction

        # Process each source file
        for file_path, file in tqdm(
            codebase.source_files.items(), desc="Extracting keywords"
        ):
            try:
                content = file.content

                if use_sentence_extraction:
                    # Sentence-based extraction
                    sentences = re.split(r"[.!?\n]+", content)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        # Remove extra whitespace and normalize
                        normalized_sentence = " ".join(sentence.split()).lower()

                        # Skip empty sentences
                        if not normalized_sentence:
                            continue

                        if normalized_sentence in unique_keywords:
                            unique_keywords[normalized_sentence].referenced_by.append(
                                file
                            )
                        else:
                            unique_keywords[normalized_sentence] = Keyword(
                                content=normalized_sentence, referenced_by=[file]
                            )
                else:
                    # Word-based extraction
                    words = content.split()
                    for word in words:
                        # Only reserve words, numbers and '.', '-', '_', '/'
                        normalized_word = re.sub(r"[^a-zA-Z0-9.-_/]", "", word.lower())

                        # Skip words that are too short (less than 3 characters)
                        if len(normalized_word) < 3:
                            continue

                        if normalized_word in unique_keywords:
                            unique_keywords[normalized_word].referenced_by.append(file)
                        else:
                            unique_keywords[normalized_word] = Keyword(
                                content=normalized_word, referenced_by=[file]
                            )

            except Exception as e:
                logger.warning(f"Error extracting keywords from file {file_path}: {e}")

        codebase.keywords = list(unique_keywords.values())

    def init_all(
        self, codebase: "Codebase", types: Optional[List[ExtractionType]] = None
    ) -> None:
        """
        Parse and populate all requested extraction types.

        Args:
            codebase: The codebase to parse and populate
            types: List of extraction types to process. If None, processes all types.
        """
        if types is None:
            types = [
                ExtractionType.CHUNKS,
                ExtractionType.SYMBOLS,
                ExtractionType.KEYWORDS,
                ExtractionType.DEPENDENCIES,
            ]

        if ExtractionType.CHUNKS in types:
            self.init_chunks(codebase)
        if ExtractionType.SYMBOLS in types:
            self.init_symbols(codebase)
        if ExtractionType.KEYWORDS in types:
            self.init_keywords(codebase)
        if ExtractionType.DEPENDENCIES in types:
            self.init_dependencies(codebase)

    def create_ast_codeblock(
        self,
        chunk: "CodeChunk",
        par_headlines: int = 3,
        show_line_numbers: bool = False,
        show_imports: bool = False,
        trunc_headlines: Optional[int] = None,
    ) -> str:
        """
        Create an AST-aware codeblock representation.

        Default implementation provides basic functionality.
        Parsers can override for enhanced AST-specific features.

        Args:
            chunk: The code chunk to represent
            par_headlines: Number of parent context lines to show
            show_line_numbers: Whether to show line numbers
            show_imports: Whether to show import statements
            trunc_headlines: Maximum number of lines to show

        Returns:
            Formatted codeblock string
        """
        parts = []
        prev_child = chunk
        parent = chunk.parent
        while parent:
            line_diff = prev_child.start_line - parent.start_line
            if line_diff > 0:
                par_lines = parent.lines()[: min(par_headlines, line_diff)]
                if show_line_numbers:
                    line_numbers = range(
                        parent.start_line, parent.start_line + len(par_lines)
                    )
                    par_lines = [
                        f"{num:4d} | {line}"
                        for num, line in zip(line_numbers, par_lines)
                    ]
                if line_diff > par_headlines:
                    par_lines.append("... Omitted Lines ...")
                parts.insert(0, "\n".join(par_lines))
            prev_child = parent
            parent = parent.parent
        if show_imports and chunk.type.value != "import":
            import_chunks = [
                c
                for c in chunk.src.chunks
                if c.type.value == "import" and c.parent is None
            ]
            parts.insert(
                0,
                "\n".join(
                    [c.code(show_line_numbers=show_line_numbers) for c in import_chunks]
                ),
            )
        parts.append("<CODE_CHUNK_IN_INTEREST>")
        code = chunk.code(
            do_dedent=False,
            show_line_numbers=show_line_numbers,
            trunc_headlines=trunc_headlines,
        )
        parts.append(code)
        parts.append("</CODE_CHUNK_IN_INTEREST>")
        parts.insert(0, f"```{chunk.src.path}:{chunk.src.lang()}")
        parts.append("```")
        return "\n".join(parts)

    # Legacy methods for backward compatibility
    def parse_file(self, file: "File") -> Any:
        """
        Legacy method for backward compatibility.

        Args:
            file: The file to parse

        Returns:
            Parser-specific state/data structure
        """
        # Default implementation - subclasses can override if needed
        return None

    def extract_chunks(self, file: "File", parse_state: Any) -> List["CodeChunk"]:
        """
        Legacy method for backward compatibility.

        Args:
            file: The file to extract chunks from
            parse_state: Parser-specific state returned by parse_file

        Returns:
            List of CodeChunk objects
        """
        # Default implementation - trigger codebase-level parsing if chunks not available
        if not file.chunks:
            self.init_chunks(file.codebase)
        return file.chunks

    def get_supported_languages(self) -> List[IDXSupportedLanguage]:
        """
        Get list of all supported languages.

        Returns:
            List of supported languages
        """
        from ..languages import IDXSupportedLanguage
        from typing import get_args

        return [
            lang
            for lang in get_args(IDXSupportedLanguage)
            if self.supports_language(lang)
        ]

    def cleanup(self):
        """
        Clean up any resources held by the parser.

        Default implementation does nothing, but subclasses can override
        to clean up temporary files, databases, etc.
        """
        pass

    def __enter__(self):
        """Support for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        self.cleanup()


class ParseError(Exception):
    """Exception raised when parsing fails."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        original_error: Optional[Exception] = None,
    ):
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(message)


class UnsupportedLanguageError(ParseError):
    """Exception raised when a language is not supported by the parser."""

    pass
