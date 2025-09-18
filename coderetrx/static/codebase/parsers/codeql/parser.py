from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from ..base import CodebaseParser, ParseError, UnsupportedLanguageError
from ...languages import IDXSupportedLanguage, IDXSupportedTag, get_language
from ....codeql.codeql import (
    CodeQLWrapper,
    CodeQLDatabase,
    CodeQLRawResult,
    CodeQLResultParser,
)
from .queries import CodeQLQueryTemplates

if TYPE_CHECKING:
    from ...codebase import File, CodeChunk, ChunkType, hunk_uuid, Codebase

logger = logging.getLogger(__name__)


@dataclass
class CodeQLParserResult:
    """Represents a single result from a CodeQL query."""

    file_path: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    query_name: Optional[str] = None


class CodeQLParserResultParser(CodeQLResultParser):
    """Default parser that produces CodeQLResult objects."""

    def parse(self, raw_result: CodeQLRawResult) -> List[CodeQLParserResult]:
        """Parse raw result into CodeQLResult objects using the default format."""
        results = []

        # Parse each tuple based on the expected format from our queries
        # Our queries typically return: [file_path, symbol_name, start_line, end_line, start_column, end_column]
        for tuple_data in raw_result.tuples:
            try:
                if len(tuple_data) >= 6:
                    file_path = str(tuple_data[0])
                    symbol_name = str(tuple_data[1]) if tuple_data[1] else None
                    start_line = (
                        int(tuple_data[2]) - 1 if tuple_data[2] is not None else 0
                    )  # Convert to 0-based
                    end_line = (
                        int(tuple_data[3]) - 1
                        if tuple_data[3] is not None
                        else start_line
                    )
                    start_column = (
                        int(tuple_data[4]) - 1 if tuple_data[4] is not None else 0
                    )  # Convert to 0-based
                    end_column = (
                        int(tuple_data[5]) - 1
                        if tuple_data[5] is not None
                        else start_column
                    )

                    # Clean up file path - remove leading slash to make it relative
                    if file_path.startswith("/"):
                        file_path = file_path[1:]

                    result = CodeQLParserResult(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_column,
                        end_column=end_column,
                        symbol_name=symbol_name,
                        query_name=raw_result.query_name,
                    )
                    results.append(result)

                else:
                    logger.warning(
                        f"Unexpected tuple format in query '{raw_result.query_name}': {tuple_data}"
                    )

            except (ValueError, IndexError, TypeError) as e:
                logger.warning(
                    f"Failed to parse tuple in query '{raw_result.query_name}': {tuple_data}, error: {e}"
                )
                continue

        logger.debug(
            f"Parsed {len(results)} results from query '{raw_result.query_name}'"
        )
        return results


class CodeQLParseState:
    """
    Represents the parse state for CodeQL parsing.

    Since CodeQL works at the codebase level rather than individual files,
    this state contains the database and query results for the entire codebase.
    """

    def __init__(
        self,
        database: CodeQLDatabase,
        query_results: Dict[str, List[CodeQLParserResult]],
    ):
        self.database = database
        self.query_results = query_results
        self._results_by_file = None

    def get_results_for_file(self, file_path: str) -> List[CodeQLParserResult]:
        """Get all query results for a specific file."""
        if self._results_by_file is None:
            self._build_file_index()
        return self._results_by_file.get(file_path, []) if self._results_by_file else []

    def _build_file_index(self):
        """Build an index of results by file path for efficient lookup."""
        self._results_by_file = defaultdict(list)
        for query_name, results in self.query_results.items():
            for result in results:
                self._results_by_file[result.file_path].append(result)


class CodeQLParser(CodebaseParser):
    """
    CodeQL based parser implementation.

    This parser uses CodeQL to analyze entire codebases and extract
    code chunks with consistent symbol identification.
    """

    def __init__(self, codeql_cli_path: Optional[str] = None, **kwargs):
        """
        Initialize CodeQL parser.

        Args:
            codeql_cli_path: Path to CodeQL CLI executable
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)

        # Force single-threaded execution to avoid cache conflicts
        wrapper_kwargs = kwargs.copy()
        wrapper_kwargs["max_workers"] = 1
        wrapper_kwargs["codeql_cli_path"] = codeql_cli_path

        self.wrapper = CodeQLWrapper(**wrapper_kwargs)
        self._databases: Dict[str, CodeQLDatabase] = {}
        self._parse_states: Dict[str, CodeQLParseState] = {}

    def supports_language(self, language: IDXSupportedLanguage) -> bool:
        """Check if CodeQL supports the given language."""
        return (
            self.wrapper.supports_language(language)
            and language in CodeQLQueryTemplates.get_supported_languages()
        )

    def parse_file(self, file: "File") -> CodeQLParseState:
        """
        Parse a file using CodeQL.

        Since CodeQL works at the codebase level, this method will create
        or reuse a database for the entire codebase and run all queries.

        Args:
            file: The file to parse

        Returns:
            CodeQLParseState containing database and query results

        Raises:
            UnsupportedLanguageError: If the file language is not supported
            ParseError: If parsing fails
        """
        if file.file_type != "source":
            raise ParseError(f"Cannot parse non-source file: {file.path}")

        lang = get_language(file.path)
        if lang is None:
            raise UnsupportedLanguageError(f"Unknown language for file: {file.path}")

        if not self.supports_language(lang):
            raise UnsupportedLanguageError(f"CodeQL does not support language: {lang}")

        # Use codebase-level caching to avoid recreating databases
        codebase_key = f"{file.codebase.id}_{lang}"

        if codebase_key not in self._parse_states:
            self.init_chunks(file.codebase)  # Ensure chunks are populated
        return self._parse_states[codebase_key]

    def extract_chunks(
        self, file: "File", parse_state: CodeQLParseState
    ) -> List["CodeChunk"]:
        """
        Extract code chunks from CodeQL query results.

        Args:
            file: The file to extract chunks from
            parse_state: CodeQLParseState from parse_file

        Returns:
            List of CodeChunk objects with consistent UUIDs
        """
        chunks = []

        # Get results for this specific file
        # Convert file path to relative path for matching
        relative_path = str(file.path)
        file_results = parse_state.get_results_for_file(relative_path)

        # Also try absolute path matching
        if not file_results:
            absolute_path = str(Path(file.codebase.dir) / file.path)
            file_results = parse_state.get_results_for_file(absolute_path)

        logger.debug(f"Found {len(file_results)} CodeQL results for {file.path}")

        # Convert CodeQL results to CodeChunk objects
        for result in file_results:
            try:
                chunk = self.create_chunk_from_codeql_result(result, file)
                if chunk:
                    chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to convert CodeQL result to chunk: {e}")
                continue

        # Populate parent-child relationships
        self._populate_nesting(chunks)

        return chunks

    def create_chunk_from_codeql_result(
        self, result: CodeQLParserResult, file: "File"
    ) -> Optional["CodeChunk"]:
        """
        Convert a CodeQL result to a CodeChunk.

        This method ensures consistent UUID generation by using the same
        hunk_uuid function as tree-sitter.

        Args:
            result: CodeQL query result
            file: Source file

        Returns:
            CodeChunk object or None if conversion fails
        """
        # Determine chunk type and tag based on query name
        chunk_type, tag = self._get_chunk_type_and_tag(result.query_name or "unknown")

        if chunk_type is None:
            return None

        from ...codebase import CodeChunk

        chunk = CodeChunk.new(
            src=file,
            start_line=result.start_line,
            end_line=result.end_line,
            start_column=result.start_column,
            end_column=result.end_column,
            chunk_type=chunk_type,
            tag=tag,  # type: ignore
            name=result.symbol_name,
        )
        return chunk

    def _get_chunk_type_and_tag(
        self, query_name: str
    ) -> tuple[Optional["ChunkType"], Optional[str]]:
        """
        Map CodeQL query names to chunk types and tags.

        Args:
            query_name: Name of the CodeQL query

        Returns:
            Tuple of (ChunkType, tag) or (None, None) if not mappable
        """
        from ...codebase import ChunkType

        # Map query names to chunk types and tags
        # Handle both custom query names and CodeQL rule IDs from analyze
        mapping = {
            # PRIMARY tags
            "functions": (ChunkType.PRIMARY, "definition.function"),
            "classes": (ChunkType.PRIMARY, "definition.class"),
            "methods": (ChunkType.PRIMARY, "definition.method"),
            "types": (ChunkType.PRIMARY, "definition.class"),  # Go types
            "structs": (ChunkType.PRIMARY, "definition.class"),  # Rust structs
            # REFERENCE tags
            "function_calls": (ChunkType.REFERENCE, "reference.call"),
            "method_calls": (ChunkType.REFERENCE, "reference.call"),
            # IMPORT tags
            "imports": (ChunkType.IMPORT, "import"),
            "includes": (ChunkType.IMPORT, "import"),  # C/C++ includes
            # CodeQL rule IDs from analyze command
            "py/unused-import": (ChunkType.IMPORT, "import.unused"),
            "js/unused-local-variable": (ChunkType.PRIMARY, "definition.variable"),
            "py/unused-local-variable": (ChunkType.PRIMARY, "definition.variable"),
            # Analyze results - map to PRIMARY by default for unknown rule IDs
            "analyze": (ChunkType.PRIMARY, "definition.function"),
            "analyze_result": (ChunkType.PRIMARY, "definition.function"),
        }

        return mapping.get(query_name, (None, None))

    def _populate_nesting(self, chunks: List["CodeChunk"]):
        """
        Populate parent-child relationships between chunks.

        Uses the same algorithm as tree-sitter parser for consistency.
        """
        # Sort chunks by start line and then by end line in reverse order
        chunks.sort(key=lambda x: (x.start_line, -x.end_line))

        # Stack to keep track of potential parent chunks
        parent_stack = []

        for chunk in chunks:
            # Pop chunks from the stack that end before the current chunk starts
            while parent_stack and parent_stack[-1][0] < chunk.start_line:
                parent_stack.pop()

            # If there's a chunk on the stack, it contains the current chunk
            if parent_stack:
                chunk.parent = parent_stack[-1][1]

            # Push the current chunk onto the stack as a potential parent
            parent_stack.append((chunk.end_line, chunk))

    def _get_query_files_for_language(
        self, language: IDXSupportedLanguage
    ) -> Dict[str, Path]:
        """
        Get available query files for a language.

        Args:
            language: The language to get queries for

        Returns:
            Dictionary mapping query_name -> query_file_path
        """
        query_files = {}

        # Get available query types for this language
        if language in CodeQLQueryTemplates.LANGUAGE_QUERY_TYPES:
            query_types = CodeQLQueryTemplates.LANGUAGE_QUERY_TYPES[language]

            for query_type in query_types:
                try:
                    query_file = CodeQLQueryTemplates.get_query(language, query_type)
                    query_files[query_type] = query_file
                except (KeyError, FileNotFoundError) as e:
                    logger.debug(
                        f"Query {query_type} not available for {language}: {e}"
                    )
                    continue

        return query_files

    def cleanup(self):
        """Clean up all CodeQL databases and resources."""
        for database in self._databases.values():
            database.cleanup()
        self._databases.clear()
        self._parse_states.clear()

        # Clean up the wrapper's cache directory
        if self.wrapper:
            self.wrapper.cleanup()

    def get_supported_languages(self) -> List[IDXSupportedLanguage]:
        """Get languages supported by both CodeQL wrapper and query templates."""
        from typing import cast

        # Get languages that the wrapper supports (based on LANGUAGE_MAP)
        wrapper_languages = set(self.wrapper.LANGUAGE_MAP.keys())
        template_languages = set(CodeQLQueryTemplates.get_supported_languages())
        intersection = wrapper_languages.intersection(template_languages)
        return [cast(IDXSupportedLanguage, lang) for lang in intersection]

    # New interface methods for direct codebase population
    def init_chunks(self, codebase: "Codebase") -> None:
        """
        Parse codebase and directly populate chunks into File objects.

        CodeQL works at the repo level, so we create one database per language
        and then filter results by file.
        """
        from tqdm import tqdm

        # Group files by language
        files_by_lang = defaultdict(list)
        for file in codebase.source_files.values():
            lang = get_language(file.path)
            if lang and self.supports_language(lang):
                files_by_lang[lang].append(file)

        # Process each language
        for lang, files in files_by_lang.items():
            logger.info(f"Processing {len(files)} {lang} files with CodeQL")

            try:
                # Create database for this language
                database = self.wrapper.create_database(
                    source_dir=Path(codebase.dir),
                    language=lang,
                    project_name=codebase.id,
                    database_name=lang,
                )

                # Use custom queries from files
                logger.info(f"Running custom CodeQL queries for {lang}")
                custom_query_files = {
                    "functions": CodeQLQueryTemplates.get_query(lang, "functions"),
                    "classes": CodeQLQueryTemplates.get_query(lang, "classes"),
                }

                if not custom_query_files:
                    raise UnsupportedLanguageError(
                        f"No custom query files found for language: {lang}"
                    )

                # Run custom queries from files
                query_results = self.wrapper.run_queries(
                    database, custom_query_files, parser=CodeQLParserResultParser()
                )

                # Create parse state
                parse_state = CodeQLParseState(database, query_results)

                # Extract chunks for each file
                for file in tqdm(files, desc=f"Extracting {lang} chunks"):
                    try:
                        chunks = self.extract_chunks(file, parse_state)
                        file.chunks.extend(chunks)
                        codebase.all_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract chunks from {file.path}: {e}"
                        )
                        continue

                # Store parse state for potential reuse
                codebase_key = f"{codebase.id}_{lang}"
                self._parse_states[codebase_key] = parse_state
                self._databases[codebase_key] = database

            except Exception as e:
                logger.warning(f"Failed to process {lang} files: {e}")
                continue

    def init_symbols(self, codebase: "Codebase") -> None:
        """
        Create symbols from existing chunks and populate into codebase.
        """
        # Ensure chunks are populated
        if not codebase.all_chunks:
            self.init_chunks(codebase)

        from ...codebase import Symbol, ChunkType
        from ...languages import OBJLIKE_TAGS, FUNCLIKE_TAGS

        # Create Symbol objects from chunks
        for chunk in codebase.all_chunks:
            symbol_name = chunk.symbol_name()
            if chunk.type == ChunkType.PRIMARY and symbol_name:
                symbol_type = "class" if chunk.tag in OBJLIKE_TAGS else "function"
                symbol = Symbol(
                    name=symbol_name,
                    type=symbol_type,
                    file=chunk.src,
                    chunk=chunk,
                    id=chunk.id,
                )
                codebase.symbols.append(symbol)

    def init_dependencies(self, codebase: "Codebase") -> None:
        """
        Extract dependencies using CodeQL and populate into codebase.
        """
        from ...codebase import Symbol, Dependency

        dependency_symbols: List[Symbol] = []

        # Group files by language for efficient processing
        files_by_lang = defaultdict(list)
        for file in codebase.source_files.values():
            lang = get_language(file.path)
            if lang and self.supports_language(lang):
                files_by_lang[lang].append(file)

        # Process each language
        for lang, files in files_by_lang.items():
            try:
                # Use existing database if available
                codebase_key = f"{codebase.id}_{lang}"
                query_type = "includes" if lang in ("c", "cpp") else "imports"
                if codebase_key in self._parse_states:
                    # Reuse existing database, run import/includes query and merge
                    parse_state = self._parse_states[codebase_key]
                    database = parse_state.database
                    try:
                        query_path = CodeQLQueryTemplates.get_query(lang, query_type)
                    except (KeyError, FileNotFoundError):
                        # No import/includes query for this language
                        continue
                    import_results = self.wrapper.run_query(
                        database,
                        query_path,
                        query_name=query_type,
                        parser=CodeQLParserResultParser(),
                    )
                    # Merge results and reset index
                    parse_state.query_results[query_type] = import_results
                    parse_state._results_by_file = None
                else:
                    # Create new database for dependency analysis
                    database = self.wrapper.create_database(
                        source_dir=Path(codebase.dir),
                        language=lang,
                        project_name=codebase.id,
                        database_name=f"{lang}_deps",
                    )
                    try:
                        query_path = CodeQLQueryTemplates.get_query(lang, query_type)
                    except (KeyError, FileNotFoundError):
                        # No import/includes query for this language
                        continue
                    import_results = self.wrapper.run_query(
                        database,
                        query_path,
                        query_name=query_type,
                        parser=CodeQLParserResultParser(),
                    )
                    query_results = {query_type: import_results}
                    parse_state = CodeQLParseState(database, query_results)
                    # Store for potential reuse
                    self._parse_states[codebase_key] = parse_state
                    self._databases[codebase_key] = database

                # Extract dependency symbols from import/includes results
                for file in files:
                    file_results = parse_state.get_results_for_file(str(file.path))
                    for result in file_results:
                        if result.query_name != query_type or not result.symbol_name:
                            continue
                        module_name = result.symbol_name
                        # Filter out local/relative imports to align with Tree-sitter
                        if module_name.startswith(".") or module_name.startswith("/"):
                            continue
                        # Create dependency chunk
                        chunk = self.create_chunk_from_codeql_result(result, file)
                        if not chunk:
                            continue
                        # Create dependency symbol
                        sym = Symbol(
                            name=module_name,
                            type="dependency",
                            file=file,
                            chunk=chunk,
                        )
                        dependency_symbols.append(sym)
                        file.chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Failed to extract dependencies for {lang}: {e}")
                continue

        # Group dependencies by module name
        dep_map: Dict[str, List["File"]] = defaultdict(list)
        for sym in dependency_symbols:
            dep_map[sym.name].append(sym.file)

        # Create Dependency objects
        for name, imported_by in dep_map.items():
            dependency = Dependency(
                id=str(hash(name)), name=name, imported_by=imported_by
            )
            codebase.dependencies.append(dependency)

        # Add dependency symbols to codebase symbols
        codebase.symbols.extend(dependency_symbols)
