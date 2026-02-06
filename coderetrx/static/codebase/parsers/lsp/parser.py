"""LSP-based parser implementation using lspyc client.

This module provides a parser that uses Language Server Protocol (LSP) to extract
code chunks and symbols from source files. It leverages the lspyc multi-language
LSP client for communication with language servers.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from lspyc import MutilLangClient
from lspyc.handle.protocol import DocumentSymbol
from tqdm.asyncio import tqdm as tqdm_asyncio

from coderetrx.utils.concurrency import run_coroutine_sync

from ...languages import IDXSupportedLanguage, IDXSupportedTag
from ..base import CodebaseParser, ParseError
from ..treesitter import TreeSitterParser
from ..treesitter.parser import TreeSitterState
from ..treesitter.queries import TreeSitterQueryTemplates
from .symbol_mapping import (
    SYMBOLKIND_CLASS,
    SYMBOLKIND_CONSTRUCTOR,
    SYMBOLKIND_ENUM,
    SYMBOLKIND_FUNCTION,
    SYMBOLKIND_INTERFACE,
    SYMBOLKIND_METHOD,
    SYMBOLKIND_STRUCT,
    symbolkind_to_chunk_type,
    symbolkind_to_tag,
)

if TYPE_CHECKING:

    from ...codebase import Codebase, CodeChunk, File, Symbol

logger = logging.getLogger(__name__)


class LSPParser(CodebaseParser):
    """
    LSP-based parser implementation.

    Uses Language Server Protocol to extract code structures from source files.
    Supports multiple languages through the lspyc multi-language client.

    This parser follows a symbols-first architecture where LSP DocumentSymbols
    are used to create both Symbol objects and their corresponding CodeChunk
    representations. TreeSitter is used as a fallback for imports and references.
    """

    # Languages supported by lspyc DEFAULT_NATIVE_FACTORIES
    SUPPORTED_LANGUAGES: List[IDXSupportedLanguage] = [
        "python",
        "javascript",
        "typescript",
        "c",
        "cpp",
        "rust",
        "go",
        "java",
    ]

    def __init__(self, workspace_root: Optional[Path] = None, **kwargs):
        """Initialize the LSP parser.

        Args:
            workspace_root: Root directory for LSP workspace (defaults to codebase dir)
            **kwargs: Additional configuration options:
                - extract_variable_definitions: Whether to extract variable definitions
                - max_concurrent_requests: Maximum concurrent LSP requests (default: 10)
        """
        super().__init__(**kwargs)
        self._workspace_root = workspace_root
        self._client: Optional["MutilLangClient"] = None
        self._max_concurrent_requests = kwargs.get("max_concurrent_requests", 10)

        # Initialize TreeSitter fallback for imports/references
        self._treesitter_fallback = None

    def _get_treesitter_fallback(self) -> "TreeSitterParser":
        """Get or create TreeSitter parser for fallback operations.

        Returns:
            TreeSitterParser instance
        """
        if self._treesitter_fallback is None:
            self._treesitter_fallback = TreeSitterParser(**self.config)
        return self._treesitter_fallback

    def supports_language(self, language: IDXSupportedLanguage) -> bool:
        """Check if LSP or TreeSitter fallback supports the given language.

        Args:
            language: The language to check

        Returns:
            True if supported, False otherwise
        """
        # Check LSP support first, fallback to TreeSitter
        return (
            language in self.SUPPORTED_LANGUAGES
            or self._get_treesitter_fallback().supports_language(language)
        )

    def _ensure_client(self, codebase: "Codebase") -> "MutilLangClient":
        """Ensure LSP client is initialized.

        Args:
            codebase: The codebase being parsed

        Returns:
            Initialized MutilLangClient instance
        """
        if self._client is None:

            workspace_root = self._workspace_root or codebase.dir
            self._client = MutilLangClient(str(workspace_root))

        return self._client

    def init_chunks(self, codebase: "Codebase") -> None:
        """Parse codebase and populate chunks using LSP + TreeSitter fallback.

        For LSP parser, chunks are created as representations of symbols.
        This method:
        1. Calls init_symbols() which creates both symbols AND their chunk representations
        2. Uses TreeSitter fallback to extract imports (which LSP doesn't provide)

        Args:
            codebase: The codebase to parse
        """
        # Step 1: LSP for symbols (creates symbol chunks)
        if not codebase._symbols_initialized:
            self.init_symbols(codebase)

        # Step 2: TreeSitter fallback for imports
        run_coroutine_sync(self._extract_import_chunks_fallback(codebase))

    async def _extract_import_chunks_fallback(self, codebase: "Codebase") -> None:
        """Use TreeSitter to extract import chunks as fallback.

        LSP DocumentSymbols don't include imports, so we use TreeSitter
        queries to extract them.

        Args:
            codebase: The codebase to populate with import chunks
        """
        from ...codebase import ChunkType

        ts_parser = self._get_treesitter_fallback()

        for file in codebase.source_files.values():
            if file.file_type != "source":
                continue

            # Skip if file has no symbols (LSP couldn't parse it)
            if not file.chunks:
                continue

            # Skip if TreeSitter doesn't support the language
            lang = file.lang()
            if lang is None or not ts_parser.supports_language(lang):
                continue

            try:
                parse_state: TreeSitterState = ts_parser.parse_file(file)
                try:
                    pattern = TreeSitterQueryTemplates.get_query(
                        parse_state.idx_language, "fine_imports"
                    )
                except FileNotFoundError:
                    # No import query for this language
                    continue

                query = ts_parser._compile_query(parse_state.ts_language, pattern)
                captures = ts_parser._execute_query_captures(
                    query, parse_state.tree.root_node
                )

                # Extract import chunks from captures
                for capture_name, nodes in captures.items():
                    if capture_name != "module":
                        continue

                    for node in nodes:
                        raw = node.text.decode() if node.text else ""
                        # Strip quotes, angle brackets, etc.
                        module_name = raw.strip("'\"<>")

                        # Skip empty or local imports
                        if not module_name or module_name.startswith((".", "/")):
                            continue

                        # Create import chunk
                        chunk = ts_parser.create_chunk_from_node(
                            node, file, ChunkType.IMPORT, name=module_name
                        )

                        # Check if chunk already exists (avoid duplicates)
                        if chunk not in file.chunks:
                            file.chunks.append(chunk)
                            codebase.all_chunks.append(chunk)

            except Exception as e:
                logger.debug(
                    f"TreeSitter fallback failed for imports in {file.path}: {e}"
                )
                continue

    def _process_document_symbols(
        self,
        doc_symbols: List[DocumentSymbol],
        file: "File",
        parent: Optional["Symbol"] = None,
        extract_variables: bool = False,
    ) -> Tuple[List["Symbol"], List["CodeChunk"]]:
        """Process LSP DocumentSymbols into Symbol and CodeChunk objects (symbols-first).

        This is the core method for the symbols-first architecture. For each DocumentSymbol:
        1. Determine if it should create a Symbol (classes/functions) or just a chunk (variables)
        2. Create CodeChunk as code representation
        3. Create Symbol if applicable
        4. Process children recursively, maintaining parent-child relationships

        Args:
            doc_symbols: List of LSP DocumentSymbols
            file: The source file being parsed
            parent: Parent symbol for nested symbols (e.g., method's parent is class)
            extract_variables: Whether to extract variable definitions

        Returns:
            Tuple of (symbols, chunks) created from the document symbols
        """
        from ...codebase import Symbol

        symbols: List["Symbol"] = []
        chunks: List["CodeChunk"] = []

        for doc_sym in doc_symbols:
            kind = doc_sym.get("kind")
            if kind is None:
                continue

            # Check if this symbol kind should be extracted
            tag = symbolkind_to_tag(kind, extract_variables=extract_variables)
            if tag is None:
                # Skip unsupported symbol kinds
                continue

            try:
                # Create CodeChunk first (as code representation)
                chunk = self._create_chunk_from_symbol(doc_sym, file, tag)
                chunk.parent = parent.chunk if parent else None
                chunks.append(chunk)

                # Determine if this should create a Symbol object
                # Only classes and functions become Symbol objects
                should_create_symbol = kind in {
                    SYMBOLKIND_CLASS,
                    SYMBOLKIND_STRUCT,
                    SYMBOLKIND_ENUM,
                    SYMBOLKIND_INTERFACE,
                    SYMBOLKIND_METHOD,
                    SYMBOLKIND_FUNCTION,
                    SYMBOLKIND_CONSTRUCTOR,
                }

                symbol = None
                if should_create_symbol:
                    # Determine symbol type
                    if kind in {
                        SYMBOLKIND_CLASS,
                        SYMBOLKIND_STRUCT,
                        SYMBOLKIND_ENUM,
                        SYMBOLKIND_INTERFACE,
                    }:
                        symbol_type = "class"
                    else:
                        symbol_type = "function"

                    # Build qualified name (e.g., "ClassName.method_name")
                    symbol_name = self._get_qualified_name(doc_sym, parent)

                    # Extract selection range from LSP DocumentSymbol
                    selection_range = doc_sym.get("selectionRange")

                    # Create Symbol from DocumentSymbol
                    symbol = Symbol(
                        name=symbol_name,
                        type=symbol_type,
                        file=file,
                        chunk=chunk,
                        id=chunk.id,
                        selection_range=selection_range,
                    )
                    symbols.append(symbol)

                # Process children recursively
                children = doc_sym.get("children", [])
                if children:
                    child_symbols, child_chunks = self._process_document_symbols(
                        children,
                        file,
                        parent=symbol,
                        extract_variables=extract_variables,
                    )
                    symbols.extend(child_symbols)
                    chunks.extend(child_chunks)

            except Exception as e:
                logger.warning(
                    f"Failed to process symbol {doc_sym.get('name', 'unknown')} "
                    f"in {file.path}: {e}"
                )
                continue

        return symbols, chunks

    def _create_chunk_from_symbol(
        self, symbol: DocumentSymbol, file: "File", tag: IDXSupportedTag
    ) -> "CodeChunk":
        """Create a CodeChunk from an LSP DocumentSymbol.

        Args:
            symbol: LSP document symbol
            file: The source file
            tag: CodeRetrX tag for the symbol

        Returns:
            CodeChunk object

        Raises:
            ValueError: If symbol range is invalid
        """
        from ...codebase import CodeChunk

        # Extract range information (LSP uses 0-based lines and columns)
        # range_data = symbol.get("range", {})
        range_data = symbol.get("range", {})
        start = range_data.get("start", {})
        end = range_data.get("end", {})

        start_line = start.get("line", 0)
        start_column = start.get("character", 0)
        end_line = end.get("line", 0)
        end_column = end.get("character", 0)

        # Determine chunk type
        kind = symbol.get("kind", 0)
        chunk_type = symbolkind_to_chunk_type(kind)

        # Create chunk with deterministic UUID
        chunk = CodeChunk.new(
            src=file,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            chunk_type=chunk_type,
            tag=tag,
            name=symbol.get("name"),
        )

        return chunk

    def _get_qualified_name(
        self, doc_sym: DocumentSymbol, parent: Optional["Symbol"]
    ) -> str:
        """Build qualified symbol name from LSP hierarchy.

        For nested symbols (e.g., methods inside classes), this creates
        a qualified name like "ClassName.method_name".

        Args:
            doc_sym: LSP DocumentSymbol
            parent: Parent symbol (if nested)

        Returns:
            Qualified symbol name
        """
        name = doc_sym.get("name", "")

        # If symbol has a class parent, prefix with parent class name
        if parent and parent.type == "class":
            return f"{parent.name}.{name}"

        return name

    def init_symbols(self, codebase: "Codebase") -> None:
        """Extract symbols using LSP DocumentSymbols (symbols-first approach).

        This is the primary method for LSP parser. It:
        1. Gets DocumentSymbols from LSP for each file
        2. Creates Symbol objects directly from DocumentSymbols
        3. Creates CodeChunk objects as code representations for symbols
        4. Handles nested symbols (e.g., methods inside classes)

        Unlike TreeSitter which extracts chunks first then identifies symbols,
        LSP provides symbols directly, so we create both Symbol and CodeChunk
        from each DocumentSymbol.

        Args:
            codebase: The codebase to populate
        """
        run_coroutine_sync(self._init_symbols_async(codebase))

    async def _init_symbols_async(self, codebase: "Codebase") -> None:
        """Async implementation of symbol extraction with concurrent file processing.

        Args:
            codebase: The codebase to populate with symbols and chunks
        """
        client = self._ensure_client(codebase)
        extract_variables = self.config.get("extract_variable_definitions", False)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)

        # Lock for thread-safe access to shared codebase state
        state_lock = asyncio.Lock()

        # Filter files to process (only LSP-supported languages)
        files_to_process = [
            file
            for file in codebase.source_files.values()
            if (lang := file.lang()) is not None and lang in self.SUPPORTED_LANGUAGES
        ]

        async def process_file(file: "File") -> None:
            """Process a single file with semaphore control."""
            async with semaphore:
                try:
                    # Get DocumentSymbols from LSP
                    doc_symbols = await client.get_document_symbols(str(file.path))

                    if not doc_symbols:
                        return

                    # Process symbols recursively (creates both Symbol and CodeChunk)
                    symbols, chunks = self._process_document_symbols(
                        doc_symbols,
                        file,
                        parent=None,
                        extract_variables=extract_variables,
                    )

                    # Thread-safe update of shared state
                    async with state_lock:
                        file.chunks.extend(chunks)
                        codebase.all_chunks.extend(chunks)
                        codebase.symbols.extend(symbols)

                except Exception as e:
                    logger.warning(
                        f"Failed to extract symbols from {file.path} with LSP: {e}"
                    )

        # Create all tasks
        tasks = [process_file(file) for file in files_to_process]

        # Execute concurrently with progress bar
        if tasks:
            await tqdm_asyncio.gather(
                *tasks,
                desc="LSP extracting symbols",
                total=len(tasks),
            )

    def init_dependencies(self, codebase: "Codebase") -> None:
        """Extract dependencies from codebase.

        LSP doesn't provide direct import extraction, so we fallback to
        tree-sitter for dependency analysis.

        Args:
            codebase: The codebase to populate
        """
        logger.info("LSP parser: Using tree-sitter fallback for dependency extraction")

        try:
            ts_parser = self._get_treesitter_fallback()
            ts_parser.init_dependencies(codebase)

        except Exception as e:
            logger.error(
                f"Failed to extract dependencies with tree-sitter fallback: {e}"
            )
            raise ParseError(f"Dependency extraction failed: {e}") from e

    def cleanup(self):
        """Clean up LSP client resources.

        Shuts down all LSP servers and closes connections.
        """
        if self._client is not None:
            try:
                run_coroutine_sync(self._client.shutdown())
            except Exception as e:
                logger.warning(f"Error during LSP client shutdown: {e}")
            finally:
                self._client = None
