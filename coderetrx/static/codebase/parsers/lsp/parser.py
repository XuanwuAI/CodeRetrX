"""LSP-based parser implementation using lspyc client.

This module provides a parser that uses Language Server Protocol (LSP) to extract
code chunks and symbols from source files. It leverages the lspyc multi-language
LSP client for communication with language servers.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from tqdm.asyncio import tqdm as tqdm_asyncio

from ..base import CodebaseParser, ParseError
from ...languages import (
    IDXSupportedLanguage,
    IDXSupportedTag,
    OBJLIKE_TAGS,
)
from coderetrx.utils.concurrency import run_coroutine_sync
from .symbol_mapping import symbolkind_to_tag, symbolkind_to_chunk_type

if TYPE_CHECKING:
    from ...codebase import File, CodeChunk, Codebase
    from lspyc.mlclient import MutilLangClient
    from lspyc.handle.protocol import DocumentSymbol

logger = logging.getLogger(__name__)


class LSPParser(CodebaseParser):
    """
    LSP-based parser implementation.

    Uses Language Server Protocol to extract code structures from source files.
    Supports multiple languages through the lspyc multi-language client.
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

    def supports_language(self, language: IDXSupportedLanguage) -> bool:
        """Check if LSP supports the given language.

        Args:
            language: The language to check

        Returns:
            True if supported, False otherwise
        """
        return language in self.SUPPORTED_LANGUAGES

    def _ensure_client(self, codebase: "Codebase") -> "MutilLangClient":
        """Ensure LSP client is initialized.

        Args:
            codebase: The codebase being parsed

        Returns:
            Initialized MutilLangClient instance
        """
        if self._client is None:
            from lspyc.mlclient import MutilLangClient

            workspace_root = self._workspace_root or codebase.dir
            self._client = MutilLangClient(str(workspace_root))

        return self._client

    def init_chunks(self, codebase: "Codebase") -> None:
        """Parse codebase and populate chunks using LSP document symbols.

        This method:
        1. Initializes LSP client
        2. Gets document symbols for each source file
        3. Converts symbols to CodeChunk objects
        4. Populates file.chunks and codebase.all_chunks

        Args:
            codebase: The codebase to parse
        """
        run_coroutine_sync(self._init_chunks_async(codebase))

    async def _init_chunks_async(self, codebase: "Codebase") -> None:
        """Async implementation of chunk initialization with concurrent file processing.

        Processes multiple files concurrently to improve performance. Uses a semaphore
        to limit concurrent LSP requests and prevent overwhelming the language servers.

        Args:
            codebase: The codebase to parse
        """
        client = self._ensure_client(codebase)
        extract_variables = self.config.get("extract_variable_definitions", False)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)

        # Lock for thread-safe access to shared codebase state
        chunks_lock = asyncio.Lock()

        # Filter files to process
        files_to_process = [
            file for file in codebase.source_files.values()
            if (lang := file.lang()) is not None and self.supports_language(lang)
        ]

        async def process_file(file: "File") -> None:
            """Process a single file with semaphore control."""
            async with semaphore:
                try:
                    # Get document symbols from LSP
                    symbols = await client.get_document_symbols(str(file.path))

                    if not symbols:
                        return

                    # Process symbols (CPU-bound, but fast)
                    chunks = self._process_symbols(
                        symbols, file, extract_variables=extract_variables
                    )

                    # Thread-safe update of shared state
                    async with chunks_lock:
                        file.chunks.extend(chunks)
                        codebase.all_chunks.extend(chunks)

                except Exception as e:
                    logger.warning(f"Failed to parse {file.path} with LSP: {e}")

        # Create all tasks
        tasks = [process_file(file) for file in files_to_process]

        # Execute concurrently with progress bar
        if tasks:
            await tqdm_asyncio.gather(
                *tasks,
                desc="LSP parsing files",
                total=len(tasks),
            )

    def _process_symbols(
        self,
        symbols: List["DocumentSymbol"],
        file: "File",
        parent_chunk: Optional["CodeChunk"] = None,
        extract_variables: bool = False,
    ) -> List["CodeChunk"]:
        """Recursively process LSP document symbols into CodeChunk objects.

        Args:
            symbols: List of LSP document symbols
            file: The source file being parsed
            parent_chunk: Parent chunk for nested symbols
            extract_variables: Whether to extract variable definitions

        Returns:
            List of CodeChunk objects
        """
        chunks: List["CodeChunk"] = []

        for symbol in symbols:
            # Determine if this symbol should be extracted
            kind = symbol.get("kind")
            if kind is None:
                continue

            tag = symbolkind_to_tag(kind, extract_variables=extract_variables)
            if tag is None:
                # Symbol kind not supported or variables not enabled
                continue

            # Create chunk from symbol
            try:
                chunk = self._create_chunk_from_symbol(symbol, file, tag)
                chunk.parent = parent_chunk
                chunks.append(chunk)

                # Process children recursively
                children = symbol.get("children", [])
                if children:
                    child_chunks = self._process_symbols(
                        children, file, parent_chunk=chunk, extract_variables=extract_variables
                    )
                    chunks.extend(child_chunks)

            except Exception as e:
                logger.warning(
                    f"Failed to create chunk from symbol {symbol.get('name', 'unknown')} "
                    f"in {file.path}: {e}"
                )
                continue

        return chunks

    def _create_chunk_from_symbol(
        self, symbol: "DocumentSymbol", file: "File", tag: IDXSupportedTag
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

    def init_symbols(self, codebase: "Codebase") -> None:
        """Extract symbols from chunks and populate codebase.symbols.

        This method creates Symbol objects from PRIMARY chunks, similar to
        the TreeSitter parser implementation.

        Args:
            codebase: The codebase to populate
        """
        # Ensure chunks are populated first
        if not codebase.all_chunks:
            self.init_chunks(codebase)

        from ...codebase import Symbol, ChunkType

        # Create Symbol objects from PRIMARY chunks
        for chunk in codebase.all_chunks:
            symbol_name = chunk.symbol_name()
            if chunk.type == ChunkType.PRIMARY and symbol_name:
                # Determine symbol type based on tag
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
        """Extract dependencies from codebase.

        LSP doesn't provide direct import extraction, so we fallback to
        tree-sitter for dependency analysis.

        Args:
            codebase: The codebase to populate
        """
        logger.info("LSP parser: Using tree-sitter fallback for dependency extraction")

        try:
            from ..treesitter import TreeSitterParser

            ts_parser = TreeSitterParser(**self.config)
            ts_parser.init_dependencies(codebase)

        except Exception as e:
            logger.error(f"Failed to extract dependencies with tree-sitter fallback: {e}")
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
