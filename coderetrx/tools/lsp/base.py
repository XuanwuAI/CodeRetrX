"""Base class for LSP-based tools with shared client pool."""

import asyncio
import logging
from pathlib import Path

from lspyc import ThreadedClient

from coderetrx.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Global client pool: repo_path -> ThreadedClient
_lsp_client_pool: dict[str, ThreadedClient] = {}
_pool_lock = asyncio.Lock()


class LSPBaseTool(BaseTool):
    """Base class for LSP-based tools with shared client pool and utilities."""

    async def _get_lsp_client(self) -> ThreadedClient:
        """Get or create an LSP client for the repository from the pool.

        Returns:
            ThreadedClient instance for this repository

        Note:
            Clients are cached in the pool and reused across tool invocations
            for better performance (avoids repeated LSP server startup overhead).
        """
        async with _pool_lock:
            repo_key = str(self.repo_path)
            if repo_key not in _lsp_client_pool:
                logger.info(f"Creating new LSP client for {repo_key}")
                _lsp_client_pool[repo_key] = ThreadedClient(repo_key)
            return _lsp_client_pool[repo_key]

    @staticmethod
    async def cleanup_client_pool():
        """Cleanup all clients in the pool - call on tool shutdown.

        This should be called when the MCP server is shutting down to
        gracefully close all LSP server connections.
        """
        async with _pool_lock:
            logger.info(f"Cleaning up {len(_lsp_client_pool)} LSP clients")
            for repo_path, client in _lsp_client_pool.items():
                try:
                    await client.ashutdown()
                    logger.info(f"Successfully shutdown LSP client for {repo_path}")
                except Exception as e:
                    logger.warning(
                        f"Error shutting down LSP client for {repo_path}: {e}"
                    )
            _lsp_client_pool.clear()

    def _extract_symbol_name(self, file_path: Path, line: int, column: int) -> str:
        """Extract symbol name at given position (0-based indices).

        Args:
            file_path: Path to the file
            line: Line number (0-based)
            column: Column number (0-based)

        Returns:
            Symbol name at the position, or "unknown" if extraction fails
        """
        try:
            if not file_path.exists():
                return "unknown"

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if line < 0 or line >= len(lines):
                return "unknown"

            line_text = lines[line]
            if column < 0 or column >= len(line_text):
                return "unknown"

            # Extract word at position using word boundaries
            # Find start of word
            start = column
            while start > 0 and (
                line_text[start - 1].isalnum() or line_text[start - 1] == "_"
            ):
                start -= 1

            # Find end of word
            end = column
            while end < len(line_text) and (
                line_text[end].isalnum() or line_text[end] == "_"
            ):
                end += 1

            symbol = line_text[start:end].strip()
            return symbol if symbol else "unknown"

        except Exception as e:
            logger.warning(f"Failed to extract symbol name: {e}")
            return "unknown"

    def _convert_index(self, value: int, zero_based: bool) -> int:
        """Convert between 1-based and 0-based indexing.

        Args:
            value: The value to convert
            zero_based: If True, value is already 0-based; if False, value is 1-based

        Returns:
            0-based index value
        """
        return value if zero_based else value - 1

    def _convert_to_output_index(self, value: int, zero_based: bool) -> int:
        """Convert 0-based index to output format (1-based or 0-based).

        Args:
            value: The 0-based value to convert
            zero_based: If True, keep as 0-based; if False, convert to 1-based

        Returns:
            Index in the desired output format
        """
        return value if zero_based else value + 1
