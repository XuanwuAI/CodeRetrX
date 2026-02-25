"""List Symbol LSP tool - lists all symbols in a file."""

import logging
from pathlib import Path
from typing import ClassVar, List, Type

from pydantic import BaseModel, Field

from coderetrx.tools.lsp.base import LSPBaseTool
from coderetrx.tools.lsp.constants import get_symbol_kind_name
from coderetrx.utils.path import safe_join
from lspyc.handle.protocol import DocumentSymbol

logger = logging.getLogger(__name__)


class ListSymbolArgs(BaseModel):
    """Arguments for list_symbol tool."""

    file_path: str = Field(
        description="Path to file relative to repository root (e.g., 'src/main.py')"
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum hierarchy depth to display (1-5, default: 3)",
    )
    zero_based: bool = Field(
        default=True,
        description="Use 0-based indexing for line/column numbers (default: True)",
    )


class ListSymbolResult(BaseModel):
    """Result entry for a symbol."""

    name: str = Field(description="Symbol name")
    kind: str = Field(description="Symbol kind (function, class, etc.)")
    line: int = Field(description="Line number (0-based by default)")
    column: int = Field(description="Column number (0-based by default)")
    detail: str = Field(default="", description="Additional symbol details")
    depth: int = Field(description="Nesting depth (0 = top-level)")

    @classmethod
    def repr(cls, entries: List["ListSymbolResult"]) -> str:
        """Format symbol list as an indented tree structure.

        Args:
            entries: List of symbol results

        Returns:
            Formatted string representation
        """
        if not entries:
            return "No symbols found in file."

        result = []
        for entry in entries:
            indent = "  " * entry.depth
            location = f"{entry.line}:{entry.column}"
            detail_str = f" - {entry.detail}" if entry.detail else ""
            result.append(f"{indent}{entry.kind} {entry.name} @ {location}{detail_str}")

        return "\n".join(result)


class ListSymbolTool(LSPBaseTool):
    """Tool for listing all symbols in a file using LSP."""

    name = "list_symbol"
    description = (
        "List all symbols (functions, classes, methods, variables, etc.) in a file.\n"
        "Uses Language Server Protocol to extract symbols in hierarchical format.\n"
        "Supports Python, JavaScript, TypeScript, Rust, Go, Java, Kotlin, C/C++.\n\n"
        "Requirements:\n"
        "- File must exist and be a supported language\n"
        "- Appropriate language server must be installed and available"
    )
    args_schema: ClassVar[Type[ListSymbolArgs]] = ListSymbolArgs

    def forward(self, file_path: str, max_depth: int = 3, zero_based: bool = True) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(file_path=file_path, max_depth=max_depth, zero_based=zero_based)

    async def _run(
        self, file_path: str, max_depth: int = 3, zero_based: bool = True
    ) -> List[ListSymbolResult]:
        """List all symbols in a file.

        Args:
            file_path: Path to file relative to repository root
            max_depth: Maximum hierarchy depth to display (1-5)
            zero_based: Use 0-based indexing (default: True)

        Returns:
            List of symbol results
        """
        try:
            # Validate file exists
            file_path = file_path.lstrip("/")
            full_path = safe_join(self.repo_path, file_path)
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return []

            # Get LSP client from pool
            client = await self._get_lsp_client()

            # Get document symbols
            symbols = await client.aget_document_symbols(str(full_path))

            if not symbols:
                return []

            # Recursively flatten symbol hierarchy
            results = []
            self._flatten_symbols(symbols, results, max_depth, zero_based, depth=0)

            return results

        except Exception as e:
            logger.warning(f"LSP list_symbol operation failed: {e}")
            return []

    def _flatten_symbols(
        self,
        symbols: List[DocumentSymbol],
        results: List[ListSymbolResult],
        max_depth: int,
        zero_based: bool,
        depth: int = 0,
    ) -> None:
        """Recursively flatten symbol hierarchy into results list.

        Args:
            symbols: List of DocumentSymbol objects
            results: Output list to append results to
            max_depth: Maximum depth to traverse
            zero_based: Use 0-based indexing
            depth: Current depth level
        """
        if depth >= max_depth:
            return

        for symbol in symbols:
            # Extract position from selectionRange (the identifier location)
            line = symbol["selectionRange"]["start"]["line"]
            column = symbol["selectionRange"]["start"]["character"]

            # Convert to output format
            output_line = self._convert_to_output_index(line, zero_based)
            output_column = self._convert_to_output_index(column, zero_based)

            # Get symbol kind name
            kind_name = get_symbol_kind_name(symbol["kind"])

            # Get detail if available
            detail = symbol.get("detail", "")

            result = ListSymbolResult(
                name=symbol["name"],
                kind=kind_name,
                line=output_line,
                column=output_column,
                detail=detail,
                depth=depth,
            )
            results.append(result)

            # Recursively process children
            if "children" in symbol and symbol["children"]:
                self._flatten_symbols(
                    symbol["children"], results, max_depth, zero_based, depth + 1
                )
