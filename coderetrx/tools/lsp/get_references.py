"""Get References LSP tool - finds all references to a symbol."""

import logging
from pathlib import Path
from typing import ClassVar, List, Type
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from coderetrx.tools.lsp.base import LSPBaseTool
from coderetrx.utils.path import safe_join
from lspyc.handle.protocol import Location

logger = logging.getLogger(__name__)


class GetReferencesArgs(BaseModel):
    """Arguments for get_references tool."""

    file_path: str = Field(
        description="Path to file relative to repository root (e.g., 'src/main.py')"
    )
    line: int = Field(ge=0, description="Line number (0-based by default)")
    column: int = Field(ge=0, description="Column number (0-based by default)")
    include_declaration: bool = Field(
        default=True, description="Include the symbol declaration in results"
    )
    include_context: bool = Field(
        default=True, description="Include code context snippets (±1 lines)"
    )
    zero_based: bool = Field(
        default=True,
        description="Use 0-based indexing (default: True)",
    )


class GetReferencesResult(BaseModel):
    """Result entry for a symbol reference."""

    symbol_name: str = Field(description="Name of the symbol")
    reference_file: str = Field(description="File path relative to repository root")
    reference_line: int = Field(description="Line number (0-based by default)")
    reference_column: int = Field(description="Column number (0-based by default)")
    context: str = Field(default="", description="Code snippet context (±1 lines)")

    @classmethod
    def repr(cls, entries: List["GetReferencesResult"]) -> str:
        """Format references results grouped by file with context.

        Args:
            entries: List of reference results

        Returns:
            Formatted string representation
        """
        if not entries:
            return "No references found."

        # Extract symbol name (should be same for all)
        symbol_name = entries[0].symbol_name if entries else "unknown"

        # Check if we need to show limit message
        total_count = len(entries)
        display_limit = 100
        limited = total_count > display_limit
        entries_to_show = entries[:display_limit] if limited else entries

        # Group by file
        by_file: dict[str, List[GetReferencesResult]] = {}
        for entry in entries_to_show:
            if entry.reference_file not in by_file:
                by_file[entry.reference_file] = []
            by_file[entry.reference_file].append(entry)

        # Format output
        result = [f"Found {total_count} reference(s) to '{symbol_name}':"]
        if limited:
            result[0] += f" (showing first {display_limit})"

        result.append("")  # Blank line

        for file_path in sorted(by_file.keys()):
            refs = by_file[file_path]
            # Sort by line number
            refs.sort(key=lambda r: r.reference_line)

            result.append(f"## {file_path} ({len(refs)} reference(s))")
            for i, ref in enumerate(refs, 1):
                location = f"{ref.reference_line}:{ref.reference_column}"
                result.append(f"{i}. Line {location}")
                if ref.context:
                    # Indent context
                    for line in ref.context.split("\n"):
                        result.append(f"   {line}")
            result.append("")  # Blank line between files

        return "\n".join(result).rstrip()


class GetReferencesTool(LSPBaseTool):
    """Tool for finding all references to a symbol using LSP."""

    name = "get_references"
    description = (
        "Find all references to a symbol (find usages).\n"
        "Provide a file path and position (line, column) to find all references.\n"
        "Uses Language Server Protocol for accurate semantic analysis.\n"
        "Supports Python, JavaScript, TypeScript, Rust, Go, Java, Kotlin, C/C++.\n\n"
        "Requirements:\n"
        "- File must exist and be a supported language\n"
        "- Appropriate language server must be installed and available\n"
        "- Position must be on a valid symbol"
    )
    args_schema: ClassVar[Type[GetReferencesArgs]] = GetReferencesArgs

    def forward(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
        include_context: bool = True,
        zero_based: bool = True,
    ) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(
            file_path=file_path,
            line=line,
            column=column,
            include_declaration=include_declaration,
            include_context=include_context,
            zero_based=zero_based,
        )

    async def _run(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
        include_context: bool = True,
        zero_based: bool = True,
    ) -> List[GetReferencesResult]:
        """Find all references to symbol at given position.

        Args:
            file_path: Path to file relative to repository root
            line: Line number (0-based by default)
            column: Column number (0-based by default)
            include_declaration: Include symbol declaration in results
            include_context: Include code context snippets
            zero_based: Use 0-based indexing (default: True)

        Returns:
            List of reference results
        """
        try:
            # Validate file exists
            file_path = file_path.lstrip("/")
            full_path = safe_join(self.repo_path, file_path)
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return []

            # Convert line/column to 0-based for LSP
            lsp_line = self._convert_index(line, zero_based)
            lsp_column = self._convert_index(column, zero_based)

            # Extract symbol name at position
            symbol_name = self._extract_symbol_name(full_path, lsp_line, lsp_column)

            # Get LSP client from pool
            client = await self._get_lsp_client()

            # Get reference locations
            locations = await client.get_references(
                str(full_path), lsp_line, lsp_column, include_declaration
            )

            if not locations:
                return []

            # Process locations
            results = []
            for location in locations:
                # Convert URI to relative path
                relative_path = self._uri_to_relative_path(location["uri"])

                # Extract position
                ref_line = location["range"]["start"]["line"]
                ref_column = location["range"]["start"]["character"]

                # Convert to output format
                output_line = self._convert_to_output_index(ref_line, zero_based)
                output_column = self._convert_to_output_index(ref_column, zero_based)

                # Extract context if requested
                context = ""
                if include_context:
                    context = self._extract_context(
                        location["uri"], ref_line, location["range"]
                    )

                result = GetReferencesResult(
                    symbol_name=symbol_name,
                    reference_file=relative_path,
                    reference_line=output_line,
                    reference_column=output_column,
                    context=context,
                )
                results.append(result)

            # Sort by file, then line number
            results.sort(key=lambda r: (r.reference_file, r.reference_line))

            return results

        except Exception as e:
            logger.warning(f"LSP get_references operation failed: {e}")
            return []

    def _uri_to_relative_path(self, uri: str) -> str:
        """Convert file URI to path relative to repository root.

        Args:
            uri: File URI (e.g., 'file:///path/to/file.py')

        Returns:
            Path relative to repository root
        """
        try:
            # Parse URI and decode percent-encoded characters
            parsed = urlparse(uri)
            abs_path = Path(unquote(parsed.path))

            # Try to make relative to repo path
            try:
                relative = abs_path.relative_to(self.repo_path)
                return str(relative)
            except ValueError:
                # Path is outside repository (e.g., external library)
                return str(abs_path)

        except Exception as e:
            logger.warning(f"Failed to convert URI to relative path: {e}")
            return uri

    def _extract_context(self, uri: str, line: int, range_dict: dict) -> str:
        """Extract code context around a reference (±1 lines).

        Args:
            uri: File URI
            line: Line number (0-based)
            range_dict: LSP Range dictionary

        Returns:
            Context string with ±1 lines, or empty string on error
        """
        try:
            # Parse URI to get file path
            parsed = urlparse(uri)
            file_path = Path(unquote(parsed.path))

            if not file_path.exists():
                return ""

            # Read file lines
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Extract context: 1 line before, current line, 1 line after
            start_line = max(0, line - 1)
            end_line = min(len(lines), line + 2)

            context_lines = []
            for i in range(start_line, end_line):
                # Truncate very long lines
                line_text = lines[i].rstrip()
                if len(line_text) > 100:
                    line_text = line_text[:97] + "..."
                context_lines.append(line_text)

            return "\n".join(context_lines)

        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")
            return ""
