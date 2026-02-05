"""Get Definition LSP tool - finds where a symbol is defined."""

import logging
from pathlib import Path
from typing import ClassVar, List, Type
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from coderetrx.tools.lsp.base import LSPBaseTool
from coderetrx.utils.path import safe_join

logger = logging.getLogger(__name__)


class GetDefinitionArgs(BaseModel):
    """Arguments for get_definition tool."""

    file_path: str = Field(
        description="Path to file relative to repository root (e.g., 'src/main.py')"
    )
    line: int = Field(ge=0, description="Line number (0-based by default)")
    column: int = Field(ge=0, description="Column number (0-based by default)")
    include_context: bool = Field(
        default=True,
        description="Include code context (target line + 5 lines after)",
    )
    zero_based: bool = Field(
        default=True,
        description="Use 0-based indexing (default: True)",
    )


class GetDefinitionResult(BaseModel):
    """Result entry for a symbol definition."""

    symbol_name: str = Field(description="Name of the symbol")
    definition_file: str = Field(description="File path relative to repository root")
    definition_line: int = Field(description="Line number (0-based by default)")
    definition_column: int = Field(description="Column number (0-based by default)")
    context: str = Field(
        default="", description="Code snippet (target line + 5 lines after)"
    )

    @classmethod
    def repr(cls, entries: List["GetDefinitionResult"]) -> str:
        """Format definition results as a numbered list.

        Args:
            entries: List of definition results

        Returns:
            Formatted string representation
        """
        if not entries:
            return "No definition found."

        if len(entries) == 1:
            entry = entries[0]
            result = [
                f"Definition of '{entry.symbol_name}':",
                f"  {entry.definition_file}:{entry.definition_line}:{entry.definition_column}",
            ]
            if entry.context:
                result.append("")  # Blank line
                # Indent context
                for line in entry.context.split("\n"):
                    result.append(f"   {line}")
            return "\n".join(result)

        # Multiple definitions (e.g., overloads)
        result = [f"Found {len(entries)} definitions:"]
        for i, entry in enumerate(entries, 1):
            result.append(
                f"{i}. {entry.definition_file}:{entry.definition_line}:{entry.definition_column}"
            )
            if entry.context:
                for line in entry.context.split("\n"):
                    result.append(f"      {line}")
                result.append("")  # Blank line between entries
        return "\n".join(result).rstrip()


class GetDefinitionTool(LSPBaseTool):
    """Tool for finding symbol definitions using LSP."""

    name = "get_definition"
    description = (
        "Find where a symbol is defined (go to definition).\n"
        "Provide a file path and position (line, column) to find the definition location.\n"
        "Uses Language Server Protocol for accurate semantic analysis.\n"
        "Supports Python, JavaScript, TypeScript, Rust, Go, Java, Kotlin, C/C++.\n\n"
        "Requirements:\n"
        "- File must exist and be a supported language\n"
        "- Appropriate language server must be installed and available\n"
        "- Position must be on a valid symbol"
    )
    args_schema: ClassVar[Type[GetDefinitionArgs]] = GetDefinitionArgs

    def forward(
        self,
        file_path: str,
        line: int,
        column: int,
        include_context: bool = True,
        zero_based: bool = True,
    ) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(
            file_path=file_path,
            line=line,
            column=column,
            include_context=include_context,
            zero_based=zero_based,
        )

    async def _run(
        self,
        file_path: str,
        line: int,
        column: int,
        include_context: bool = True,
        zero_based: bool = True,
    ) -> List[GetDefinitionResult]:
        """Find definition of symbol at given position.

        Args:
            file_path: Path to file relative to repository root
            line: Line number (0-based by default)
            column: Column number (0-based by default)
            include_context: Include code context (target line + 5 lines after)
            zero_based: Use 0-based indexing (default: True)

        Returns:
            List of definition results (usually 1, but can be multiple for overloads)
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

            # Get definition locations
            locations = await client.get_definition(
                str(full_path), lsp_line, lsp_column
            )

            if not locations:
                return []

            # Process locations
            results = []
            for location in locations:
                # Convert URI to relative path
                relative_path = self._uri_to_relative_path(location["uri"])

                # Extract position
                def_line = location["range"]["start"]["line"]
                def_column = location["range"]["start"]["character"]

                # Convert to output format
                output_line = self._convert_to_output_index(def_line, zero_based)
                output_column = self._convert_to_output_index(def_column, zero_based)

                # Extract context if requested
                context = ""
                if include_context:
                    context = self._extract_context(
                        location["uri"], def_line, location["range"]
                    )

                result = GetDefinitionResult(
                    symbol_name=symbol_name,
                    definition_file=relative_path,
                    definition_line=output_line,
                    definition_column=output_column,
                    context=context,
                )
                results.append(result)

            return results

        except Exception as e:
            logger.warning(f"LSP get_definition operation failed: {e}")
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
        """Extract code context (target line + 5 lines after).

        Args:
            uri: File URI
            line: Line number (0-based)
            range_dict: LSP Range dictionary

        Returns:
            Context string with target line + 5 lines after, or empty string on error
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

            # Extract context: current line + 5 lines after (6 lines total)
            start_line = line
            end_line = min(len(lines), line + 6)

            context_lines = []
            for i in range(start_line, end_line):
                # Truncate very long lines
                line_text = lines[i].rstrip()
                if len(line_text) > 150:
                    line_text = line_text[:147] + "..."
                context_lines.append(line_text)

            return "\n".join(context_lines)

        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")
            return ""
