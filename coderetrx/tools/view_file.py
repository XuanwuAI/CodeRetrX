from typing import ClassVar, Optional, Type

import aiofiles
from pydantic import BaseModel, Field

from coderetrx.tools.base import BaseTool
from coderetrx.utils.path import safe_join


class ViewFileArgs(BaseModel):
    """Parameters for file viewing operation"""

    file_path: str = Field(
        ...,
        description="Absolute path of file to view",
    )
    show_line_numbers: bool = Field(
        True,
        description="If true, prefix each returned line with its 0-based line number.",
    )
    start_line: Optional[int] = Field(
        None,
        description="0-based inclusive starting line number. Optional, defaults to 0.",
        ge=0,
    )
    end_line: Optional[int] = Field(
        None,
        description="0-based exclusive ending line number. Optional, defaults to the file line count. May equal the file line count.",
        ge=0,
    )



class ViewFileTool(BaseTool):
    name = "view_file"
    description = (
        "View the contents of a file. Line numbers are 0-based. When a range is provided, this tool uses the half-open interval [start_line, end_line): start_line is inclusive, end_line is exclusive, and end_line may equal the file line count. The requested range length must be less than or equal to 1000 lines.\n\n"
        "When using this tool to gather information, it's your responsibility to ensure you have the COMPLETE context. Specifically, each time you call this command you should:\n"
        "1) Assess if the file contents you viewed are sufficient to proceed with your task.\n"
        "2) Take note of where there are lines not shown. These are represented by <... XX more lines from [code item] not shown ...> in the tool response.\n"
        "3) If the file contents you have viewed are insufficient, and you suspect they may be in lines not shown, proactively call the tool again to view those lines.\n"
        "4) When in doubt, call this tool again to gather more information. Remember that partial file views may miss critical dependencies, imports, or functionality."
    )
    args_schema: ClassVar[Type[ViewFileArgs]] = ViewFileArgs # type: ignore

    def forward(
        self,
        file_path: str,
        show_line_numbers: bool,
        start_line: int,
        end_line: int,
    ) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            show_line_numbers=show_line_numbers,
        )

    async def _run(
        self,
        file_path: str,
        show_line_numbers: bool,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """View file content with optional line range."""
        full_path = safe_join(self.repo_path, file_path.lstrip("/"))

        if not full_path.exists():
            return "File Not Exists.\n"

        if full_path.is_dir():
            return "Path is a directory, not a file.\n"

        try:
            async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
                content = await f.read()
        except UnicodeDecodeError:
            return "File is not a text file or uses unsupported encoding.\n"

        # splitlines() keeps the visible file lines without inventing a trailing
        # empty EOF line when the file ends with '\n'.
        lines = content.splitlines()
        total_lines = len(lines)

        # Handle line range
        if start_line is not None or end_line is not None:
            start = start_line if start_line is not None else 0
            end = end_line if end_line is not None else total_lines

            # Validate line numbers
            if start < 0 or end > total_lines or start > end:
                return (
                    "Invalid line range. Expected "
                    f"0 <= start_line <= end_line <= {total_lines}, "
                    "where end_line is exclusive.\n"
                )

            threshold_line = 1000
            if end - start > threshold_line:
                return (
                    f"File is too large ({total_lines} lines). Please specify a line "
                    f"range with at most {threshold_line} lines, or search by keyword.\n"
                )

            selected_lines = lines[start:end]
            actual_start = start
        else:
            selected_lines = lines
            actual_start = 0

        # Format output with or without line numbers
        if show_line_numbers:
            max_line_num = actual_start + len(selected_lines) - 1
            width = len(str(max_line_num))
            formatted_lines = [
                f"{actual_start + i:>{width}} | {line}"
                for i, line in enumerate(selected_lines)
            ]
            result = "\n".join(formatted_lines)
        else:
            result = "\n".join(selected_lines)

        # Add line count info unless we're showing the whole file
        if start_line is not None or end_line is not None:
            result += (
                f"\n\n(This file has total {total_lines} lines. "
                "Range contract: 0-based, [start_line, end_line), "
                f"valid range 0 <= start_line <= end_line <= {total_lines}.)"
            )

        return result
