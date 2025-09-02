import asyncio
import os
from typing import Optional, Type, ClassVar
from pathlib import Path
from pydantic import BaseModel, Field
from coderetrx.tools.base import BaseTool
import aiofiles
from coderetrx.utils.path import safe_join


class ViewFileArgs(BaseModel):
    """Parameters for file viewing operation"""

    file_path: str = Field(
        ...,
        description="Absolute path of file to view",
    )
    start_line: Optional[int] = Field(
        None, description="Starting line number. Optional, default to be 0.", ge=0
    )
    end_line: Optional[int] = Field(
        None,
        description="Ending line number. Optional, default to be the last line.",
        ge=0,
    )


class ViewFileTool(BaseTool):
    name = "view_file"
    description = (
        "View the contents of a file. The lines of the file are 0-indexed, and the output of this tool call will be the file contents from StartLine to EndLine. The line range should be less than or equal 1000.\n\n"
        "When using this tool to gather information, it's your responsibility to ensure you have the COMPLETE context. Specifically, each time you call this command you should:\n"
        "1) Assess if the file contents you viewed are sufficient to proceed with your task.\n"
        "2) Take note of where there are lines not shown. These are represented by <... XX more lines from [code item] not shown ...> in the tool response.\n"
        "3) If the file contents you have viewed are insufficient, and you suspect they may be in lines not shown, proactively call the tool again to view those lines.\n"
        "4) When in doubt, call this tool again to gather more information. Remember that partial file views may miss critical dependencies, imports, or functionality."
    )
    args_schema: ClassVar[Type[ViewFileArgs]] = ViewFileArgs

    def forward(self, file_path: str, start_line: int, end_line: int) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(
            file_path=file_path, start_line=start_line, end_line=end_line
        )

    async def _run(
        self,
        file_path: str,
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

        lines = content.split("\n")
        total_lines = len(lines)

        # Handle line range
        if start_line is not None or end_line is not None:
            start = start_line if start_line is not None else 0
            end = end_line if end_line is not None else total_lines

            # Validate line numbers
            if start < 0 or end > total_lines or start > end:
                return f"Invalid line range (0-{total_lines}).\n"

            threshold_line = 1000
            if end - start >= threshold_line:
                return f"File is too large ({total_lines} lines), please specify a line range less than or equal {threshold_line}, or search keyword in the file.\n"

            selected_lines = lines[start:end]
            result = "\n".join(selected_lines)
        else:
            result = content

        # Add line count info unless we're showing the whole file
        if start_line is not None or end_line is not None:
            result += f"\n\n(This file has total {total_lines} lines.)"

        return result
