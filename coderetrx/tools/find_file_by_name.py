import asyncio
import os
from typing import ClassVar, List, Type
from pathlib import Path
from pydantic import BaseModel, Field
from coderetrx.static.ripgrep import ripgrep_glob  # type: ignore
from coderetrx.utils.path import safe_join
from coderetrx.tools.base import BaseTool
import filetype


class FindFileByNameResult(BaseModel):
    path: str = Field(description="The path of matched file name")
    type: str = Field(description="The type of the file")

    @classmethod
    def repr(cls, entries: list["FindFileByNameResult"]):
        """
        convert a list of FindFileByNameResult to a readable string
        """
        if not entries:
            return "No result Found."
        if not entries[0].path:
            return "The SearchDirectory does not exist. Please make sure that the SearchDirectory you are investigating exists."
        tool_result = ""
        for i, file in enumerate(entries, 1):
            tool_result += f"# **{i}. {file.path}**\n"

        return tool_result


class FindFileByNameTool(BaseTool):
    name = "find_file_by_name"
    description = (
        "This tool searches for files and directories within a specified directory, similar to the Linux `find` command. "
        "The returned result paths are relative to the root path."
    )
    inputs = {
        "dir_path": {
            "description": "The directory to search within",
            "type": "string",
        },
        "pattern": {
            "description": "pattern to search for. Based on keyword matching, wildcards are not supported and are not required.",
            "type": "string",
        },
    }
    output_type = "string"

    def _get_file_type(self, path: str) -> str:
        type = filetype.guess(path)
        if type:
            return type.mime
        else:
            return "Unknown"

    def forward(self, dir_path: str, pattern: str) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(dir_path=dir_path, pattern=pattern)

    async def _run(self, dir_path: str, pattern: str) -> list[FindFileByNameResult]:
        """Search for files matching the pattern in the specified directory"""
        full_dir_path = safe_join(self.repo_path, dir_path.lstrip("/"))

        if not os.path.exists(full_dir_path):
            return [FindFileByNameResult(path="", type="Directory Not Exists")]
            

        matched_files = await ripgrep_glob(
            full_dir_path, pattern, extra_argv=["-g", "!.git"]
        )

        results = []
        for file_path in matched_files:
            full_file_path = full_dir_path / file_path
            file_type = self._get_file_type(str(full_file_path))
            results.append(FindFileByNameResult(path=file_path, type=file_type))

        return results 


if __name__ == "__main__":
    # Example usage
    tool = FindFileByNameTool("https://github.com/apache/flink.git")
    results = asyncio.run(tool._run(dir_path="/", pattern="*.md"))
    print(results)
