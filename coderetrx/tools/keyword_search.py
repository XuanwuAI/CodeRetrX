import asyncio
import os
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from coderetrx.static.ripgrep import ripgrep_search  # type: ignore
from coderetrx.tools.base import BaseTool
from coderetrx.utils.llm import count_tokens_openai
from coderetrx.utils.path import safe_join


class KeywordSearchResult(BaseModel):
    path: str = Field(description="The path of the file containing the match.")
    start_line: int = Field(description="The start line of the match.")
    end_line: int = Field(description="The end line of the match.")
    content: str = Field(description="The content of the match.")

    @classmethod
    def repr(cls, entries: list["KeywordSearchResult"]) -> str:
        """Covnert a list of KeywordSearchResult to a readable string, grouped by file"""
        if not entries:
            return "No result Found."
        if not entries[0].path:
            return "The SearchDirectory does not exist. Please make sure that the SearchDirectory you are investigating exists."
        # group results by file path
        file_groups = {}
        for entry in entries:
            if entry.path not in file_groups:
                file_groups[entry.path] = []
            file_groups[entry.path].append(entry)

        tool_result = ""
        file_count = 0
        threshold_tokens = 10000  # reference: 6262 tokens = 830 lines python
        cur_tokens = 0

        # iterate over each file and its matches
        for file_path, matches in file_groups.items():
            file_count += 1
            each_result = ""
            each_result += f"# **{file_count}. {file_path}**\n"
            # iterate over each match in the file
            for match_index, match in enumerate(matches, 1):
                if match.start_line > 0:
                    each_result += f"## **{match_index}. Lines {match.start_line}-{match.end_line}**\n"
                    if match.content:
                        each_result += f"{match.content}\n"

            cur_tokens += count_tokens_openai(each_result)
            if cur_tokens > threshold_tokens:
                if entries[0].content != "":
                    tool_result += f"Omitted...\nReturn contents are too long. Please refine your query.\n"
                else:
                    tool_result += f"Omitted...\nReturn contents are too long. Please modify the parameters, for example, query may be too general, or the dir_path is too broad.\n"
                break
            tool_result += each_result

        return tool_result


class KeywordSearchTool(BaseTool):
    name = "keyword_search"
    description = (
        "High-performance code search using regular expressions (powered by ripgrep)\n"
        "Features:\n"
        "1. Supports both regex and plain text modes\n"
        "2. File type filtering with regular expressions\n"
        "3. Automatic truncation after 50 matches\n\n"
        "Usage tips:\n"
        "- Simple search: Set query_with_regexp=False for plain text matching\n"
        "- Precise search: Use regex for complex pattern matching\n"
        "- File filtering: Specify included file types with glob_pattern_includes (e.g. .*\\.java)"
    )
    inputs = {
        "dir_path": {
            "description": "The directory from which to run the ripgrep command. This path must be a directory, not a file.",
            "type": "string",
        },
        "query": {
            "description": "The search term or pattern to look for within files.",
            "type": "string",
        },
        "query_with_regexp": {
            "description": "If true, query accept a REGULAR expressions as a input.",
            "type": "boolean",
        },
        "glob_pattern_includes": {
            "description": "Defines glob patterns for file name filtering to include files in the search (e.g., *\\.java). Leave empty if no filtering is required.",
            "type": "string",
        },
        "glob_pattern_excludes": {
            "description": "Defines glob patterns for file name filtering to exclude files in the search (e.g., *\\.java). Leave empty if no filtering is required.",
            "type": "string",
        },
        "case_insensitive": {
            "description": "If true, performs a case-insensitive search.",
            "type": "boolean",
        },
        "include_content": {
            "description": "Whether to return the content of the code snippets. If set to `False`, only the location information of the code snippets will be returned.",
            "type": "boolean",
        },
    }
    output_type = "string"

    def forward(
        self,
        dir_path: str,
        query: str,
        query_with_regexp: bool,
        glob_pattern_includes: str,
        glob_pattern_excludes: str,
        case_insensitive: bool,
        include_content: bool,
    ) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(
            dir_path=dir_path,
            query=query,
            query_with_regexp=query_with_regexp,
            glob_pattern_includes=glob_pattern_includes,
            glob_pattern_excludes=glob_pattern_excludes,
            case_insensitive=case_insensitive,
            include_content=include_content,
        )

    async def _run(
        self,
        query: str,
        dir_path: str = "/",
        query_with_regexp: bool = False,
        glob_pattern_includes: str = "",
        glob_pattern_excludes: str = "",
        case_insensitive: bool = True,
        include_content: bool = True,
    ) -> list[KeywordSearchResult]:
        # Convert the query to a list of regexes
        regexes = [query]
        full_dir_path = safe_join(self.repo_path, dir_path.lstrip("/"))

        if not full_dir_path.exists():
            return [
                    KeywordSearchResult(
                        **{
                            "path": "",
                            "start_line": 0,
                            "end_line": 0,
                            "content": "Directory Not Exists",
                        }
                    )
                ]

        # Call ripgrep_search with the appropriate parameters
        rg_results = await ripgrep_search(
            search_dir=full_dir_path,
            regexes=regexes,
            extra_argvs=(
                ["--fixed-strings", "-g", "!.git"]
                if not query_with_regexp
                else ["-g", "!.git"]
            ),
            case_sensitive=not case_insensitive,
            include_file_pattern=(
                glob_pattern_includes if glob_pattern_includes else None
            ),
            exclude_file_pattern=(
                glob_pattern_excludes if glob_pattern_excludes else None
            ),
        )
        # Convert GrepMatchResult to KeywordSearchResult
        results = []
        for result in rg_results:
            search_result = KeywordSearchResult(
                path=str(result.file_path),
                start_line=result.line_number,
                end_line=result.line_number,
                content=result.line_text if include_content else "",
            )
            results.append(search_result)

        return results
