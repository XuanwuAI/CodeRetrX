import asyncio
import os
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from coderetrx.static.ripgrep import ripgrep_search  # type: ignore
from coderetrx.tools.base import BaseTool
from coderetrx.utils.llm import count_tokens_openai
from coderetrx.tools.keyword_search import KeywordSearchResult


class GetReferenceResult(KeywordSearchResult):
    symbol_idx: int = Field(
        default=-1,
        description="The index of the symbol in the list of symbols. if -1, it means not applicable.",
    )
    symbol_name: str = Field(
        default="", description="The symbolic name whose reference is to be retrieved."
    )
    symbol_location: str = Field(default="", description="The location of the symbol.")

    @classmethod
    def repr(cls, entries: list["GetReferenceResult"]):
        """
        convert a list of GetReferenceResult to a readable string
        """
        if not entries:
            return "No entries found."

        result = ""
        current_symbol_idx = -1
        entries_count = {}
        entries = sorted(
            entries, key=lambda x: (x.symbol_idx, x.start_line, x.end_line)
        )
        # symbol_idx is -1 means the multi-symbol cross-reference is not applicable
        if entries[0].symbol_idx == -1:
            for idx, ref in enumerate(entries):
                result += f"## **{idx}. {ref.path}:L{ref.start_line}:{ref.end_line}**\n"
                result += f"{ref.content}\n"
            return result

        threshold_tokens = 5000  # reference: 6262 tokens = 830 lines python
        cur_tokens = 0
        for idx, ref in enumerate(entries):
            each_result = ""
            # When processing a new symbol, add a symbol title.
            if ref.symbol_idx != current_symbol_idx:
                current_symbol_idx = ref.symbol_idx
                entries_count[current_symbol_idx] = 0
                each_result += f"# **{current_symbol_idx}. {ref.symbol_name}** location:{ref.symbol_location}\n"

                if not ref.path:
                    each_result += "No entries Found.\n"
                    continue

            entries_count[current_symbol_idx] += 1

            if entries_count[current_symbol_idx] == 1:
                # calulate the total reference count of the current symbol
                total_refs = sum(
                    1 for r in entries if r.symbol_idx == current_symbol_idx and r.path
                )
                each_result += f"{total_refs} entries found:\n"

            each_result += f"## **{entries_count[current_symbol_idx]}. {ref.path}:L{ref.start_line}:{ref.end_line}**\n"
            each_result += f"{ref.content}\n"

            cur_tokens += count_tokens_openai(each_result)
            if cur_tokens > threshold_tokens:
                result += f"Omitted...\nReturn contents are too long. Please refine your query.\n"
                break
            result += each_result

        return result


class GetReferenceTool(BaseTool):
    name = "get_reference"
    description = (
        "Used to find symbol direct references in the codebase, should be used when tracking code usage. "
        "Finding multiple levels of references requires multiple calls."
    )
    inputs = {
        "symbol_name": {
            "description": "The symbolic name whose reference is to be retrieved.",
            "type": "string",
        },
    }
    output_type = "string"

    def forward(self, symbol_name: str) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(symbol_name=symbol_name)

    async def _run(self, symbol_name: str) -> list[GetReferenceResult]:
        """Find references to a symbol in the codebase"""
        # Convert the query to a list of regexes
        regexes = [f"\\b{symbol_name}\\b"]

        # Call ripgrep_search with the appropriate parameters
        rg_results = await ripgrep_search(
            search_dir=Path(self.repo_path),
            regexes=regexes,
            case_sensitive=True,
            exclude_file_pattern=".git",
        )

        # Convert GrepMatchResult to KeywordSearchResult
        results = []
        for result in rg_results:
            search_result = GetReferenceResult(
                path=str(result.file_path),
                start_line=result.line_number,
                end_line=result.line_number,
                content=result.line_text,
            )
            results.append(search_result)

        return results 


if __name__ == "__main__":
    # Example usage
    tool = GetReferenceTool("https://github.com/apache/flink.git")
    result = asyncio.run(tool._run(symbol_name="upload"))
    print(result)
