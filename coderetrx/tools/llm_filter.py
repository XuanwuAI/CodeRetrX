from typing import List, Literal, ClassVar, Type, Dict, Optional
from pydantic import BaseModel, Field

from coderetrx.static import Symbol, File, Keyword
from coderetrx.static.codebase import ChunkType
from coderetrx.retrieval import (
    coderetrx_filter,
    LLMMapFilterTargetType,
    llm_traversal_filter,
)
from coderetrx.retrieval.code_recall import CodeRecallSettings, CoarseRecallStrategyType
from coderetrx.tools.base import BaseTool
from coderetrx.utils.path import safe_join


class LLMCodeFilterArgs(BaseModel):
    filter_prompt: str = Field(
        description="Natural language description of what code elements to find. Examples: 'functions that handle authentication', 'classes related to database operations', 'error handling code'"
    )
    subdirs_or_files: List[str] = Field(
        default=[],
        description="List of subdirectories or files to scope the search (relative to repo root). Examples: ['src/api'], ['backend/auth.py']. Empty list searches entire repository",
    )
    return_content: bool = Field(
        default=False,
        description="Whether to include the actual source code content in results. True returns code snippets, False returns only metadata and reasons (faster, less verbose)",
    )


class LLMCodeFilterResult(BaseModel):
    type: str = Field(description="Type of code element: 'primary', 'query_result', or 'other'")
    start_line: int = Field(description="Start line of the code element")
    end_line: int = Field(description="End line of the code element")
    start_column: int = Field(description="Start column of the code element")
    end_column: int = Field(description="End column of the code element")
    src: str = Field(description="Source file path")
    content: str = Field(description="Content of the code element (if return_content=True)")
    reason: str = Field(description="Reason why this element was filtered/selected")

    @classmethod
    def repr(cls, entries: List["LLMCodeFilterResult"]) -> str:
        """Convert a list of LLMCodeFilterResult to a readable string"""
        if not entries:
            return "No results found."

        # Group results by file path
        file_groups = {}
        for entry in entries:
            if entry.src not in file_groups:
                file_groups[entry.src] = []
            file_groups[entry.src].append(entry)

        output = ""
        file_count = 0

        # Iterate over each file and its matches
        for file_path, matches in file_groups.items():
            file_count += 1
            output += f"# **{file_count}. {file_path}**\n"

            # Iterate over each match in the file
            for match_index, match in enumerate(matches, 1):
                output += f"## **{match_index}. Lines {match.start_line}-{match.end_line}** ({match.type})\n"
                output += f"**Reason:** {match.reason}\n"
                if match.content:
                    output += f"```\n{match.content}\n```\n"
                output += "\n"

        return output


class LLMCodeFilterTool(BaseTool):
    name = "llm_code_filter"
    description = (
        "Semantic code filtering using LLM-powered understanding to find relevant code elements based on natural language queries.\n\n"
        "Use this tool to:\n"
        "- Find functions, classes, or code that match specific criteria\n"
        "- Search for code implementing particular features or patterns\n"
        "- Locate error handling, authentication, or other semantic code patterns\n\n"
        "Examples:\n"
        "- 'functions that handle authentication'\n"
        "- 'classes related to database operations'\n"
        "- 'error handling code for network requests'\n"
        "- 'code implementing cryptographic algorithms'\n\n"
        "Usage tips:\n"
        "- Use specific, descriptive prompts for best results\n"
        "- Narrow scope with subdirs_or_files to improve performance and relevance\n"
        "- Set return_content=True to see the actual code, or False for just summaries"
    )
    args_schema: ClassVar[Type[LLMCodeFilterArgs]] = LLMCodeFilterArgs

    # Class-level cache for initialized codebases
    _codebase_cache: ClassVar[Dict[str, "Codebase"]] = {}

    def __init__(
        self,
        repo_url: str,
        uuid: Optional[str] = None,
        coarse_recall_strategy: CoarseRecallStrategyType = "auto",
        enable_secondary_recall: bool = False,
        target_type: Literal[
            "symbol_content",
            "file_content",
            "function_content",
            "class_content",
            "leaf_symbol_content",
            "root_symbol_content",
        ] = "symbol_content",
    ):
        """
        Initialize LLM Code Filter Tool.

        Args:
            repo_url: Repository URL or path
            uuid: Optional unique identifier
            coarse_recall_strategy: Strategy for initial code recall (default: "auto")
            enable_secondary_recall: Whether to use two-stage filtering (default: False)
            target_type: What level of code element to analyze (default: "symbol_content")
        """
        super().__init__(repo_url, uuid)
        self.coarse_recall_strategy = coarse_recall_strategy
        self.enable_secondary_recall = enable_secondary_recall
        self.target_type = target_type

    def _get_or_create_codebase(self):
        """Get cached codebase or create a new one."""
        cache_key = f"{self.repo_id}:{self.repo_path}"

        if cache_key not in LLMCodeFilterTool._codebase_cache:
            from coderetrx.retrieval import CodebaseFactory
            LLMCodeFilterTool._codebase_cache[cache_key] = CodebaseFactory.new(
                id=self.repo_id,
                dir=self.repo_path,
            )

        return LLMCodeFilterTool._codebase_cache[cache_key]

    async def _run(
        self,
        filter_prompt: str,
        subdirs_or_files: List[str] = [],
        return_content: bool = False,
    ) -> List[LLMCodeFilterResult]:
        """
        Filter code elements based on the specified prompt.

        Args:
            filter_prompt: The filtering prompt
            subdirs_or_files: List of subdirectories or files to filter by
            return_content: Whether to include source code content in results

        Returns:
            List of LLMCodeFilterResult objects containing filtered code elements
        """
        # Get or create cached codebase
        codebase = self._get_or_create_codebase()

        # Use the appropriate method based on strategy
        if self.coarse_recall_strategy == "precise":
            elements, llm_results = await llm_traversal_filter(
                codebase,
                prompt=filter_prompt,
                subdirs_or_files=subdirs_or_files,
                target_type=self.target_type,  # type:ignore
            )
        else:
            elements, llm_results = await coderetrx_filter(
                codebase,
                prompt=filter_prompt,
                subdirs_or_files=subdirs_or_files,
                target_type=self.target_type,
                coarse_recall_strategy=self.coarse_recall_strategy,
                settings=CodeRecallSettings(),
                enable_secondary_recall=self.enable_secondary_recall,
            )

        # Convert elements to LLMCodeFilterResult
        results = []
        for idx, element in enumerate(elements):
            if isinstance(element, Symbol):
                # For Symbol objects, use the chunk
                chunk = element.chunk
                sem_type = "other"
                if chunk.type == ChunkType.PRIMARY:
                    sem_type = "primary"
                elif chunk.type == ChunkType.QUERY_RESULT:
                    sem_type = "query_result"
                results.append(
                    LLMCodeFilterResult(
                        type=sem_type,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        start_column=chunk.start_column,
                        end_column=chunk.end_column,
                        src=str(chunk.src.path),
                        content=chunk.code() if return_content else "",
                        reason=llm_results[idx].reason,
                    )
                )
            elif isinstance(element, File):
                # For File objects, create a code chunk representation
                content = element.content
                lines = element.lines
                results.append(
                    LLMCodeFilterResult(
                        type="primary",
                        start_line=0,
                        end_line=len(lines) - 1,
                        start_column=0,
                        end_column=len(lines[-1]) if lines else 0,
                        src=str(element.path),
                        content=content if return_content else "",
                        reason=llm_results[idx].reason,
                    )
                )
            elif isinstance(element, Keyword):
                # For Keyword objects, create a simple representation
                results.append(
                    LLMCodeFilterResult(
                        type="other",
                        start_line=0,
                        end_line=0,
                        start_column=0,
                        end_column=len(element.content),
                        src=(
                            str(element.referenced_by[0].path)
                            if element.referenced_by
                            else "keyword"
                        ),
                        content=element.content if return_content else "",
                        reason=llm_results[idx].reason,
                    )
                )

        return results
