from typing import List, Literal, ClassVar, Type, Dict, Optional
from pydantic import BaseModel, Field

from coderetrx.static.codebase import Symbol, File, Keyword, ChunkType
from coderetrx.retrieval import (
    coderetrx_mapping,
    LLMMapFilterTargetType,
    llm_traversal_mapping,
)
from coderetrx.retrieval.code_recall import CodeRecallSettings, CoarseRecallStrategyType
from coderetrx.tools.base import BaseTool


class LLMCodeMappingArgs(BaseModel):
    mapping_prompt: str = Field(
        description="Natural language instruction for transforming/extracting information from code. Examples: 'Extract function signatures and return types', 'Summarize what each class does in one sentence', 'List all API endpoints and their HTTP methods', 'Extract security-related configurations'"
    )
    subdirs_or_files: List[str] = Field(
        default=[],
        description="List of subdirectories or files to scope the mapping (relative to repo root). Examples: ['src/api'], ['backend/models.py']. Empty list maps entire repository",
    )
    return_content: bool = Field(
        default=False,
        description="Whether to include the original source code content alongside mapped results. True returns both original and transformed content, False returns only the mapped/extracted information (cleaner output)",
    )


class LLMCodeMappingResult(BaseModel):
    type: str = Field(description="Type of code element: 'primary', 'query_result', or 'other'")
    start_line: int = Field(description="Start line of the code element")
    end_line: int = Field(description="End line of the code element")
    start_column: int = Field(description="Start column of the code element")
    end_column: int = Field(description="End column of the code element")
    src: str = Field(description="Source file path")
    content: str = Field(description="Original content of the code element (if return_content=True)")
    mapped_content: str = Field(description="Mapped/transformed content based on the prompt")

    @classmethod
    def repr(cls, entries: List["LLMCodeMappingResult"]) -> str:
        """Convert a list of LLMCodeMappingResult to a readable string"""
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
                output += f"**Mapped Content:**\n{match.mapped_content}\n"
                if match.content:
                    output += f"\n**Original Content:**\n```\n{match.content}\n```\n"
                output += "\n"

        return output


class LLMCodeMappingTool(BaseTool):
    name = "llm_code_mapping"
    description = (
        "Filters code to find relevant elements, then extracts specific information from each match.\n\n"
        "This is a two-stage process:\n"
        "1. Filter: Find code elements matching your criteria (similar to llm_code_filter)\n"
        "2. Map: Extract specific information from each filtered result\n\n"
        "Use this tool to:\n"
        "- Find all functions of a certain type and extract their parameters\n"
        "- Locate API endpoints and extract their HTTP methods and routes\n"
        "- Find configuration classes and extract their settings\n"
        "- Identify test functions and extract what they test\n\n"
        "Examples:\n"
        "- 'Find all system functions and extract their parameters'\n"
        "- 'Find API endpoints and list their HTTP methods and paths'\n"
        "- 'Find database models and extract their field definitions'\n"
        "- 'Find error handlers and describe what errors they handle'\n"
        "- 'Find authentication functions and summarize their approach'\n\n"
        "Usage tips:\n"
        "- Describe both WHAT to find and WHAT to extract from it\n"
        "- Narrow scope with subdirs_or_files to improve performance\n"
        "- Set return_content=False for cleaner output (only extracted info, not original code)"
    )
    args_schema: ClassVar[Type[LLMCodeMappingArgs]] = LLMCodeMappingArgs

    # Class-level cache for initialized codebases
    _codebase_cache: ClassVar[Dict[str, "Codebase"]] = {}

    def __init__(
        self,
        repo_url: str,
        uuid: Optional[str] = None,
        coarse_recall_strategy: CoarseRecallStrategyType = "auto",
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
        Initialize LLM Code Mapping Tool.

        Args:
            repo_url: Repository URL or path
            uuid: Optional unique identifier
            coarse_recall_strategy: Strategy for initial code recall (default: "auto")
            target_type: What level of code element to analyze (default: "symbol_content")
        """
        super().__init__(repo_url, uuid)
        self.coarse_recall_strategy = coarse_recall_strategy
        self.target_type = target_type

    def _get_or_create_codebase(self):
        """Get cached codebase or create a new one."""
        cache_key = f"{self.repo_id}:{self.repo_path}"

        if cache_key not in LLMCodeMappingTool._codebase_cache:
            from coderetrx.retrieval import CodebaseFactory
            LLMCodeMappingTool._codebase_cache[cache_key] = CodebaseFactory.new(
                id=self.repo_id,
                dir=self.repo_path,
            )

        return LLMCodeMappingTool._codebase_cache[cache_key]

    async def _run(
        self,
        mapping_prompt: str,
        subdirs_or_files: List[str] = [],
        return_content: bool = False,
    ) -> List[LLMCodeMappingResult]:
        """
        Map code elements based on the specified prompt.

        Args:
            mapping_prompt: The mapping prompt
            subdirs_or_files: List of subdirectories or files to map
            return_content: Whether to include original source code content in results

        Returns:
            List of LLMCodeMappingResult objects containing mapped code elements
        """
        # Get or create cached codebase
        codebase = self._get_or_create_codebase()

        # Use the appropriate method based on strategy
        if self.coarse_recall_strategy == "precise":
            elements, llm_results = await llm_traversal_mapping(
                codebase,
                prompt=mapping_prompt,
                subdirs_or_files=subdirs_or_files,
                target_type=self.target_type,  # type:ignore
            )
        else:
            elements, llm_results = await coderetrx_mapping(
                codebase,
                prompt=mapping_prompt,
                subdirs_or_files=subdirs_or_files,
                target_type=self.target_type,
                coarse_recall_strategy=self.coarse_recall_strategy,
                settings=CodeRecallSettings(),
            )

        # Convert elements to LLMCodeMappingResult
        results = []
        for idx, element in enumerate(elements):
            mapped_content = llm_results[idx].result

            if isinstance(element, Symbol):
                # For Symbol objects, use the chunk
                chunk = element.chunk
                sem_type = "other"
                if chunk.type == ChunkType.PRIMARY:
                    sem_type = "primary"
                elif chunk.type == ChunkType.QUERY_RESULT:
                    sem_type = "query_result"
                results.append(
                    LLMCodeMappingResult(
                        type=sem_type,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        start_column=chunk.start_column,
                        end_column=chunk.end_column,
                        src=str(chunk.src.path),
                        content=chunk.code() if return_content else "",
                        mapped_content=str(mapped_content),
                    )
                )
            elif isinstance(element, File):
                # For File objects, create a code chunk representation
                lines = element.lines
                results.append(
                    LLMCodeMappingResult(
                        type="primary",
                        start_line=0,
                        end_line=len(lines) - 1,
                        start_column=0,
                        end_column=len(lines[-1]) if lines else 0,
                        src=str(element.path),
                        content=element.content if return_content else "",
                        mapped_content=str(mapped_content),
                    )
                )
            elif isinstance(element, Keyword):
                # For Keyword objects, create a simple representation
                results.append(
                    LLMCodeMappingResult(
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
                        mapped_content=str(mapped_content),
                    )
                )

        return results
