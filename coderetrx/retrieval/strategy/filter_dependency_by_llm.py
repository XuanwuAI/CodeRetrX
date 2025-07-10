"""
Strategy for filtering dependencies using LLM.
"""

from typing import List, override
from .base import FilterByLLMStrategy, StrategyExecuteResult
from ..smart_codebase import SmartCodebase as Codebase, LLMMapFilterTargetType
from coderetrx.static import Dependency, Symbol

class FilterDependencyByLLMStrategy(FilterByLLMStrategy[Dependency]):
    """Strategy to filter dependencies by LLM and retrieve code chunks that use these dependencies."""

    @override
    def get_strategy_name(self) -> str:
        return "FILTER_DEPENDENCY_BY_LLM"

    @override
    def get_target_type(self) -> LLMMapFilterTargetType:
        return "dependency_name"

    @override
    def extract_file_paths(
        self, elements: List[Dependency], codebase: Codebase
    ) -> List[str]:
        """Extract file paths from dependencies by getting files that import these dependencies."""
        file_paths = []
        for dependency in elements:
            if isinstance(dependency, Dependency):
                file_paths.extend([str(f.path) for f in dependency.imported_by])
            elif isinstance(dependency, Symbol) and dependency.type == "dependency":
                file_paths.append(str(dependency.file.path))
        return list(set(file_paths))

    @override
    async def execute(
        self, codebase: Codebase, prompt: str, subdirs_or_files: List[str]
    ) -> StrategyExecuteResult:
        enhanced_prompt = f"""
        A dependency that matches the following criteria is highly likely to be relevant:
        <dependency_criteria>
        {prompt}
        </dependency_criteria>
        <note>
        The objective is to identify dependencies based on their names that match the specified criteria.
        Files that import these matching dependencies will be retrieved for further analysis.
        Focus on dependency names, package names, module names, and library names that are relevant to the criteria.
        </note>
        """
        return await super().execute(codebase, enhanced_prompt, subdirs_or_files)

