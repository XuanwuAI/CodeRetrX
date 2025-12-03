"""
Strategy for filtering filenames using LLM.
"""

from typing import List
from .base import FilterByLLMStrategy, StrategyExecuteResult
from ..smart_codebase import SmartCodebase as Codebase, LLMMapFilterTargetType
from coderetrx.static import File


class FilterFilenameByLLMStrategy(FilterByLLMStrategy[File]):
    """Strategy to filter filenames using LLM."""

    
    def get_strategy_name(self) -> str:
        return "FILTER_FILENAME_BY_LLM"

    
    def get_target_type(self) -> LLMMapFilterTargetType:
        return "file_name"

    
    def extract_file_paths(self, elements: List[File], codebase: Codebase) -> List[str]:
        return [str(file.path) for file in elements]

    
    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
        target_type: str = "symbol_content",
    ) -> StrategyExecuteResult:
        prompt = f"""
        A file with this path is highly likely to contain content that matches the following criteria:
        <content_criterias>
        {prompt}
        </content_criterias>
        <note>
        The objective of this requirement is to preliminarily identify files based on their paths that are likely to meet specific content criteria.
        Files with matching paths will proceed to a deeper analysis in the content filter (content_criterias) at a later stage (not in this run).  
        </note>
        """
        return await super().execute(codebase, prompt, subdirs_or_files, target_type)
