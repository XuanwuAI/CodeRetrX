"""
Strategy for filtering keywords using vector similarity search followed by LLM refinement.
"""

from typing import Any, List, Union, Optional, override

from coderetrx.retrieval.strategy.base import StrategyExecuteResult
from coderetrx.retrieval.strategy.base import FilterByVectorAndLLMStrategy
from coderetrx.retrieval.smart_codebase import SmartCodebase as Codebase, LLMMapFilterTargetType, SimilaritySearchTargetType
from coderetrx.static import Keyword, Symbol, File


class FilterKeywordByVectorAndLLMStrategy(FilterByVectorAndLLMStrategy):
    """Strategy to filter keywords using vector similarity search followed by LLM refinement."""

    name: str = "FILTER_KEYWORD_BY_VECTOR_AND_LLM"

    @override
    def get_strategy_name(self) -> str:
        return self.name

    @override
    def get_target_types_for_vector(self) -> List[SimilaritySearchTargetType]:
        return ["keyword"]

    @override
    def get_target_type_for_llm(self) -> LLMMapFilterTargetType:
        return "keyword"

    @override
    def get_collection_size(self, codebase: Codebase) -> int:
        return len(codebase.keywords)

    @override
    def filter_elements(
        self,
        elements: List[Any],
        subdirs_or_files: List[str] = [],
        codebase: Optional[Codebase] = None,
    ) -> List[Union[Keyword, Symbol, File]]:
        keyword_elements: List[Union[Keyword, Symbol, File]] = []
        for element in elements:
            if isinstance(element, Keyword):
                # If subdirs_or_files is provided and codebase is available, filter by subdirs
                if subdirs_or_files and codebase:
                    for ref_file in element.referenced_by:
                        if any(
                            str(ref_file.path).startswith(subdir)
                            for subdir in subdirs_or_files
                        ):
                            keyword_elements.append(element)
                            break
                else:
                    keyword_elements.append(element)
        return keyword_elements

    @override
    def collect_file_paths(
        self,
        filtered_elements: List[Any],
        codebase: Codebase,
        subdirs_or_files: List[str],
    ) -> List[str]:
        referenced_paths = set()
        for keyword in filtered_elements:
            if isinstance(keyword, Keyword) and keyword.referenced_by:
                for ref_file in keyword.referenced_by:
                    if str(ref_file.path).startswith(tuple(subdirs_or_files)):
                        referenced_paths.add(str(ref_file.path))
        return list(referenced_paths)

    @override
    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
    ) -> StrategyExecuteResult:
        prompt = f"""
        A code chunk containing the specified keywords is highly likely to meet the following criteria:
        <content_criteria>
        {prompt}
        </content_criteria>
        <note>
        The objective of this requirement is to preliminarily filter files that are likely to meet specific content criteria based on the keywords they contain. 
        Files with matching keywords will proceed to deeper analysis in the content filter (content_criteria) at a later stage (not in this run). 
        </note>
        """
        return await super().execute(codebase, prompt, subdirs_or_files)
