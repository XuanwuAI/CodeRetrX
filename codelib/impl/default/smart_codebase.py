from typing import Optional, List, Tuple, Any, Union, Literal
from codelib.static import Codebase
from codelib.static.codebase import Symbol, Keyword, File
from codelib.retrieval import SmartCodebase as SmartCodebaseBase
from attrs import define, field
from codelib.utils.embedding import SimilaritySearcher
import os
import logging
from pathlib import Path
from .prompt import (
    llm_filter_prompt_template,
    llm_mapping_prompt_template,
    CodeMapFilterResult,
)

logger = logging.getLogger(__name__)

LLMMapFilterTargetType = Literal[
    "file_name",
    "file_content",
    "symbol_name",
    "symbol_content",
    "class_name",
    "class_content",
    "function_name",
    "function_content",
    "dependency_name",
    "dependency_reference",
    "dependency",
    "keyword",
]

SimilaritySearchTargetType = Literal["symbol_name", "symbol_content", "keyword"]


@define
class SmartCodebase(SmartCodebaseBase):
    symbol_name_searcher: Optional["SimilaritySearcher"] = field(default=None)
    symbol_content_searcher: Optional["SimilaritySearcher"] = field(default=None)
    keyword_searcher: Optional["SimilaritySearcher"] = field(default=None)

    async def llm_filter(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: List[str] = [],
        additional_code_elements: List[Any] = [],
        prompt_template: str = llm_filter_prompt_template,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Filter code elements based on the specified type and prompt using LLM.

        Args:
            prompt: The filtering prompt
            target_type: The type of code element to filter
            subdirs_or_files: List of subdirectories or files to filter by
            additional_code_elements: Additional code elements to include in filtering

        Returns:
            A tuple containing:
            - Filtered elements
            - LLM results with reasoning
        """
        from codelib.utils.jsonparser import TolerantJsonParser
        from codelib.utils.llm import call_llm_with_fallback


        # Get elements based on type
        elements = []
        if target_type in ["file_name", "file_content"]:
            elements = [file for file in self.source_files]
        elif target_type in ["symbol_name", "symbol_content"]:
            elements = [symbol for symbol in self.symbols]
        elif target_type in ["class_name", "class_content"]:
            elements = [symbol for symbol in self.symbols if symbol.type == "class"]
        elif target_type in ["function_name", "function_content"]:
            elements = [symbol for symbol in self.symbols if symbol.type == "function"]
        elif target_type in ["dependency_name", "dependency_reference", "dependency"]:
            elements = [
                symbol for symbol in self.symbols if symbol.type == "dependency"
            ]
        elif target_type == "keyword":
            elements = [keyword for keyword in self.keywords]

        # Filter by subdirectories/files if specified
        filtered_elements = []
        for path_prefix in subdirs_or_files:
            path_prefix = path_prefix.replace("\\", "/").lstrip("/")
            if path_prefix == "." or path_prefix == "./":
                path_prefix = ""
            path_prefix = str(Path(self.dir) / path_prefix)

            for element in elements:
                if isinstance(element, Symbol) and str(element.file.path).startswith(
                    path_prefix
                ):
                    filtered_elements.append(element)
                elif isinstance(element, File) and str(element.path).startswith(
                    path_prefix
                ):
                    filtered_elements.append(element)
                elif isinstance(element, Keyword):
                    for path in element.referenced_by:
                        if str(path).startswith(path_prefix):
                            filtered_elements.append(element)
                            break

        # Add additional elements if provided
        if additional_code_elements:
            filtered_elements.extend(additional_code_elements)

        elements = filtered_elements if filtered_elements else elements

        # Process elements in batches
        async def process_element_batch(
            elements_batch: List[Any], target_type: str, requirement: str
        ) -> Tuple[List[Any], List[Any]]:
            text = ""
            for i, element in enumerate(elements_batch):
                content = ""
                if isinstance(element, Symbol):
                    content = (
                        element.chunk.code()
                        if target_type.endswith("_content")
                        else element.name
                    )
                    element_type = f"{element.type} symbol"
                elif isinstance(element, File):
                    content = (
                        element.content
                        if target_type == "file_content"
                        else str(element.path)
                    )
                    element_type = "file"
                elif isinstance(element, Keyword):
                    content = element.content
                    element_type = "keyword"
                else:
                    content = str(element)
                    element_type = "unknown"

                text += f"\n<code_element type={element_type} index={i}>\n{content}\n</code_element>"

            invoke_input = {
                "code_elements": text,
                "code_element_number": len(elements_batch),
                "requirement": requirement,
            }

            try:
                model_ids = [os.environ["LLM_MAPFILTER_MODEL_ID"],"anthropic/claude-3.7-sonnet"]
                llm_results = await call_llm_with_fallback(
                    response_model=List[CodeMapFilterResult],
                    input_data=invoke_input,
                    prompt_template=prompt_template,
                    model_ids=model_ids,
                )

                right_llm_results = [result for result in llm_results if result.result]
                right_elements = [
                    elements_batch[result.index]
                    for result in right_llm_results
                    if result.index < len(elements_batch)
                ]

                return right_llm_results, right_elements
            except Exception as e:
                logger.error(f"LLM processing failed: {str(e)}")
                return [], []

        # Batch elements based on content length
        element_tasks = []
        cur_batch = []
        max_batch_length = int(os.environ.get("LLM_MAPFILTER_MAX_BATCH_LENGTH", 10000)) 
        max_batch_size =  int(os.environ.get("LLM_MAPFILTER_MAX_BATCH_SIZE", 100)) 

        for element in elements:
            content_length = 0
            if isinstance(element, Symbol):
                content_length = len(element.chunk.code())
            elif isinstance(element, File):
                content_length = len(element.content)
            else:
                content_length = len(str(element))

            if content_length > max_batch_length:
                element_tasks.append([element])
                continue

            cur_batch_length = sum(
                len(
                    getattr(
                        e,
                        "chunk",
                        getattr(e, "_content", getattr(e, "content", str(e))),
                    )
                )
                for e in cur_batch
            )

            if (
                content_length + cur_batch_length <= max_batch_length
                and len(cur_batch) < max_batch_size
            ):
                cur_batch.append(element)
            else:
                element_tasks.append(cur_batch)
                cur_batch = [element]

        if cur_batch:
            element_tasks.append(cur_batch)

        # Process all batches
        from tqdm.asyncio import tqdm

        gather_results = await tqdm.gather(
            *[
                process_element_batch(batch, target_type, prompt)
                for batch in element_tasks
            ]
        )

        flattened_llm_results = []
        flattened_right_elements = []

        for llm_result, right_elements in gather_results:
            flattened_llm_results.extend(llm_result)
            flattened_right_elements.extend(right_elements)

        return flattened_right_elements, flattened_llm_results

    async def llm_map(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: List[str] = [],
        additional_code_elements: List[Any] = [],
    ) -> Tuple[List[Any], List[Any]]:
        """
        Map code elements based on the specified type and prompt using LLM.

        Args:
            prompt: The mapping prompt
            target_type: The type of code element to map
            subdirs_or_files: List of subdirectories or files to filter by
            additional_code_elements: Additional code elements to include in mapping

        Returns:
            A tuple containing:
            - Mapped elements
            - LLM results with mapping content
        """
        # Reuse llm_filter implementation since the core logic is the same
        # Just use a different prompt template

        return await self.llm_filter(
            prompt=prompt,
            target_type=target_type,
            subdirs_or_files=subdirs_or_files,
            additional_code_elements=additional_code_elements,
            prompt_template=llm_mapping_prompt_template,
        )

    async def similarity_search(
        self,
        target_types: List[SimilaritySearchTargetType],
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 100,
    ) -> List[Any]:
        """
        Perform similarity search on symbols based on the specified types.

        Args:
            target_types: List of types to search on (symbol_name, symbol_content, keyword)
            query: The search query
            threshold: Similarity threshold, defaults to value from environment variable
            top_k: Maximum number of results to return

        Returns:
            List of symbols or keywords that match the search criteria

        Raises:
            ValueError: If searcher is not initialized for the requested type
        """
        threshold = (
            float(os.environ.get("SIMILARITY_SEARCH_THRESHOLD", 0.1))
            if threshold is None
            else threshold
        )
        results = []

        for search_type in target_types:
            if search_type == "symbol_name":
                if self.symbol_name_searcher is None:
                    raise ValueError("Symbol name searcher is not initialized")
                symbol_by_name = {symbol.name: symbol for symbol in self.symbols}
                for doc, score in self.symbol_name_searcher.search(query, top_k):
                    if score >= threshold and doc in symbol_by_name:
                        results.append(symbol_by_name[doc])

            elif search_type == "symbol_content":
                if self.symbol_content_searcher is None:
                    raise ValueError("Symbol content searcher is not initialized")
                symbol_by_content = {
                    symbol.chunk.code(): symbol for symbol in self.symbols
                }
                for doc, score in self.symbol_content_searcher.search(query, top_k):
                    if score >= threshold and doc in symbol_by_content:
                        results.append(symbol_by_content[doc])

            elif search_type == "keyword":
                if self.keyword_searcher is None:
                    raise ValueError("Keyword searcher is not initialized")
                keyword_map = {keyword.content: keyword for keyword in self.keywords}
                for doc, score in self.keyword_searcher.search(query, top_k):
                    if score >= threshold and doc in keyword_map:
                        results.append(keyword_map[doc])

        return results
