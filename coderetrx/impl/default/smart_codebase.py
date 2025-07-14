from collections import defaultdict
from typing import Optional, List, Tuple, Any, Literal, TYPE_CHECKING
from coderetrx.static import Codebase, Dependency
from coderetrx.static.codebase import Symbol, Keyword, File, CodeLine
from coderetrx.retrieval import SmartCodebase as SmartCodebaseBase, LLMCallMode
from coderetrx.retrieval.smart_codebase import (
    CodeMapFilterResult,
    LLMMapFilterTargetType,
    SimilaritySearchTargetType,
)
from attrs import define, field
from coderetrx.utils.embedding import (
    embed_batch_with_retry,
    SimilaritySearcher,
)
import logging
from pathlib import Path
from tqdm.asyncio import tqdm
from .prompt import (
    llm_filter_prompt_template,
    llm_mapping_prompt_template,
    llm_filter_function_call_system_prompt,
    llm_mapping_function_call_system_prompt,
    filter_and_mapping_function_call_user_prompt_template,
    get_filter_function_definition,
    get_mapping_function_definition,
)
import asyncio

if TYPE_CHECKING:
    from .factory import SmartCodebaseSettings

logger = logging.getLogger(__name__)


def get_smart_codebase_settings() -> "SmartCodebaseSettings":
    from .factory import SmartCodebaseSettings

    return SmartCodebaseSettings()


@define
class SmartCodebase(SmartCodebaseBase):
    symbol_name_searcher: Optional["SimilaritySearcher"] = field(default=None)
    symbol_content_searcher: Optional["SimilaritySearcher"] = field(default=None)
    keyword_searcher: Optional["SimilaritySearcher"] = field(default=None)
    # unified codeline searcher for all symbols with metadata filtering
    codeline_searcher: Optional["SimilaritySearcher"] = field(default=None)
    settings: "SmartCodebaseSettings" = field(factory=get_smart_codebase_settings)

    def _fixup_path(self, path: str) -> str:
        original_path = path
        path = path.replace("\\", "/")

        # If it's an absolute path, try to make it relative to the codebase directory
        if Path(path).is_absolute():
            try:
                abs_codebase_dir = Path(self.dir).resolve()
                abs_path = Path(path).resolve()
                path = str(abs_path.relative_to(abs_codebase_dir))
            except ValueError:
                print(
                    f"Warning: Path {original_path} is not within codebase directory {self.dir}"
                )
        else:
            # For relative paths, strip leading slash and handle current directory
            path = path.lstrip("/")
            if path == "." or path == "./":
                path = ""
        return path

    def _get_filtered_elements(
        self,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: Optional[List[str]] = None,
        additional_code_elements: Optional[List[Any]] = None,
    ) -> List[Any]:
        """
        Filter code elements based on the specified type and prompt using LLM.

        Args:
            target_type: The type of code element to filter
            subdirs_or_files: List of subdirectories or files to filter by
            additional_code_elements: Additional code elements to include in filtering

        Returns:
            List of filtered elements
        """
        if subdirs_or_files is None:
            subdirs_or_files = ["/"]
        if additional_code_elements is None:
            additional_code_elements = []
        # Get elements based on type
        elements = []
        if target_type in ["file_name", "file_content"]:
            elements = [file for file in self.source_files]
        elif target_type in ["symbol_name", "symbol_content"]:
            elements = [symbol for symbol in self.symbols]
        elif target_type in ["root_symbol_name", "root_symbol_content"]:
            elements = [symbol for symbol in self.symbols if not symbol.chunk.parent]
        elif target_type in ["leaf_symbol_name", "leaf_symbol_content"]:
            elements = [
                symbol
                for symbol in self.symbols
                if not self.childs_of_symbol[symbol.id]  # No children means leaf
            ]
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
        subdirs_or_files = [self._fixup_path(path) for path in subdirs_or_files]
        for element in elements:
            if isinstance(element, Symbol) and any(
                str(element.file.path).startswith(subdir) for subdir in subdirs_or_files
            ):
                filtered_elements.append(element)
            elif isinstance(element, File) and any(
                str(element.path).startswith(subdir) for subdir in subdirs_or_files
            ):
                filtered_elements.append(element)
            elif isinstance(element, Keyword):
                for path in element.referenced_by:
                    if any(str(path).startswith(subdir) for subdir in subdirs_or_files):
                        filtered_elements.append(element)
                        break

        # Add additional elements if provided
        if additional_code_elements:
            filtered_elements.extend(additional_code_elements)

        logger.debug(f"filtered_elements size: {len(filtered_elements)}")
        elements = filtered_elements if filtered_elements else elements

        return elements

    @classmethod
    def new(
        cls,
        id: str,
        dir: Path,
        url: Optional[str] = None,
        lazy: bool = False,
        version: str = "v0.0.1",
        ignore_tests: bool = True,
        settings: Optional["SmartCodebaseSettings"] = None,
    ) -> "SmartCodebase":
        if settings is None:
            settings = get_smart_codebase_settings()
        codebase = Codebase.new(id, dir, url, lazy, version, ignore_tests)
        return cls(
            id=codebase.id,
            dir=codebase.dir,
            source_files=codebase.source_files,
            dependency_files=codebase.dependency_files,
            symbols=codebase.symbols,
            keywords=codebase.keywords,
            dependencies=codebase.dependencies,
            settings=settings,
        )

    async def llm_filter(
        self,
        prompt: str,
        target_type: LLMMapFilterTargetType,
        subdirs_or_files: Optional[List[str]] = None,
        additional_code_elements: Optional[List[Any]] = None,
        prompt_template: str = llm_filter_prompt_template,
        llm_call_mode: LLMCallMode = "traditional",
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Filter code elements based on the specified type and prompt using LLM.

        Args:
            prompt: The filtering prompt
            target_type: The type of code element to filter
            subdirs_or_files: List of subdirectories or files to filter byf
            additional_code_elements: Additional code elements to include in filtering
            prompt_template: The prompt template to use (for traditional mode)
            llm_call_mode: The LLM call mode - "traditional", "function_call"
            model_id: The model ID to use for the LLM call
            If model_id is not provided, the model ID will be set to the value of the LLM_MAPFILTER_MODEL_ID environment variable

        Returns:
            A tuple containing:
            - Filtered elements
            - LLM results with reasoning

        Note:
            For symbol_content filtering, duplicate analysis may occur—for example, both a class and its methods might be analyzed.
            This is intentional. Analyzing only the root symbols may lead to suboptimal LLM performance if the symbol content is too large,
            while analyzing only the leaf symbols might miss certain features that require a broader scope to identify.
        """
        if subdirs_or_files is None:
            subdirs_or_files = ["/"]
        if additional_code_elements is None:
            additional_code_elements = []

        elements = self._get_filtered_elements(
            target_type, subdirs_or_files, additional_code_elements
        )
        if (
            target_type == "keyword"
            or target_type == "symbol_name"
            or target_type == "dependency_name"
        ):
            model_id = self.settings.llm_mapfilter_special_model_id
        if llm_call_mode == "function_call":
            right_elements, llm_results = (
                await self._process_elements_with_function_call(
                    elements, target_type, prompt, is_filter=True, model_id=model_id
                )
            )
        else:
            right_elements, llm_results = await self._process_elements_traditional(
                elements, target_type, prompt, prompt_template, model_id=model_id
            )

        return right_elements, llm_results

    async def _process_elements_traditional(
        self,
        elements: List[Any],
        target_type: str,
        prompt: str,
        prompt_template: str,
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Process elements using traditional prompt-based approach."""
        from coderetrx.utils.llm import call_llm_with_fallback

        # Process elements in batches
        async def process_element_batch(
            elements_batch: List[Any],
            target_type: str,
            requirement: str,
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
                model_ids = [
                    model_id or self.settings.llm_mapfilter_model_id,
                    self.settings.llm_fallback_model_id,
                ]
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
        max_batch_length = self.settings.llm_mapfilter_max_batch_length
        max_batch_size = self.settings.llm_mapfilter_max_batch_size

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

        max_concurrent_requests = self.settings.llm_max_concurrent_requests
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def process_with_semaphore(batch):
            async with semaphore:
                return await process_element_batch(batch, target_type, prompt)

        gather_results = await tqdm.gather(
            *[process_with_semaphore(batch) for batch in element_tasks],
            desc="Processing elements",
            total=len(element_tasks),
        )

        flattened_llm_results = []
        flattened_right_elements = []

        for llm_result, right_elements in gather_results:
            flattened_llm_results.extend(llm_result)
            flattened_right_elements.extend(right_elements)

        return flattened_right_elements, flattened_llm_results

    async def _process_elements_with_function_call(
        self,
        elements: List[Any],
        target_type: str,
        prompt: str,
        is_filter: bool = True,
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Process elements using function call approach."""
        from coderetrx.utils.llm import call_llm_with_function_call

        # Process elements in batches
        async def process_element_batch_with_function_call(
            elements_batch: List[Any],
            target_type: str,
            requirement: str,
            is_filter: bool,
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

            # Prepare prompts and function definition
            system_prompt = (
                llm_filter_function_call_system_prompt
                if is_filter
                else llm_mapping_function_call_system_prompt
            )
            user_prompt = filter_and_mapping_function_call_user_prompt_template.format(
                code_elements=text,
                code_element_number=len(elements_batch),
                code_element_number_minus_one=len(elements_batch) - 1,
                requirement=requirement,
            )

            function_definition = (
                get_filter_function_definition()
                if is_filter
                else get_mapping_function_definition()
            )

            try:
                # Call LLM with function call
                model_ids = [
                    model_id or self.settings.llm_function_call_model_id,
                    self.settings.llm_fallback_model_id,
                ]
                function_args = await call_llm_with_function_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    function_definition=function_definition,
                    model_ids=model_ids,
                )

                # Parse function call results
                analyses = function_args.get("analyses", [])
                llm_results = []
                right_elements = []

                for i, analysis in enumerate(analyses):
                    # Add type checking to handle cases where analysis is not a dict
                    if not isinstance(analysis, dict):
                        logger.warning(
                            f"Analysis item {i} is not a dictionary, got {type(analysis)}: {analysis}"
                        )
                        # Skip this analysis item or create a default one
                        analysis = {
                            "index": i,
                            "reason": f"Invalid analysis format: {type(analysis)}",
                            "result": False if is_filter else "",
                        }

                    index = analysis.get("index", -1)
                    reason = analysis.get("reason", "")
                    result = analysis.get("result")

                    if result is None:
                        logger.warning(
                            f"Analysis item {i} missing 'result' field: {analysis}"
                        )

                    # Create CodeMapFilterResult object
                    llm_result = CodeMapFilterResult(
                        index=index,
                        reason=reason,
                        result=result,
                    )
                    llm_results.append(llm_result)

                    # For filter mode, include elements where result is True
                    # For mapping mode, include all elements with non-empty results
                    if is_filter and result is True:
                        if 0 <= index < len(elements_batch):
                            right_elements.append(elements_batch[index])
                    elif not is_filter and result:  # For mapping, result is string
                        if 0 <= index < len(elements_batch):
                            right_elements.append(elements_batch[index])

                return llm_results, right_elements

            except Exception as e:
                logger.error(f"Function call processing failed: {str(e)}")
                return [], []

        # Batch elements based on content length (same logic as traditional)
        element_tasks = []
        cur_batch = []
        max_batch_length = self.settings.llm_mapfilter_max_batch_length
        max_batch_size = self.settings.llm_mapfilter_max_batch_size

        for element in elements:
            if isinstance(element, Symbol):
                content_length = len(element.chunk.code())
            elif isinstance(element, File):
                content_length = len(element.content)
            elif isinstance(element, Dependency):
                content_length = len(element.name)
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

        max_concurrent_requests = self.settings.llm_max_concurrent_requests
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def process_with_semaphore(batch):
            async with semaphore:
                return await process_element_batch_with_function_call(
                    batch, target_type, prompt, is_filter
                )

        gather_results = await tqdm.gather(
            *[process_with_semaphore(batch) for batch in element_tasks],
            desc="Processing elements with function call",
            total=len(element_tasks),
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
        subdirs_or_files: Optional[List[str]] = None,
        additional_code_elements: Optional[List[Any]] = None,
        llm_call_mode: LLMCallMode = "traditional",
        model_id: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Map code elements based on the specified type and prompt using LLM.

        Args:
            prompt: The mapping prompt
            target_type: The type of code element to map
            subdirs_or_files: List of subdirectories or files to filter by
            additional_code_elements: Additional code elements to include in mapping
            llm_call_mode: The LLM call mode - "traditional", "function_call"

        Returns:
            A tuple containing:
            - Mapped elements
            - LLM results with mapping content
        """
        if subdirs_or_files is None:
            subdirs_or_files = ["/"]
        if additional_code_elements is None:
            additional_code_elements = []

        if llm_call_mode == "function_call":
            # Get and filter elements using the shared method
            elements = self._get_filtered_elements(
                target_type, subdirs_or_files, additional_code_elements
            )
            return await self._process_elements_with_function_call(
                elements, target_type, prompt, is_filter=False, model_id=model_id
            )
        else:
            return await self.llm_filter(
                prompt=prompt,
                target_type=target_type,
                subdirs_or_files=subdirs_or_files,
                additional_code_elements=additional_code_elements,
                prompt_template=llm_mapping_prompt_template,
                llm_call_mode="traditional",
                model_id=model_id,
            )

    async def similarity_search(
        self,
        target_types: List[SimilaritySearchTargetType],
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 100,
        subdirs_or_files: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Perform similarity search on symbols based on the specified types.

        Args:
            target_types: List of types to search on (symbol_name, symbol_content, keyword, symbol_codeline)
            query: The search query
            threshold: Similarity threshold, defaults to value from environment variable
            top_k: Maximum number of results to return

        Returns:
            List of symbols or keywords that match the search criteria

        Raises:
            ValueError: If searcher is not initialized for the requested type
        """
        if subdirs_or_files is None:
            subdirs_or_files = ["/"]

        threshold = (
            self.settings.similarity_search_threshold
            if threshold is None
            else threshold
        )
        results = []

        for search_type in target_types:
            if search_type == "symbol_name":
                if self.symbol_name_searcher is None:
                    raise ValueError("Symbol name searcher is not initialized")
                symbol_by_name = {symbol.name: symbol for symbol in self.symbols}
                for doc, score in await self.symbol_name_searcher.asearch_with_score(query, top_k, threshold):
                    if score >= threshold and doc in symbol_by_name:
                        results.append(symbol_by_name[doc])

            elif search_type == "symbol_content":
                if self.symbol_content_searcher is None:
                    raise ValueError("Symbol content searcher is not initialized")
                symbol_by_content = {
                    symbol.chunk.code(): symbol for symbol in self.symbols
                }
                for doc, score in await self.symbol_content_searcher.asearch_with_score(
                    query, top_k, threshold
                ):
                    if score >= threshold and doc in symbol_by_content:
                        results.append(symbol_by_content[doc])

            elif search_type == "keyword":
                if self.keyword_searcher is None:
                    raise ValueError("Keyword searcher is not initialized")
                keyword_map = {keyword.content: keyword for keyword in self.keywords}
                for doc, score in await self.keyword_searcher.asearch_with_score(query, top_k, threshold):
                    if score >= threshold and doc in keyword_map:
                        results.append(keyword_map[doc])

        return results

    async def similarity_search_lines_per_symbol(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 10,
        scope: Literal[
            "root_symbol", "leaf_symbol", "symbol", "class", "function"
        ] = "symbol",
        subdirs_or_files: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Search for similar lines within a specific symbol using metadata filtering.

        Args:
            query: The search query
            threshold: Similarity threshold, defaults to value from environment variable
            top_k: Maximum number of results to return
            scope: The scope of symbols to search within - "root_symbol", "leaf_symbol", "symbol", "class", "function"

        Returns:
            List of CodeLine objects that match the search criteria within the specified symbol

        Raises:
            ValueError: If symbol codeline searcher is not initialized
        """
        if subdirs_or_files is None:
            subdirs_or_files = ["/"]

        if self.codeline_searcher is None:
            raise ValueError("Symbol codeline searcher is not initialized")
        subdirs_or_files = [self._fixup_path(path) for path in subdirs_or_files]
        threshold = (
            self.settings.similarity_search_threshold
            if threshold is None
            else threshold
        )
        import numpy as np
        
        query_vector = (await embed_batch_with_retry([query]))[0]
        query_vec = np.array(query_vector)
        query_norm = query_vec / np.linalg.norm(query_vec)

        if scope == "symbol":
            symbols = self.symbols
        elif scope == "root_symbol":
            # search only on top-level symbols (those without a parent chunk)
            symbols = [symbol for symbol in self.symbols if not symbol.chunk.parent]
        elif scope == "leaf_symbol":
            symbols = [
                symbol
                for symbol in self.symbols
                if not self.childs_of_symbol[symbol.id]
            ]
        else:
            symbols = [symbol for symbol in self.symbols if symbol.type == scope]
        symbols = [
            symbol
            for symbol in symbols
            if any(
                str(symbol.file.path).startswith(subdir) for subdir in subdirs_or_files
            )
        ]
        
        # Get all data for manual similarity calculation using lists to avoid hashable issues
        results = []
        symbol_tasks = []
        for symbol in symbols:
            task = self.codeline_searcher.asearch_by_vector(
                query_vector=query_vector,
                k=top_k,
                where={"symbol_id": symbol.id},
            )
            symbol_tasks.append((symbol, task))
        
        # Execute search tasks with concurrency limiting to prevent connection overload
        semaphore = asyncio.Semaphore(20)
        
        async def search_with_limit(search_task):
            async with semaphore:
                return await search_task
        
        search_task_results = await tqdm.gather(
            *[search_with_limit(task) for symbol, task in symbol_tasks],
            desc="Getting topk code lines per symbol",
            total=len(symbol_tasks),
        )
        
        # Flatten and sort all results by similarity score
        all_candidates = []
        for idx, (symbol, _) in enumerate(symbol_tasks):
            search_result = search_task_results[idx]
            for doc, score in search_result:
                if score >= threshold:
                    all_candidates.append((doc, symbol, score))
        
        # Sort by similarity score (highest first)
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Convert to CodeLine objects
        for doc, symbol, score in all_candidates:
            results.append(
                CodeLine.new(line_content=doc, symbol=symbol, score=score)
            )

        return results
