"""
Strategy for filtering top-k lines using vector similarity search followed by LLM refinement.
"""

from typing import List, Tuple, Union, Optional, override, Any
from tqdm import tqdm

from coderetrx.retrieval.topic_extractor import TopicExtractor
from .base import (
    FilterByVectorAndLLMStrategy,
    RecallStrategyExecutor,
    StrategyExecuteResult,
    deduplicate_elements,
)
from ..smart_codebase import (
    SmartCodebase as Codebase,
    LLMCallMode,
)
from coderetrx.static.codebase import CodeLine
import logging

logger = logging.getLogger(__name__)

import asyncio


class FilterLinePerSymbolByVectorAndLLMStrategy(RecallStrategyExecutor):
    """
    Intelligent filtering strategy that performs line-level vector recall using builtin codeline searcher,
    then uses LLM to judge and select the relevant lines across all symbols.
    """

    llm_call_mode: LLMCallMode
    name = "FILTER_LINE_PER_SYMBOL_BY_VECTOR_AND_LLM"

    def __init__(
            self,
            topic_extractor: Optional[TopicExtractor] = None,
            llm_call_mode: LLMCallMode = "traditional",
    ):
        """
        Initialize the filtering strategy.

        Args:
            top_k: Number of top lines to recall for each symbol from builtin searcher (default: 15)
            max_iteration: Maximum number of LLM queries allowed (default: 30)
            topic_extractor: Optional TopicExtractor instance
            llm_call_mode: Mode for LLM calls
        """
        super().__init__(topic_extractor=topic_extractor, llm_call_mode=llm_call_mode)
        self.top_k = 15
        self.max_iteration = int(self.top_k * 2)

    def get_strategy_name(self) -> str:
        return "INTELLIGENT_FILTER"

    async def _llm_recall_judgment(
            self, line_candidates: List[CodeLine], query: str
    ) -> List[CodeLine]:
        """
        Use LLM to judge and select the relevant lines from candidates.

        Args:
            line_candidates: List of (line_content, symbol_name, file_path) tuples
            query: Original query for relevance judgment

        Returns:
            List of selected line contents
        """
        if not line_candidates:
            return []

        max_batch_size = 30
        all_selected_lines = []

        # Create tasks for all batches
        tasks = []
        for i in range(0, len(line_candidates), max_batch_size):
            batch = line_candidates[i: i + max_batch_size]
            task = self._process_candidate_batch(batch, query)
            tasks.append(task)

        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)

        # Combine results
        for batch_selected in batch_results:
            all_selected_lines.extend(batch_selected)

        if len(line_candidates) > max_batch_size:
            logger.debug(
                f"Processed {len(tasks)} batches concurrently: Selected {len(all_selected_lines)} total lines"
            )

        return all_selected_lines

    async def _process_candidate_batch(
        self, line_candidates: List[CodeLine], query: str, retry_cnt=0
    ) -> List[CodeLine]:
        """Process a single batch of line candidates."""
        try:
            from coderetrx.utils.llm import call_llm_with_function_call

            # Prepare candidates text
            candidates_text = []
            for i, codeline in enumerate(line_candidates):
                line = codeline.line_content
                symbol_name = codeline.symbol.name
                file_path = str(codeline.symbol.file.path)
                candidates_text.append(
                    f"{i + 1}. [{symbol_name} in {file_path}] {line.strip()}"
                )

            candidates_str = "\n".join(candidates_text)

            # Define function for line selection
            function_definition = {
                "name": "select_relevant_lines",
                "description": "Select the most relevant code lines based on the query criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of indices (1-based) of the most relevant lines",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation for the selection",
                        },
                    },
                    "required": ["selected_indices", "reasoning"],
                },
            }

            system_prompt = f"""You are analyzing code lines to find the most relevant ones for a specific query.

Query: {query}

Select the most relevant lines from the candidates below. Focus on:
1. Direct relevance to the query requirements
2. Functional significance and completeness
3. Quality and clarity of the code

Be selective and choose only the lines that truly match the query criteria."""

            user_prompt = f"""Here are the candidate code lines:

{candidates_str}

Select the most relevant lines for the query: "{query}"

Call the select_relevant_lines function with your analysis."""

            function_args = await call_llm_with_function_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_definition=function_definition,
                model_ids=["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4"],
            )

            selected_indices = function_args.get("selected_indices", [])
            reasoning = function_args.get("reasoning", "")

            logger.debug(
                f"LLM selected {len(selected_indices)} lines from batch of {len(line_candidates)}. Reasoning: {reasoning}"
            )

            # Extract selected lines
            selected_lines = []
            for idx in selected_indices:
                if 1 <= idx <= len(line_candidates):
                    selected_lines.append(line_candidates[idx - 1])

            return selected_lines

        except Exception as e:
            logger.error(f"Error in LLM recall judgment batch: {e}")
            # Fallback: return first few candidates from this batch
            return [candidate[0] for candidate in line_candidates[:10]]

    async def execute(
        self,
        codebase: Codebase,
        prompt: str,
        subdirs_or_files: List[str],
        target_type: str = "symbol_content",
    ) -> StrategyExecuteResult:
        """
        Execute the filter strategy using builtin codeline searcher.

        Args:
            codebase: The codebase to search in
            prompt: The prompt for filtering
            subdirs_or_files: List of subdirectories or files to process
            target_type: The target_type level for retrieval (default: "symbol_content")

        Returns:
            StrategyExecuteResult containing file_paths, elements, and llm_results
        """
        strategy_name = self.get_strategy_name()
        logger.info(
            f"Using {strategy_name} strategy with builtin codeline searcher and target_type: {target_type}"
        )

        try:
            # Extract topic for vector search
            topic = (
                await self.topic_extractor.extract_topic(
                    input_text=prompt, llm_call_mode=self.llm_call_mode
                )
                if self.topic_extractor
                else prompt
            )

            if not topic:
                logger.warning("Topic extraction failed, using original prompt")
                topic = prompt
            else:
                logger.info(
                    f"Using extracted topic '{topic}' for filtering"
                )

            # Step 1: Use builtin codeline searcher to get line-level results
            logger.info(
                "Step 1 - Using builtin codeline searcher for line-level vector recall"
            )
            # Use the builtin similarity_search with symbol_codeline target type
            vector_per_symbol_recalled_lines: list[CodeLine] = (
                await codebase.similarity_search_lines_per_symbol(
                    query=topic,
                    threshold=0,
                    top_k=self.top_k,
                    scope="symbol",
                    subdirs_or_files=subdirs_or_files,
                )
            )

            logger.info(
                f"Builtin codeline searcher found {len(vector_per_symbol_recalled_lines)} line candidates"
            )

            if not vector_per_symbol_recalled_lines:
                logger.info("No lines recalled from builtin codeline searcher")
                return StrategyExecuteResult(file_paths=[], elements=[], llm_results=[])

            # Sort by vector similarity score (descending) - CodeLine objects should have score attribute
            vector_per_symbol_recalled_lines.sort(key=lambda x: x.score, reverse=True)
            logger.info("Sorted line candidates by vector similarity score")

            for i in range(min(10, len(vector_per_symbol_recalled_lines))):
                entry = vector_per_symbol_recalled_lines[i]
                file_path = str(entry.symbol.file.path)
                logger.debug(
                    f"Candidate {i + 1}: {file_path}, {entry.symbol.name}, {entry.line_content} (score: {entry.score:.3f})"
                )

            # Step 2: LLM processing with dynamic batch evaluation
            logger.info("Step 2 - LLM processing with dynamic batch evaluation")
            recalled_file_paths = set()
            recalled_symbols = []
            recalled_symbol_ids = set()
            # Calculate dynamic batch size based on total candidates
            max_candidates_per_round = int((len(vector_per_symbol_recalled_lines) / self.top_k) * 0.2)
            logger.info(f"Max candidates per round: {max_candidates_per_round}")

            # Process candidates in dynamic batches
            current_round_lines = vector_per_symbol_recalled_lines

            for round in tqdm(range(self.max_iteration)):
                visited_symbol = set()
                line_idx = 0
                next_round_heading_lines = []
                current_round_candidates: list[CodeLine] = []

                while len(current_round_candidates) < max_candidates_per_round and line_idx < len(
                    current_round_lines
                ):
                    entry = current_round_lines[line_idx]
                    line_idx += 1
                    # Skip if symbol already visited
                    if entry.symbol.id in visited_symbol:
                        next_round_heading_lines.append(entry)
                        continue

                    # Skip if symbol already recalled
                    symbol_id = (
                        entry.symbol.id
                        or f"{entry.symbol.name}_{entry.symbol.file.path}"
                    )
                    if symbol_id in recalled_symbol_ids:
                        continue

                    # Skip if file path already selected
                    file_path = str(entry.symbol.file.path)
                    if file_path in recalled_file_paths:
                        continue

                    visited_symbol.add(symbol_id)
                    current_round_candidates.append(entry)

                if not current_round_candidates:
                    break

                # Get LLM judgment on this batch
                selected_lines = await self._llm_recall_judgment(current_round_candidates, prompt)

                # Mark selected symbols as recalled
                for entry in selected_lines:
                    recalled_symbols.append(entry.symbol)
                    recalled_symbol_ids.add(entry.symbol.id)
                    recalled_file_paths.add(str(entry.symbol.file.path))
                    logger.debug(
                        f"Selected symbol '{entry.symbol.name}' from {entry.symbol.file.path} (score: {entry.score:.3f})"
                    )
                    break

                logger.debug(
                    f"Processed round {round}: Selected {len(selected_lines)} symbols from {len(current_round_candidates)} candidates"
                )
                current_round_lines = next_round_heading_lines + current_round_lines[line_idx:]

            file_paths = list(recalled_file_paths)
            logger.info(
                f"Intelligent filtering completed. Selected {len(file_paths)} files from {len(recalled_symbols)} symbols."
            )

            return StrategyExecuteResult(
                file_paths=file_paths,
                elements=deduplicate_elements(recalled_symbols),
                llm_results=[],
            )

        except Exception as e:
            logger.error(f"Error in {strategy_name} strategy: {e}")
            raise e
