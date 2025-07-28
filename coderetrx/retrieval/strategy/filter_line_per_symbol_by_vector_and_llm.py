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
    BaseLLMBatchProcessor,
)
from ..smart_codebase import (
    SmartCodebase as Codebase,
    LLMCallMode,
)
from coderetrx.static.codebase import CodeLine
from coderetrx.utils.cost_tracking import calc_llm_costs, calc_input_tokens, calc_output_tokens
from coderetrx.utils.llm import llm_settings as default_llm_settings
import logging

logger = logging.getLogger(__name__)

import asyncio


class LinePerSymbolBatchProcessor(BaseLLMBatchProcessor):
    """Batch processor for symbol-based line filtering."""
    
    def format_candidate(self, index: int, codeline: CodeLine) -> str:
        """Format a single candidate for display in the batch."""
        line = codeline.line_content
        symbol_name = codeline.symbol.name
        file_path = str(codeline.symbol.file.path)
        return f"{index + 1}. [{symbol_name} in {file_path}] {line.strip()}"
    
    def get_system_prompt(self, query: str) -> str:
        """Get the system prompt for the LLM."""
        return f"""You are analyzing code lines to find the relevant ones for a specific query.

Query: {query}

Select the relevant lines from the candidates below. Focus on:
1. Direct relevance to the query requirements
2. Functional significance and completeness
3. Quality and clarity of the code

Be selective and choose only the lines that truly match the query criteria."""
    
    def get_user_prompt(self, candidates_str: str, query: str) -> str:
        """Get the user prompt for the LLM."""
        return f"""Here are the candidate code lines:

{candidates_str}

Select the relevant lines for the query: "{query}"

Call the select_relevant_lines function with your analysis."""


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
        self.top_k = 10
        self.max_iteration = int(self.top_k * 3)

    def get_strategy_name(self) -> str:
        return "FILTER_LINE_PER_SYMBOL_BY_VECTOR_AND_LLM"

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
                logger.info(f"Using extracted topic '{topic}' for filtering")

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
            
            # Record initial cost before Step 2
            log_path = (codebase.llm_settings or default_llm_settings).get_json_log_path()
            initial_step2_cost = await calc_llm_costs(log_path)
            initial_step2_input_tokens = calc_input_tokens(log_path)
            initial_step2_output_tokens = calc_output_tokens(log_path)
            recalled_file_paths = set()
            recalled_symbols = []
            recalled_symbol_ids = set()
            # Calculate dynamic batch size based on total candidates
            max_candidates_per_round = int(
                (len(vector_per_symbol_recalled_lines) / self.top_k) * 0.2
            )
            logger.info(f"Max candidates per round: {max_candidates_per_round}")

            # Process candidates in dynamic batches
            current_round_lines = vector_per_symbol_recalled_lines

            for round in tqdm(range(self.max_iteration)):
                visited_symbol = set()
                line_idx = 0
                next_round_heading_lines = []
                current_round_candidates: list[CodeLine] = []

                while len(
                    current_round_candidates
                ) < max_candidates_per_round and line_idx < len(current_round_lines):
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

                # Get LLM judgment on this batch using the batch processor
                batch_processor = LinePerSymbolBatchProcessor(codebase)
                selected_lines = await batch_processor.llm_recall_judgment(
                    current_round_candidates, prompt
                )

                # Mark selected symbols as recalled
                for entry in selected_lines:
                    recalled_symbols.append(entry.symbol)
                    recalled_symbol_ids.add(entry.symbol.id)
                    recalled_file_paths.add(str(entry.symbol.file.path))
                    logger.debug(
                        f"Selected symbol '{entry.symbol.name}' from {entry.symbol.file.path} (score: {entry.score:.3f})"
                    )

                logger.debug(
                    f"Processed round {round}: Selected {len(selected_lines)} symbols from {len(current_round_candidates)} candidates"
                )
                current_round_lines = (
                    next_round_heading_lines + current_round_lines[line_idx:]
                )

            # Record final cost after Step 2
            final_step2_cost = await calc_llm_costs(log_path)
            final_step2_input_tokens = calc_input_tokens(log_path)
            final_step2_output_tokens = calc_output_tokens(log_path)
            
            # Calculate Step 2 cost difference
            step2_cost = final_step2_cost - initial_step2_cost
            step2_input_tokens = final_step2_input_tokens - initial_step2_input_tokens
            step2_output_tokens = final_step2_output_tokens - initial_step2_output_tokens

            logger.info(
                f"Step 2 LLM cost: ${step2_cost:.6f} (Input tokens: {step2_input_tokens}, Output tokens: {step2_output_tokens})"
            )

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
