"""
Strategy for filtering top-k lines using vector similarity search followed by LLM refinement.
"""

from typing import List, Tuple, Union, Optional, override, Any

from coderetrx.retrieval.topic_extractor import TopicExtractor
from .base import (
    FilterByVectorAndLLMStrategy,
    RecallStrategyExecutor,
    StrategyExecuteResult,
    deduplicate_elements,
)
from ..smart_codebase import (
    SmartCodebase as Codebase,
    LLMMapFilterTargetType,
    SimilaritySearchTargetType,
    LLMCallMode,
)
from coderetrx.static import Keyword, Symbol, File
from coderetrx.static.codebase import CodeLine
import logging

logger = logging.getLogger(__name__)


class FilterTopkLineByVectorAndLLMStrategy(RecallStrategyExecutor):
    """
    Intelligent filtering strategy that performs line-level vector recall using builtin codeline searcher,
    then uses LLM to judge and select the most relevant lines across all symbols.
    """

    llm_call_mode: LLMCallMode

    def __init__(
        self,
        top_k_by_symbol: int = 5,
        max_queries: int = 20,
        topic_extractor: Optional[TopicExtractor] = None,
        llm_call_mode: LLMCallMode = "traditional",
    ):
        """
        Initialize the intelligent filtering strategy.

        Args:
            top_k: Number of top lines to recall for each symbol from builtin searcher (default: 5)
            max_queries: Maximum number of LLM queries allowed (default: 20)
            topic_extractor: Optional TopicExtractor instance
            llm_call_mode: Mode for LLM calls
        """
        super().__init__(topic_extractor=topic_extractor, llm_call_mode=llm_call_mode)
        self.top_k = top_k_by_symbol
        self.max_queries = max_queries

    def get_strategy_name(self) -> str:
        return "INTELLIGENT_FILTER"

    async def _llm_recall_judgment(
        self, line_candidates: List[Tuple[str, str, str]], query: str
    ) -> List[str]:
        """
        Use LLM to judge and select the most relevant lines from candidates.

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

        # Process in batches
        for i in range(0, len(line_candidates), max_batch_size):
            batch = line_candidates[i : i + max_batch_size]
            batch_selected = await self._process_candidate_batch(batch, query)
            all_selected_lines.extend(batch_selected)

            if len(line_candidates) > max_batch_size:
                logger.info(
                    f"Processed batch {i//max_batch_size + 1}: Selected {len(batch_selected)} lines from {len(batch)} candidates"
                )

        return all_selected_lines

    async def _process_candidate_batch(
        self, line_candidates: List[Tuple[str, str, str]], query: str
    ) -> List[str]:
        """Process a single batch of line candidates."""
        try:
            from coderetrx.utils.llm import call_llm_with_function_call

            # Prepare candidates text
            candidates_text = []
            for i, (line, symbol_name, file_path) in enumerate(line_candidates):
                candidates_text.append(
                    f"{i+1}. [{symbol_name} in {file_path}] {line.strip()}"
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

            logger.info(
                f"LLM selected {len(selected_indices)} lines from batch of {len(line_candidates)}. Reasoning: {reasoning}"
            )

            # Extract selected lines
            selected_lines = []
            for idx in selected_indices:
                if 1 <= idx <= len(line_candidates):
                    selected_lines.append(line_candidates[idx - 1][0])

            return selected_lines

        except Exception as e:
            logger.error(f"Error in LLM recall judgment batch: {e}")
            # Fallback: return first few candidates from this batch
            return [candidate[0] for candidate in line_candidates[:3]]

    async def execute(
        self, codebase: Codebase, prompt: str, subdirs_or_files: List[str]
    ) -> StrategyExecuteResult:
        """
        Execute the intelligent filter strategy using builtin codeline searcher.

        Args:
            codebase: The codebase to search in
            prompt: The prompt for filtering
            subdirs_or_files: List of subdirectories or files to process

        Returns:
            StrategyExecuteResult containing file_paths, elements, and llm_results
        """
        strategy_name = self.get_strategy_name()
        logger.info(f"Using {strategy_name} strategy with builtin codeline searcher")

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
                    f"Using extracted topic '{topic}' for intelligent filtering"
                )

            # Step 1: Use builtin codeline searcher to get line-level results
            logger.info(
                "Step 1 - Using builtin codeline searcher for line-level vector recall"
            )

            # Use the builtin similarity_search with symbol_codeline target type
            all_recalled_lines: list[
                CodeLine
            ] = await codebase.similarity_search_lines_per_symbol(
                query=topic,
                threshold=0,
                top_k=self.top_k,
            )

            logger.info(
                f"Builtin codeline searcher found {len(all_recalled_lines)} line candidates"
            )

            if not all_recalled_lines:
                logger.info("No lines recalled from builtin codeline searcher")
                return StrategyExecuteResult(file_paths=[], elements=[], llm_results=[])

            # Filter by subdirectories if specified
            if subdirs_or_files:
                filtered_lines = []
                for code_line in all_recalled_lines:
                    file_path = str(code_line.symbol.file.path)
                    if any(file_path.startswith(subdir) for subdir in subdirs_or_files):
                        filtered_lines.append(code_line)
                all_recalled_lines = filtered_lines
                logger.info(
                    f"Filtered to {len(all_recalled_lines)} lines from specified subdirectories"
                )

            # Sort by vector similarity score (descending) - CodeLine objects should have score attribute
            all_recalled_lines.sort(key=lambda x: x.score, reverse=True)
            logger.info("Sorted line candidates by vector similarity score")

            # Step 2: LLM processing with dynamic batch evaluation
            logger.info("Step 2 - LLM processing with dynamic batch evaluation")
            selected_file_paths = set()
            recalled_symbols = []
            recalled_symbol_ids = set()
            batch_size = 30  # Smaller batch size for better LLM processing

            # Process candidates in dynamic batches
            current_index = 0
            batch_count = 0
            query_count = 0

            # Progress bar for LLM queries
            from tqdm import tqdm

            with tqdm(
                total=self.max_queries, desc="LLM batch queries", unit="query"
            ) as pbar:
                while (
                    current_index < len(all_recalled_lines)
                    and query_count < self.max_queries
                ):
                    # Collect next batch of valid candidates
                    batch_candidates: list[CodeLine] = []

                    while len(batch_candidates) < batch_size and current_index < len(
                        all_recalled_lines
                    ):
                        entry = all_recalled_lines[current_index]
                        current_index += 1

                        # Skip if symbol already recalled
                        symbol_id = (
                            entry.symbol.id
                            or f"{entry.symbol.name}_{entry.symbol.file.path}"
                        )
                        if symbol_id in recalled_symbol_ids:
                            continue

                        # Skip if file path already selected
                        file_path = str(entry.symbol.file.path)
                        if file_path in selected_file_paths:
                            continue

                        batch_candidates.append(entry)

                    if not batch_candidates:
                        break

                    batch_count += 1

                    # Prepare batch data for LLM judgment
                    batch_data = [
                        (
                            entry.line_content,
                            entry.symbol.name,
                            str(entry.symbol.file.path),
                        )
                        for entry in batch_candidates
                    ]

                    # Get LLM judgment on this batch
                    selected_lines = await self._llm_recall_judgment(batch_data, prompt)
                    query_count += 1
                    pbar.update(1)  # Update progress bar

                    # Mark selected symbols as recalled
                    for selected_line in selected_lines:
                        for entry in batch_candidates:
                            symbol_id = (
                                entry.symbol.id
                                or f"{entry.symbol.name}_{entry.symbol.file.path}"
                            )
                            file_path = str(entry.symbol.file.path)
                            if (
                                entry.line_content == selected_line
                                and symbol_id not in recalled_symbol_ids
                            ):
                                recalled_symbols.append(entry.symbol)
                                recalled_symbol_ids.add(symbol_id)
                                selected_file_paths.add(file_path)
                                logger.info(
                                    f"Selected symbol {entry.symbol.name} from {file_path} (score: {entry.score:.3f})"
                                )
                                break

                    logger.info(
                        f"Processed batch {batch_count}: Selected {len(selected_lines)} symbols from {len(batch_candidates)} candidates"
                    )

            file_paths = list(selected_file_paths)
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
