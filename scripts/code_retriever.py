from typing import Literal
from dotenv import load_dotenv
from coderetrx.retrieval import LLMCallMode
load_dotenv()
import asyncio
import json
from pathlib import Path
from coderetrx.impl.default import CodebaseFactory, TopicExtractor
from coderetrx.retrieval import coderetrx_filter, llm_traversal_filter
from coderetrx.retrieval.code_recall import CodeRecallSettings, CoarseRecallStrategyType
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from coderetrx.utils.path import get_data_dir
from coderetrx.utils.llm import llm_settings
from coderetrx.utils.cost_tracking import calc_llm_costs, calc_input_tokens, calc_output_tokens
import logging
import datetime
import argparse
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeRetriever:
    def __init__(self, repo_url: str, coarse_recall_strategy: CoarseRecallStrategyType = "line_per_symbol", use_function_call: bool = True):
        self.repo_url = repo_url
        self.repo_path = get_data_dir() / "repos" / get_repo_id(repo_url)
        self.topic_extractor = TopicExtractor()
        self.coarse_recall_strategy: CoarseRecallStrategyType = coarse_recall_strategy 
        if coarse_recall_strategy != "precise" and coarse_recall_strategy not in CoarseRecallStrategyType.__args__:
            raise ValueError(f"Invalid coarse_recall_strategy '{coarse_recall_strategy}'. Must be one of: {CoarseRecallStrategyType.__args__} or 'precise'")
        self.use_function_call = use_function_call

    def generate_prompts(self, limit: int = 10) -> list[str]:
        all_prompts = []
        
        # Get the absolute path of the query file from queries folder
        query_file = Path(__file__).parent.parent / "bench" / "queries.json"
        
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
                for query in queries:
                    if len(all_prompts) >= limit:
                        break
                    # Extract filter prompts from resources
                    filter_prompt = query.get('filter_prompt')
                    if filter_prompt:
                        all_prompts.append(filter_prompt)
        except Exception as e:
            logger.warning(f"Failed to load query file {query_file}: {e}")
            
        logger.info(f"Generated {len(all_prompts)} prompts for analysis")
        return all_prompts[:limit]
        
    def prepare_codebase(self):
        database_path = get_data_dir() / "databases" / f"{get_repo_id(self.repo_url)}.json"
        clone_repo_if_not_exists(self.repo_url, str(self.repo_path))

        if database_path.exists():
            codebase = CodebaseFactory.from_json(
                json.load(open(database_path, "r", encoding="utf-8"))
            )
        else:
            codebase = CodebaseFactory.new(get_repo_id(self.repo_url), self.repo_path)
        with open(f"{self.repo_path}.json", "w") as f:
            json.dump(codebase.to_json(), f, indent=4)
        return codebase

    async def find_codes(self, subdirs: list[str] = ["/"], enable_secondary_recall: bool = False, limit: int = 10) -> tuple[list[dict], dict, dict]:
        """Find potential codes in the codebase using the specified coarse_recall_strategy"""
        start_time = time.time()
        timing_info = {
            "total_duration": 0.0,
            "preparation_duration": 0.0,
            "prompt_durations": [],
            "average_prompt_duration": 0.0
        }
        cost_info = {
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "prompt_costs": [],
            "average_prompt_cost": 0.0
        }
        
        prep_start = time.time()
        codebase = self.prepare_codebase()
        prep_end = time.time()
        timing_info["preparation_duration"] = prep_end - prep_start
        
        code_prompts = self.generate_prompts(limit)
        all_codes = []

        # Determine function call coarse_recall_strategy based on the use_function_call parameter
        llm_call_coarse_recall_strategy: LLMCallMode = "function_call" if self.use_function_call else "traditional"

        print(f"\nUsing coarse_recall_strategy: {self.coarse_recall_strategy}")
        print(f"Preparation time: {timing_info['preparation_duration']:.2f} seconds")
        
        for prompt in code_prompts:
            print(f"\n{'='*80}")
            print(f"Searching for codes with prompt: {prompt}")
            print(f"{'='*80}")
            logger.debug(f"Searching for codes with prompt: {prompt} (coarse_recall_strategy: {self.coarse_recall_strategy})")
            
            prompt_start_time = time.time()
            
            # Track cost before this prompt
            log_path = llm_settings.get_json_log_path()
            # Ensure log file exists by creating initial entry
            if not log_path.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.touch()
                
            initial_cost = await calc_llm_costs(log_path)
            initial_input_tokens = calc_input_tokens(log_path)
            initial_output_tokens = calc_output_tokens(log_path)
            
            result = []
            llm_output = []
            
            try:
                # Create settings with appropriate llm_call_coarse_recall_strategy
                settings = CodeRecallSettings(llm_call_mode=llm_call_coarse_recall_strategy)
                
                if self.coarse_recall_strategy == "precise":
                    result, llm_output = await llm_traversal_filter(
                        codebase=codebase,
                        subdirs_or_files=subdirs,
                        prompt=prompt,
                        target_type="symbol_content",
                        topic_extractor=self.topic_extractor,
                        settings=settings,
                        enable_secondary_recall=enable_secondary_recall
                    )
                else:
                    # Use optimized coarse_recall_strategies: file_name, symbol, line_per_symbol, auto, custom
                    result, llm_output = await coderetrx_filter(
                        codebase=codebase,
                        subdirs_or_files=subdirs,
                        prompt=prompt,
                        target_type="symbol_content",
                        coarse_recall_strategy=self.coarse_recall_strategy,
                        topic_extractor=None,
                        settings=settings,
                        enable_secondary_recall=enable_secondary_recall
                    )
                
                if result:
                    # Calculate cost after this prompt
                    final_cost = await calc_llm_costs(llm_settings.get_json_log_path())
                    final_input_tokens = calc_input_tokens(llm_settings.get_json_log_path())
                    final_output_tokens = calc_output_tokens(llm_settings.get_json_log_path())
                    
                    prompt_cost = final_cost - initial_cost
                    prompt_input_tokens = final_input_tokens - initial_input_tokens
                    prompt_output_tokens = final_output_tokens - initial_output_tokens
                    
                    # Add to cost_info
                    cost_info["prompt_costs"].append({
                        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        "cost": prompt_cost,
                        "input_tokens": prompt_input_tokens,
                        "output_tokens": prompt_output_tokens
                    })
                    
                    filtered_llm_output = None
                    if llm_output:
                        filtered_llm_output = [r for r in llm_output if hasattr(r, 'result') and r.result]
                    
                    code_info = {
                        "code_type": prompt,
                        "locations": result,
                        "llm_analysis": filtered_llm_output
                    }
                    all_codes.append(code_info)
                    
                    print(f"\nFound {len(result)} potential issues:")
                    print(f"code Type: {prompt}")
                    print("\nLocations:")
                    for i, location in enumerate(result, 1):
                        print(f"  {i}. {location}")
                    
                    if filtered_llm_output:
                        print(f"\nLLM Analysis: Found {len(filtered_llm_output)} potential security risks out of {len(llm_output)} analyzed code snippets")
                        for i, risky_result in enumerate(filtered_llm_output[:3], 1):
                            print(f"  {i}. Risk at index {risky_result.index}: {risky_result.reason}")
                        if len(filtered_llm_output) > 3:
                            print(f"  ... and {len(filtered_llm_output) - 3} more risks")
                    elif llm_output:
                        print(f"\nLLM Analysis: Analyzed {len(llm_output)} code snippets, no security risks detected")
                    
                    logger.info(f"Found {len(result)} potential issues for prompt: {prompt}")
                else:
                    print(f"\nNo issues found for this prompt")
                    logger.info(f"No issues found for prompt: {prompt}")
                    
            except Exception as e:
                print(f"\nError while processing prompt: {e}")
                logger.error(f"Error while processing prompt '{prompt}': {e}")
            finally:
                prompt_end_time = time.time()
                prompt_duration = prompt_end_time - prompt_start_time
                timing_info["prompt_durations"].append({
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "duration": prompt_duration
                })
                print(f"Prompt processing time: {prompt_duration:.2f} seconds")
                
                # Calculate and print per-prompt cost for all cases
                final_cost = await calc_llm_costs(llm_settings.get_json_log_path())
                final_input_tokens = calc_input_tokens(llm_settings.get_json_log_path())
                final_output_tokens = calc_output_tokens(llm_settings.get_json_log_path())
                
                prompt_cost = final_cost - initial_cost
                prompt_input_tokens = final_input_tokens - initial_input_tokens
                prompt_output_tokens = final_output_tokens - initial_output_tokens
                
                # Add to cost_info even if no results found
                if not result:
                    cost_info["prompt_costs"].append({
                        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        "cost": prompt_cost,
                        "input_tokens": prompt_input_tokens,
                        "output_tokens": prompt_output_tokens
                    })
                
                print(f"Prompt cost: ${prompt_cost:.6f} (Input: {prompt_input_tokens}, Output: {prompt_output_tokens})")
            
            print(f"{'='*80}\n")
        
        # Calculate final timing statistics
        end_time = time.time()
        timing_info["total_duration"] = end_time - start_time
        
        if timing_info["prompt_durations"]:
            durations = [p["duration"] for p in timing_info["prompt_durations"]]
            timing_info["average_prompt_duration"] = sum(durations) / len(durations)
        
        # Calculate final cost statistics
        if cost_info["prompt_costs"]:
            costs = [p["cost"] for p in cost_info["prompt_costs"]]
            cost_info["total_cost"] = sum(costs)
            cost_info["average_prompt_cost"] = sum(costs) / len(costs)
            
            input_tokens = [p["input_tokens"] for p in cost_info["prompt_costs"]]
            output_tokens = [p["output_tokens"] for p in cost_info["prompt_costs"]]
            cost_info["total_input_tokens"] = sum(input_tokens)
            cost_info["total_output_tokens"] = sum(output_tokens)
        
        # Print timing summary
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"Total duration: {timing_info['total_duration']:.2f} seconds")
        print(f"Preparation duration: {timing_info['preparation_duration']:.2f} seconds")
        print(f"Average prompt duration: {timing_info['average_prompt_duration']:.2f} seconds")
        print(f"Number of prompts processed: {len(timing_info['prompt_durations'])}")
        print(f"{'='*60}")
        
        # Print cost summary
        print(f"\n{'='*60}")
        print(f"COST SUMMARY")
        print(f"{'='*60}")
        print(f"Total cost: ${cost_info['total_cost']:.6f}")
        print(f"Average prompt cost: ${cost_info['average_prompt_cost']:.6f}")
        print(f"Total input tokens: {cost_info['total_input_tokens']:,}")
        print(f"Total output tokens: {cost_info['total_output_tokens']:,}")
        print(f"Number of prompts processed: {len(cost_info['prompt_costs'])}")
        print(f"{'='*60}")
                
        return all_codes, timing_info, cost_info

    async def save_results(self, codes: list[dict], timing_info: dict, cost_info: dict, output_file: str = "code_report.json"):
        # Create code_reports/{repo_name} directory if it doesn't exist
        repo_id = get_repo_id(self.repo_url)
        code_report_dir = Path("code_reports") / repo_id
        code_report_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure output file is saved in code_reports/{repo_name} directory
        if not Path(output_file).is_absolute():
            output_file = code_report_dir / output_file
        
        def make_serializable(obj, seen=None):
            if seen is None:
                seen = set()
            
            # Handle None values
            if obj is None:
                return None
            
            # Handle primitive types first
            if isinstance(obj, (str, int, float, bool)):
                return obj
            
            # Prevent circular references
            obj_id = id(obj)
            if obj_id in seen:
                return f"<circular reference to {type(obj).__name__}>"
            
            # Handle CodeMapFilterResult specifically
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'CodeMapFilterResult':
                # Directly return dictionary without recursion to avoid circular reference
                result_value = obj.result
                # Handle result value carefully
                if isinstance(result_value, (str, int, float, bool, type(None))):
                    serialized_result = result_value
                else:
                    serialized_result = str(result_value)
                
                return {
                    'index': obj.index,
                    'reason': str(obj.reason),
                    'result': serialized_result
                }
            
            seen.add(obj_id)
            
            try:
                # Handle other Pydantic coarse_recall_strategyls
                if hasattr(obj, 'coarse_recall_strategyl_dump'):
                    try:
                        return obj.coarse_recall_strategyl_dump()
                    except Exception:
                        # Fallback to dict conversion
                        return {k: make_serializable(v, seen) for k, v in obj.__dict__.items()}
                
                # Handle objects with to_json method
                if hasattr(obj, 'to_json'):
                    try:
                        return obj.to_json()
                    except Exception:
                        # Fallback to dict conversion
                        return {k: make_serializable(v, seen) for k, v in obj.__dict__.items()}
                
                # Handle lists and tuples
                if isinstance(obj, (list, tuple)):
                    return [make_serializable(item, seen) for item in obj]
                
                # Handle dictionaries
                if isinstance(obj, dict):
                    return {k: make_serializable(v, seen) for k, v in obj.items()}
                
                # Handle objects with __dict__
                if hasattr(obj, '__dict__'):
                    return {k: make_serializable(v, seen) for k, v in obj.__dict__.items()}
                
                # Final fallback - try JSON serialization
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
            finally:
                seen.discard(obj_id)
        
        serializable_codes = make_serializable(codes)
        
        # Calculate LLM cost and tokens
        cost = await calc_llm_costs(llm_settings.get_json_log_path())
        input_tokens = calc_input_tokens(llm_settings.get_json_log_path())
        output_tokens = calc_output_tokens(llm_settings.get_json_log_path())
        
        report = {
            "repository": self.repo_url,
            "timestamp": str(datetime.datetime.now()),
            "coarse_recall_strategy": self.coarse_recall_strategy,
            "function_call_coarse_recall_strategy": self.use_function_call,
            "llm_cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timing_info": timing_info,
            "cost_info": cost_info,
            "codes": serializable_codes
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"code report saved to {output_file}")

def parse_arguments():
    """Parse command line_per_symbol arguments"""
    parser = argparse.ArgumentParser(
        description="code Finder - Analyze code repositories for potential codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available coarse_recall_strategies:
  file_name   - Uses file_name filtering only (fastest, least accurate)
  symbol_name      - Uses adaptive symbol name vector filtering (balanced speed/accuracy) 
  line_per_symbol  - Uses filtering with line_per_symbol-level vector recall and LLM judgment [DEFAULT]
  dependency  - Uses dependency analysis to find related code
  auto        - Uses LLM to determine the best strategy based on the prompt
  precise     - Uses full LLM processing (most accurate but slowest)
  custom      - Uses custom strategies
        """
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default="https://github.com/ollama/ollama.git",
        help="Repository URL to analyze (default: Ollama)"
    )
    
    parser.add_argument(
        "--coarse_recall_strategy", "-m",
        type=str,
        choices=["file_name", "symbol_name", "line_per_symbol", "dependency", "auto", "precise", "custom"],
        default="line_per_symbol",
        help="Analysis coarse_recall_strategy (default: line_per_symbol)"
    )
    
    parser.add_argument(
        "--subdirs", "-s",
        nargs="+",
        default=["/"],
        help="Subdirectories to analyze (default: entire repo)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Number of prompts to test (default: 10)"
    )

    parser.add_argument(
        "--sec",
        action="store_true",
        help="Enable secondary recall (default: False)"
    )

    parser.add_argument(
        "--use-function-call", "-f",
        action="store_true",
        help="Use function call (default: False, uses traditional LLM calls)"
    )

    return parser.parse_args()

async def main():
    args = parse_arguments()
    
    print(f"code Finder - Repository Analysis")
    print(f"Repository: {args.repo}")
    print(f"Mode: {args.coarse_recall_strategy}")
    print(f"Subdirectories: {args.subdirs}")
    print("-" * 60)
    
    try:
        # Run the code finder
        code_finder = CodeRetriever(args.repo, coarse_recall_strategy=args.coarse_recall_strategy, use_function_call=args.use_function_call)
        
        codes, timing_info, cost_info = await code_finder.find_codes(subdirs=args.subdirs, enable_secondary_recall=args.sec, limit=args.limit)

        # Determine secondary recall suffix based on --sec flag
        secondary_recall_suffix = "sec" if args.sec else "pri"

        output_file = f"code_report_{args.limit}_{args.coarse_recall_strategy}_{secondary_recall_suffix}.json"

        await code_finder.save_results(codes, timing_info, cost_info, output_file)
        
        print(f"\n" + "=" * 60)
        print(f"code analysis completed!")
        print(f"Found {len(codes)} different types of potential issues")
        print(f"Results saved to {output_file}")
        print(f"=" * 60)
        
        # Display summary
        if codes:
            print(f"\nSummary of issues found:")
            for i, code in enumerate(codes, 1):
                code_type = code['code_type']
                locations = code['locations']
                print(f"  {i}. {code_type}: {len(locations)} locations")
        else:
            print(f"\nNo potential issues found.")
        
        cost = await calc_llm_costs(llm_settings.get_json_log_path())
        input_tokens = calc_input_tokens(llm_settings.get_json_log_path())
        output_tokens = calc_output_tokens(llm_settings.get_json_log_path())
        print(f"Total LLM cost: ${cost:.6f}")
        print(f"Input tokens: {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")

    except KeyboardInterrupt:
        print(f"\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())