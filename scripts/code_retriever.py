from typing import Literal

from dotenv import load_dotenv

from codelib.retrieval import LLMCallMode

load_dotenv()

import os

import asyncio
import json
from pathlib import Path
from codelib.impl.default import CodebaseFactory, TopicExtractor
from codelib.retrieval import coderecx_precise, coderecx_optimised
from codelib.retrieval.strategies import CodeRecallSettings
from codelib.utils.git import clone_repo_if_not_exists, get_repo_id, get_data_dir
from codelib.utils.llm import llm_settings
from codelib.utils.cost_tracking import calc_llm_costs, calc_input_tokens, calc_output_tokens
import logging
import chromadb
from codelib.utils import embedding
import datetime
import argparse
import sys
import time

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class CodeRetriever:
    def __init__(self, repo_url: str, mode: Literal["filename", "symbol", "line", "auto", "precise", "custom"] = "line", use_function_call: bool = False):
        """
        Initialize the code finder with a repository URL
        
        Args:
            repo_url: The repository URL to analyze
            mode: Analysis mode - "filename", "symbol", "line", "auto", "precise", "custom"
                - filename: Uses filename filtering only
                - symbol: Uses adaptive symbol vector filtering
                - line: Uses intelligent line-level vector recall and LLM judgment
                - auto: Uses LLM to determine the best strategy based on the prompt
                - precise: Uses full LLM processing (slowest but comprehensive)
            use_function_call: Whether to use function call mode (default: False)
        """
        self.repo_url = repo_url
        self.repo_path = get_data_dir() / "repos" / get_repo_id(repo_url)
        self.topic_extractor = TopicExtractor()
        self.mode: Literal["filename", "symbol", "line", "auto", "precise", "custom"] = self._validate_mode(mode)
        self.use_function_call = use_function_call
        self.setup_environment()

    def _validate_mode(self, mode: Literal["filename", "symbol", "line", "auto", "precise", "custom"]) -> Literal["filename", "symbol", "line", "auto", "precise", "custom"]:
        """Validate and return the mode parameter"""
        valid_modes = ["filename", "symbol", "line", "auto", "precise", "custom"]
        if mode not in valid_modes:
            logger.warning(f"Invalid mode '{mode}'. Valid modes are: {valid_modes}. Defaulting to 'line'.")
            return "line"
        return mode

    def setup_environment(self):
        """Setup the necessary environment for code finding"""
        if os.environ.get("DISABLE_LLM_CACHE", "").lower() == "true":
            del os.environ["DISABLE_LLM_CACHE"]
        
        os.environ["KEYWORD_EMBEDDING"] = "true"
        os.environ["SYMBOL_NAME_EMBEDDING"] = "true"
        os.environ["SYMBOL_CONTENT_EMBEDDING"] = "true"
        
        cache_root = Path(__file__).parent.parent / ".cache"
        chroma_dir = cache_root / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        embedding.chromadb_client = chromadb.PersistentClient(path=str(chroma_dir))
        
        llm_cache_dir = cache_root / "llm"
        llm_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_mode_description(self) -> str:
        """Get description for the current mode"""
        descriptions = {
            "filename": "Uses filename filtering only (fastest, least accurate)",
            "symbol": "Uses adaptive symbol vector filtering (balanced speed/accuracy)",
            "line": "Uses intelligent filtering with line-level vector recall and LLM judgment (most accurate)",
            "auto": "Uses LLM to determine the best strategy based on the prompt",
            "precise": "Uses full LLM processing (most accurate but slowest)",
            "custom": "Uses custom strategies"
        }
        return descriptions.get(self.mode, f"Unknown mode: {self.mode}")

    def _get_key_extraction_mode(self) -> str:
        """Get the key extraction mode from environment variable"""
        use_sentence_extraction = os.environ.get("KEYWORD_SENTENCE_EXTRACTION", "false").lower() == "true"
        return "sentence" if use_sentence_extraction else "word"

    def generate_prompts(self, limit: int = 10) -> list[str]:
        """Load test prompts from a feature_outline_refiner JSON file"""
        all_prompts = []
        
        # Get the absolute path of the feature file from features folder
        feature_file = Path(__file__).parent.parent / "features" / "features.json"
        
        try:
            with open(feature_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
                for feature in features:
                    if len(all_prompts) >= limit:
                        break
                    # Extract filter prompts from resources
                    for resource in feature.get('resources', []):
                        if resource.get('type') == 'ToolCallingResource':
                            filter_prompt = resource.get('tool_input_kwargs', {}).get('filter_prompt')
                            if filter_prompt:
                                all_prompts.append(filter_prompt)
                                if len(all_prompts) >= limit:
                                    break
        except Exception as e:
            logger.warning(f"Failed to load feature file {feature_file}: {e}")
            
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
        """Find potential codes in the codebase using the specified mode"""
        start_time = time.time()
        timing_info = {
            "total_duration": 0,
            "preparation_duration": 0,
            "prompt_durations": [],
            "average_prompt_duration": 0
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
        
        # Determine function call mode based on the use_function_call parameter
        llm_call_mode: LLMCallMode = "function_call" if self.use_function_call else "traditional"
        
        print(f"\nUsing mode: {self.mode}")
        print(f"Mode description: {self._get_mode_description()}")
        print(f"Function call mode: {llm_call_mode}")
        print(f"Key extraction mode: {self._get_key_extraction_mode()}")
        print(f"Preparation time: {timing_info['preparation_duration']:.2f} seconds")
        
        for prompt in code_prompts:
            print(f"\n{'='*80}")
            print(f"Searching for codes with prompt: {prompt}")
            print(f"Mode: {self.mode}")
            print(f"{'='*80}")
            logger.info(f"Searching for codes with prompt: {prompt} (mode: {self.mode})")
            
            prompt_start_time = time.time()
            
            # Track cost before this prompt
            log_path = llm_settings.get_json_log_path()
            if log_path.exists():
                initial_cost = await calc_llm_costs(log_path)
                initial_input_tokens = calc_input_tokens(log_path)
                initial_output_tokens = calc_output_tokens(log_path)
            else:
                initial_cost = 0.0
                initial_input_tokens = 0
                initial_output_tokens = 0
            
            try:
                # Create settings with appropriate llm_call_mode
                settings = CodeRecallSettings(llm_call_mode=llm_call_mode)
                
                if self.mode == "precise":
                    result, llm_output = await coderecx_precise(
                        codebase=codebase,
                        subdirs_or_files=subdirs,
                        prompt=prompt,
                        granularity="symbol_content",
                        topic_extractor=self.topic_extractor,
                        settings=settings,
                        enable_secondary_recall=enable_secondary_recall
                    )
                else:
                    # Use optimized modes: filename, symbol, line, auto, custom
                    result, llm_output = await coderecx_optimised(
                        codebase=codebase,
                        subdirs_or_files=subdirs,
                        prompt=prompt,
                        granularity="symbol_content",
                        coarse_recall_strategy=self.mode,
                        topic_extractor=self.topic_extractor,
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
        """Save the code finding results to a JSON file"""
        # Create code_reports/{repo_name} directory if it doesn't exist
        repo_id = get_repo_id(self.repo_url)
        code_report_dir = Path("code_reports") / repo_id
        code_report_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure output file is saved in code_reports/{repo_name} directory
        if not Path(output_file).is_absolute():
            output_file = code_report_dir / output_file
        
        def make_serializable(obj):
            """Convert objects to JSON-serializable format"""
            if hasattr(obj, 'to_json'):
                return obj.to_json()
            elif hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        serializable_codes = make_serializable(codes)
        
        # Calculate LLM cost and tokens
        cost = await calc_llm_costs(llm_settings.get_json_log_path())
        input_tokens = calc_input_tokens(llm_settings.get_json_log_path())
        output_tokens = calc_output_tokens(llm_settings.get_json_log_path())
        
        report = {
            "repository": self.repo_url,
            "timestamp": str(datetime.datetime.now()),
            "mode": self.mode,
            "function_call_mode": self.use_function_call,
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="code Finder - Analyze code repositories for potential codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  filename  - Uses filename filtering only (fastest, least accurate)
  symbol    - Uses adaptive symbol vector filtering (balanced speed/accuracy) 
  line      - Uses intelligent filtering with line-level vector recall and LLM judgment [DEFAULT]
  auto      - Uses LLM to determine the best strategy based on the prompt
  precise   - Uses full LLM processing (most accurate but slowest)
  custom    - Uses custom strategies

Function call modes:
  traditional     - Traditional mode (default)
  function_calling - Use function call mode (enabled with --use_function_call)
        """
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default="https://github.com/ollama/ollama.git",
        help="Repository URL to analyze (default: PaddleX)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["filename", "symbol", "line", "auto", "precise", "custom"],
        default="line",
        help="Analysis mode (default: line)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file name (default: code_report_{mode}.json)"
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
        "--use_function_call", "-f",
        action="store_true",
        help="Use function call mode instead of traditional mode (default: False)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Clear cache to run tests precisely (default: False)"
    )

    parser.add_argument(
        "--sec",
        action="store_true",
        help="Enable secondary recall (default: False)"
    )

    return parser.parse_args()

async def main():
    """Main function with command line argument support"""
    args = parse_arguments()

    if args.test:
        # Clear cache for testing
        cache_root = Path(__file__).parent.parent / ".cache"
        chroma_dir = cache_root / "chroma"
        if chroma_dir.exists():
            import shutil
            shutil.rmtree(chroma_dir)
            print(f"Cleared ChromaDB cache at {chroma_dir}")

        llm_cache_dir = cache_root / "llm"
        if llm_cache_dir.exists():
            import shutil
            shutil.rmtree(llm_cache_dir)
            print(f"Cleared LLM cache at {llm_cache_dir}")

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        logger.setLevel(logging.INFO)
    
    print(f"code Finder - Repository Analysis")
    print(f"Repository: {args.repo}")
    print(f"Mode: {args.mode}")
    print(f"Function Call: {args.use_function_call}")
    
    # Create a temporary code_finder instance to get key extraction mode
    temp_code_finder = CodeRetriever(args.repo, mode=args.mode, use_function_call=args.use_function_call)
    key_extraction_mode = temp_code_finder._get_key_extraction_mode()
    print(f"Key Extraction: {key_extraction_mode}")
    
    print(f"Subdirectories: {args.subdirs}")
    if args.output:
        print(f"Output file: {args.output}")
    
    print("-" * 60)
    
    try:
        # Run the code finder
        code_finder = CodeRetriever(args.repo, mode=args.mode, use_function_call=args.use_function_call)
        
        codes, timing_info, cost_info = await code_finder.find_codes(subdirs=args.subdirs, enable_secondary_recall=args.sec, limit=args.limit)
        
        # Set default output file if not specified
        if not args.output:
            key_extraction_mode = code_finder._get_key_extraction_mode()
            
            # Determine secondary recall suffix based on --sec flag
            secondary_recall_suffix = "_sec" if args.sec else "_pri"
            
            default_filename = f"code_report_{args.limit}_{args.mode}_{key_extraction_mode}_{args.use_function_call}{secondary_recall_suffix}.json"
            output_file = default_filename
        else:
            output_file = args.output
            
        await code_finder.save_results(codes, timing_info, cost_info, output_file)
        
        print(f"\n" + "="*60)
        print(f"code analysis completed!")
        print(f"Found {len(codes)} different types of potential issues")
        print(f"Results saved to {output_file}")
        print(f"="*60)
        
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
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())