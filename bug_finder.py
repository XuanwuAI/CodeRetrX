from typing import Literal

from dotenv import load_dotenv

from codelib.retrieval import LLMCallMode

load_dotenv()

import os

import asyncio
import json
from pathlib import Path
from codelib.impl.default import CodebaseFactory, TopicExtractor
from codelib.retrieval.code_recall import multi_strategy_code_filter, RecallStrategy
from codelib.utils.git import clone_repo_if_not_exists, get_repo_id, get_data_dir
import logging
import chromadb
from codelib.utils import embedding
import datetime
import argparse
import sys

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class BugFinder:
    def __init__(self, repo_url: str, mode: Literal["fast", "balance", "precise", "custom"] = "balance", use_function_call: bool = False):
        """
        Initialize the bug finder with a repository URL
        
        Args:
            repo_url: The repository URL to analyze
            mode: Analysis mode - "fast", "balance", "precise", or "adaptive"
                - fast: Uses filename filtering only (fastest)
                - balance: Uses keyword vector + filename filtering (balanced speed/accuracy)
                - precise: Uses full LLM processing (most accurate but slowest)
                - adaptive: Uses adaptive vector + LLM strategy (smart filtering)
            use_function_call: Whether to use function call mode (default: False)
        """
        self.repo_url = repo_url
        self.repo_path = get_data_dir() / "repos" / get_repo_id(repo_url)
        self.topic_extractor = TopicExtractor()
        self.mode: Literal["fast", "balance", "precise", "custom"] = self._validate_mode(mode)
        self.use_function_call = use_function_call
        self.setup_environment()

    def _validate_mode(self, mode: Literal["fast", "balance", "precise", "custom"]) -> Literal["fast", "balance", "precise", "custom"]:
        """Validate and return the mode parameter"""
        valid_modes = ["fast", "balance", "precise", "adaptive"]
        if mode not in valid_modes:
            logger.warning(f"Invalid mode '{mode}'. Valid modes are: {valid_modes}. Defaulting to 'balance'.")
            return "balance"
        return mode

    def setup_environment(self):
        """Setup the necessary environment for bug finding"""
        if os.environ.get("DISABLE_LLM_CACHE", "").lower() == "true":
            del os.environ["DISABLE_LLM_CACHE"]            
        
        os.environ["KEYWORD_EMBEDDING"] = "true"
        os.environ["SYMBOL_NAME_EMBEDDING"] = "true"
        os.environ["SYMBOL_CONTENT_EMBEDDING"] = "true"
        
        cache_root = Path.home() / ".cache"
        chroma_dir = cache_root / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        embedding.chromadb_client = chromadb.PersistentClient(path=str(chroma_dir))
        
        llm_cache_dir = cache_root / "llm"
        llm_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_mode_description(self) -> str:
        """Get description for the current mode"""
        descriptions = {
            "fast": "Uses filename filtering only (fastest, least accurate)",
            "balance": "Uses keyword vector + filename filtering (balanced speed/accuracy)",
            "precise": "Uses full LLM processing (most accurate but slowest)",
            "adaptive": "Uses adaptive vector + LLM strategy (smart filtering with early exit)"
        }
        return descriptions.get(self.mode, f"Unknown mode: {self.mode}")

    def generate_prompts(self, limit: int = 10) -> list[str]:
        """Load test prompts from a feature_outline_refiner JSON file"""
        all_prompts = []
        
        # Get the absolute path of the feature file
        feature_file = Path(__file__).parent / "feature_outline_refiner_0_6d53965443.json"
        
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
        """Prepare the codebase for analysis"""
        database_path = get_data_dir() / "databases" / f"{get_repo_id(self.repo_url)}.json"
        clone_repo_if_not_exists(self.repo_url, str(self.repo_path))

        if database_path.exists():
            with open(database_path, "r", encoding="utf-8") as f:
                codebase = CodebaseFactory.from_json(json.load(f))
        else:
            codebase = CodebaseFactory.new(get_repo_id(self.repo_url), self.repo_path)
            
        with open(f"{self.repo_path}.json", "w") as f:
            json.dump(codebase.to_json(), f, indent=4)
            
        return codebase

    async def find_bugs(self, subdirs: list[str] = ["/"]) -> list[dict]:
        """Find potential bugs in the codebase using the specified mode"""
        codebase = self.prepare_codebase()
        
        bug_prompts = self.generate_prompts(parse_arguments().limit)
        all_bugs = []
        
        # Determine function call mode based on the use_function_call parameter
        llm_call_mode: LLMCallMode = "function_call" if self.use_function_call else "traditional"
        
        print(f"\nUsing mode: {self.mode}")
        print(f"Mode description: {self._get_mode_description()}")
        print(f"Function call mode: {llm_call_mode}")
        
        for prompt in bug_prompts:
            print(f"\n{'='*80}")
            print(f"Searching for bugs with prompt: {prompt}")
            print(f"Mode: {self.mode}")
            print(f"{'='*80}")
            logger.info(f"Searching for bugs with prompt: {prompt} (mode: {self.mode})")
            try:
                if self.mode == "adaptive":
                    # Use custom mode with adaptive strategies
                    result, llm_output = await multi_strategy_code_filter(
                        codebase=codebase,
                        subdirs_or_files=subdirs,
                        prompt=prompt,
                        granularity="symbol_content",
                        mode="custom",
                        custom_strategies=[
                            RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM,
                            RecallStrategy.ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM,
                        ],
                        topic_extractor=self.topic_extractor,
                        llm_call_mode=llm_call_mode
                    )
                else:
                    # Use built-in modes: fast, balance, precise
                    result, llm_output = await multi_strategy_code_filter(
                        codebase=codebase,
                        subdirs_or_files=subdirs,
                        prompt=prompt,
                        granularity="symbol_content",
                        mode=self.mode,
                        topic_extractor=self.topic_extractor,
                        llm_call_mode=llm_call_mode
                    )
                
                if result:
                    filtered_llm_output = None
                    if llm_output:
                        filtered_llm_output = [r for r in llm_output if hasattr(r, 'result') and r.result]
                    
                    bug_info = {
                        "bug_type": prompt,
                        "locations": result,
                        "llm_analysis": filtered_llm_output
                    }
                    all_bugs.append(bug_info)
                    
                    print(f"\nFound {len(result)} potential issues:")
                    print(f"Bug Type: {prompt}")
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
            
            print(f"{'='*80}\n")
                
        return all_bugs

    def save_results(self, bugs: list[dict], output_file: str = "bug_report.json"):
        """Save the bug finding results to a JSON file"""
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
        
        serializable_bugs = make_serializable(bugs)
        
        report = {
            "repository": self.repo_url,
            "timestamp": str(datetime.datetime.now()),
            "bugs": serializable_bugs
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Bug report saved to {output_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Bug Finder - Analyze code repositories for potential bugs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bug_finder.py --repo https://github.com/user/repo.git
  python bug_finder.py --mode fast
  python bug_finder.py --repo https://github.com/user/repo.git --mode precise
  python bug_finder.py --mode adaptive --output my_report.json
  python bug_finder.py --repo https://github.com/user/repo.git --subdirs src/ lib/
  python bug_finder.py --limit 5 --mode balance
  python bug_finder.py --use_function_call --mode precise

Available modes:
  fast      - Uses filename filtering only (fastest, least accurate)
  balance   - Uses keyword vector + filename filtering (balanced speed/accuracy) [DEFAULT]
  precise   - Uses full LLM processing (most accurate but slowest)
  adaptive  - Uses adaptive vector + LLM strategy (smart filtering with early exit)

Function call modes:
  traditional     - Traditional mode (default)
  function_calling - Use function call mode (enabled with --use_function_call)
        """
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default="https://github.com/PaddlePaddle/PaddleX.git",
        help="Repository URL to analyze (default: PaddleX)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["fast", "balance", "precise", "adaptive"],
        default="balance",
        help="Analysis mode (default: balance)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file name (default: bug_report_{mode}.json)"
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
    
    return parser.parse_args()

async def main():
    """Main function with command line argument support"""
    args = parse_arguments()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        logger.setLevel(logging.INFO)
    
    print(f"Bug Finder - Repository Analysis")
    print(f"Repository: {args.repo}")
    print(f"Mode: {args.mode}")
    print(f"Function Call: {args.use_function_call}")
    print(f"Subdirectories: {args.subdirs}")
    if args.output:
        print(f"Output file: {args.output}")
    
    print("-" * 60)
    
    try:
        # Run the bug finder
        bug_finder = BugFinder(args.repo, mode=args.mode, use_function_call=args.use_function_call)
        
        bugs = await bug_finder.find_bugs(subdirs=args.subdirs)
        
        output_file = args.output or f"bug_report_{args.mode}.json"
        bug_finder.save_results(bugs, output_file)
        
        print(f"\n" + "="*60)
        print(f"Bug analysis completed!")
        print(f"Found {len(bugs)} different types of potential issues")
        print(f"Results saved to {output_file}")
        print(f"="*60)
        
        # Display summary
        if bugs:
            print(f"\nSummary of issues found:")
            for i, bug in enumerate(bugs, 1):
                bug_type = bug['bug_type']
                locations = bug['locations']
                print(f"  {i}. {bug_type}: {len(locations)} locations")
        else:
            print(f"\nNo potential issues found.")
            
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