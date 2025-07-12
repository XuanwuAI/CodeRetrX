from dotenv import load_dotenv
load_dotenv()

# Disable LLM cache before importing any modules that might set it
import os
os.environ["DISABLE_LLM_CACHE"] = "true"

from typing import Literal
import unittest
import asyncio
import json
import time
from pathlib import Path
from coderetrx.impl.default import CodebaseFactory, TopicExtractor
from coderetrx.retrieval.code_recall import multi_strategy_code_filter, RecallStrategy
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from coderetrx.utils.path import get_cache_dir, get_data_dir 
import logging
import chromadb
from coderetrx.utils import embedding
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

TEST_REPO = "https://github.com/PaddlePaddle/PaddleX.git"

def prepare_codebase(repo_url: str, repo_path: Path):
    """Prepare codebase for testing"""
    database_path = get_data_dir() / "databases" / f"{get_repo_id(repo_url)}.json"
    clone_repo_if_not_exists(repo_url, str(repo_path))

    if database_path.exists():
        codebase = CodebaseFactory.from_json(json.load(open(database_path, "r", encoding="utf-8")))
    else:
        codebase = CodebaseFactory.new(get_repo_id(repo_url), repo_path)
    with open(f"{repo_path}.json", "w") as f:
        json.dump(codebase.to_json(), f, indent=4)
    return codebase

def setup_embedding_environment():
    """Setup embedding environment variables"""
    os.environ["KEYWORD_EMBEDDING"] = "true"
    os.environ["SYMBOL_NAME_EMBEDDING"] = "true"
    os.environ["SYMBOL_CONTENT_EMBEDDING"] = "true"

def setup_persistent_chromadb():
    """Setup persistent ChromaDB for the entire test session"""
    cache_root = get_cache_dir() 
    chroma_dir = cache_root / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # Use persistent ChromaDB
    embedding.chromadb_client = chromadb.PersistentClient(path=str(chroma_dir))
    return chroma_dir

def generate_prompts(limit=10):
    """Load test prompts from feature_outline_refiner JSON file"""
    all_prompts = []
    
    # Add the specific authentication prompt first
    all_prompts.append("Is the code snippet used for user authentication?")
    
    feature_file = "feature_outline_refiner_0_6d53965443.json"
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
    return all_prompts[:limit]

class TestAdaptiveVsPreciseMode(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.repo_url = TEST_REPO
        self.repo_path = get_data_dir() / "repos" / get_repo_id(self.repo_url)
        
        # Setup embedding environment
        setup_embedding_environment()
        
        # Setup persistent ChromaDB
        self.chroma_dir = setup_persistent_chromadb()
        
        # Prepare codebase
        self.codebase = prepare_codebase(self.repo_url, self.repo_path)
        self.assertIsNotNone(self.codebase)
        self.assertGreater(len(self.codebase.symbols), 0)
        
        # Initialize similarity searchers once (will be cached)
        print("Initializing similarity searchers...")
        CodebaseFactory._initialize_similarity_searchers(self.codebase)
        print("Similarity searchers initialized")
        
        self.test_dir = "/"
        self.test_prompts = generate_prompts()
        self.topic_extractor = TopicExtractor()
    
    async def recall_with_mode(self, mode: Literal["fast", "balance", "precise", "custom"], prompt: str):
        """Run tool with specific mode"""
        if mode == "custom":
            return await multi_strategy_code_filter(
                codebase=self.codebase,
                subdirs_or_files=[self.test_dir],
                prompt=prompt,
                target_type="symbol_content",
                mode=mode,
                custom_strategies=[RecallStrategy.ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM],
                topic_extractor=self.topic_extractor,
            )
        return await multi_strategy_code_filter(
            codebase=self.codebase,
            subdirs_or_files=[self.test_dir],
            prompt=prompt,
            target_type="symbol_content",
            mode=mode,
        )
    
    async def test_multiple_prompts(self):
        """Test precise and adaptive mode functionality"""
        
        results = []
        
        test_modes = [
            {'name': 'adaptive', 'mode': 'custom', 'description': 'Adaptive'},
            {'name': 'balanced', 'mode': 'balance', 'description': 'Balanced'},
            {'name': 'precise', 'mode': 'precise', 'description': 'Precise'}
        ]
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"\nTesting prompt {i+1}/{len(self.test_prompts)}: {prompt[:50]}...")
            
            test_result = {'prompt': prompt}
            
            for mode_config in test_modes:
                mode_name = mode_config['name']
                mode_value = mode_config['mode']
                description = mode_config['description']
                
                try:
                    start_time = time.time()
                    result, llm_output = await self.recall_with_mode(mode_value, prompt)
                    
                    execution_time = time.time() - start_time
                    
                    test_result[mode_name] = {
                        'results_count': len(result),
                        'execution_time': execution_time,
                        'llm_calls': len(llm_output) if llm_output else 0
                    }
                    print(f"{description}: Found {len(result)} results in {execution_time:.2f}s")
                    
                except Exception as e:
                    print(f"{description} failed: {e}")
                    test_result[mode_name] = {
                        'results_count': 0,
                        'execution_time': 0,
                        'llm_calls': 0,
                        'error': str(e)
                    }
            
            # Show comparison for this prompt
            precise_data = test_result.get('precise', {})
            balanced_data = test_result.get('balanced', {})
            adaptive_data = test_result.get('adaptive', {})
            
            if ('error' not in precise_data and 'error' not in balanced_data and 'error' not in adaptive_data):
                precise_time = precise_data['execution_time']
                balanced_time = balanced_data['execution_time']
                adaptive_time = adaptive_data['execution_time']
                precise_count = precise_data['results_count']
                balanced_count = balanced_data['results_count']
                adaptive_count = adaptive_data['results_count']
                
                print(f"Comparison for this prompt:")
                print(f"   Results: Precise({precise_count}) vs Balanced({balanced_count}) vs Adaptive({adaptive_count})")
                print(f"   Time: Precise({precise_time:.2f}s) vs Balanced({balanced_time:.2f}s) vs Adaptive({adaptive_time:.2f}s)")
                
                if adaptive_time > 0:
                    speed_ratio_pa = precise_time / adaptive_time
                    print(f"   Speed ratio (Precise/Adaptive): {speed_ratio_pa:.2f}x")
                
                if balanced_time > 0:
                    speed_ratio_pb = precise_time / balanced_time
                    speed_ratio_ba = balanced_time / adaptive_time
                    print(f"   Speed ratio (Precise/Balanced): {speed_ratio_pb:.2f}x")
                    print(f"   Speed ratio (Balanced/Adaptive): {speed_ratio_ba:.2f}x")
                
                if precise_count == balanced_count == adaptive_count:
                    print(f"   All modes found same number of results")
                else:
                    print(f"   Result differences:")
                    if balanced_count != precise_count:
                        diff = balanced_count - precise_count
                        print(f"      Balanced vs Precise: {diff:+d}")
                    if adaptive_count != precise_count:
                        diff = adaptive_count - precise_count
                        print(f"      Adaptive vs Precise: {diff:+d}")
                    if adaptive_count != balanced_count:
                        diff = adaptive_count - balanced_count
                        print(f"      Adaptive vs Balanced: {diff:+d}")
            else:
                if 'error' in precise_data:
                    print(f"Precise mode failed: {precise_data.get('error', 'Unknown error')}")
                if 'error' in balanced_data:
                    print(f"Balanced mode failed: {balanced_data.get('error', 'Unknown error')}")
                if 'error' in adaptive_data:
                    print(f"Adaptive mode failed: {adaptive_data.get('error', 'Unknown error')}")
            
            print("-" * 60)
            results.append(test_result)
        
        # Generate summary statistics
        mode_summaries = {}
        for mode_config in test_modes:
            mode_name = mode_config['name']
            description = mode_config['description']
            
            successful = [r for r in results if 'error' not in r.get(mode_name, {})]
            failed = [r for r in results if 'error' in r.get(mode_name, {})]
            
            mode_summaries[mode_name] = {
                'successful': successful,
                'failed': failed,
                'description': description
            }
            
            print(f"\n=== {description.upper()} MODE SUMMARY ===")
            if successful:
                total_time = sum(r[mode_name]['execution_time'] for r in successful)
                avg_results = sum(r[mode_name]['results_count'] for r in successful) / len(successful)
                print(f"Successful: {len(successful)}")
                print(f"Average results: {avg_results:.1f}")
                print(f"Total time: {total_time:.2f}s")
            
            if failed:
                print(f"Failed: {len(failed)}")
                for test in failed:
                    print(f"- {test['prompt'][:30]}...: {test[mode_name]['error']}")
        
        # Comparison summary
        if (mode_summaries['precise']['successful'] and 
            mode_summaries['balanced']['successful'] and
            mode_summaries['adaptive']['successful']):
            print(f"\n=== COMPARISON ===")
            
            precise_successful = mode_summaries['precise']['successful']
            balanced_successful = mode_summaries['balanced']['successful']
            adaptive_successful = mode_summaries['adaptive']['successful']
            
            precise_avg_time = sum(r['precise']['execution_time'] for r in precise_successful) / len(precise_successful)
            balanced_avg_time = sum(r['balanced']['execution_time'] for r in balanced_successful) / len(balanced_successful)
            adaptive_avg_time = sum(r['adaptive']['execution_time'] for r in adaptive_successful) / len(adaptive_successful)
            precise_avg_results = sum(r['precise']['results_count'] for r in precise_successful) / len(precise_successful)
            balanced_avg_results = sum(r['balanced']['results_count'] for r in balanced_successful) / len(balanced_successful)
            adaptive_avg_results = sum(r['adaptive']['results_count'] for r in adaptive_successful) / len(adaptive_successful)
            
            print(f"Average time - Precise: {precise_avg_time:.2f}s, Balanced: {balanced_avg_time:.2f}s, Adaptive: {adaptive_avg_time:.2f}s")
            print(f"Average results - Precise: {precise_avg_results:.1f}, Balanced: {balanced_avg_results:.1f}, Adaptive: {adaptive_avg_results:.1f}")
            
            if adaptive_avg_time > 0 and balanced_avg_time > 0:
                speedup_pa = precise_avg_time / adaptive_avg_time
                speedup_pb = precise_avg_time / balanced_avg_time
                speedup_ba = balanced_avg_time / adaptive_avg_time
                print(f"Speed ratios:")
                print(f"  Precise/Adaptive: {speedup_pa:.2f}x")
                print(f"  Precise/Balanced: {speedup_pb:.2f}x")
                print(f"  Balanced/Adaptive: {speedup_ba:.2f}x")
        
        return results

async def main():
    """Run tests"""
    test_case = TestAdaptiveVsPreciseMode()
    test_case.setUp()
    results = await test_case.test_multiple_prompts()
    print(f"\nCompleted testing with {len(results)} prompts")

if __name__ == "__main__":
    asyncio.run(main())