import unittest
import asyncio
from coderetrx.retrieval.code_recall import _determine_strategy_by_llm
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_prompts(limit=10):
    all_prompts = []
    bug_name = []
    feature_file = "features/test.json"
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
                            bug_name.append(feature.get('name'))
                            if len(all_prompts) >= limit:
                                break
    except Exception as e:
        logger.warning(f"Failed to load feature file {feature_file}: {e}")
    return all_prompts[:limit], bug_name[:limit]

class TestDetermineStrategy(unittest.TestCase):
    def test_determine_strategy(self):
        async def run_test():
            res = generate_prompts(25)
            for idx in range(len(res[0])):
                bug_name = res[1][idx]
                prompt = res[0][idx]
                print(f"Bug Name: {bug_name}")
                strategy = await _determine_strategy_by_llm(prompt)
                print(f"Strategy: {strategy}")
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()