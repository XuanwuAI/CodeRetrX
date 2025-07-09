from dotenv import load_dotenv
load_dotenv()
import json
from pathlib import Path
from coderetrx.impl.default import CodebaseFactory
from coderetrx.impl.default import TopicExtractor
from coderetrx.retrieval.code_recall import multi_strategy_code_filter, multi_strategy_code_mapping
import os
import asyncio
import unittest
from typing import Literal
from unittest.mock import patch, MagicMock
from coderetrx.utils.embedding import create_documents_embedding
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id, get_data_dir
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TEST_REPOS = ["https://github.com/apache/dubbo-admin.git"]


def prepare_codebase(repo_url: str, repo_path: Path):
    """Helper function to prepare codebase for testing"""
    database_path = get_data_dir() / "databases" / f"{get_repo_id(repo_url)}.json"
    # Create a test codebase
    clone_repo_if_not_exists(repo_url, str(repo_path))

    if database_path.exists():
        codebase = CodebaseFactory.from_json(
            json.load(open(database_path, "r", encoding="utf-8"))
        )
    else:
        codebase = CodebaseFactory.new(get_repo_id(repo_url), repo_path)
    with open(f"{repo_path}.json", "w") as f:
        json.dump(codebase.to_json(), f, indent=4)
    return codebase


class TestLLMCodeFilterTool(unittest.TestCase):
    """Test LLMCodeFilterTool functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.repo_url = TEST_REPOS[0]
        self.repo_path = get_data_dir() / "repos" / get_repo_id(self.repo_url)
        self.codebase = prepare_codebase(self.repo_url, self.repo_path)
        self.test_dir = "/"
        self.test_prompt = "Is the code snippet used for user authentication?"
        self.topic_extractor = TopicExtractor()
    
    async def recall_with_mode(self, mode: Literal["fast", "balance", "precise", "custom"]):
        """Helper method to run the tool with a specific mode"""
        result, llm_output = await multi_strategy_code_filter(
            codebase=self.codebase,
            subdirs_or_files=[self.test_dir],
            prompt=self.test_prompt,
            granularity="symbol_content",
            mode=mode,
            topic_extractor=self.topic_extractor,
        )
        return result
    
    def test_initialization(self):
        """Test initialization of test environment"""
        self.assertIsNotNone(self.codebase)
        self.assertIsNotNone(self.test_dir)
    
    def test_fast_mode(self):
        """Test LLMCodeFilterTool in fast mode"""
        result = asyncio.run(self.recall_with_mode("fast"))
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        logger.info(f"Fast mode results count: {len(result)}")
        if result:
            logger.info(f"Sample result: {result[0]}")
    
    def test_balance_mode(self):
        """Test LLMCodeFilterTool in balance mode"""
        result = asyncio.run(self.recall_with_mode("balance"))
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        logger.info(f"Balance mode results count: {len(result)}")
        if result:
            logger.info(f"Sample result: {result[0]}")
    
    def test_precise_mode(self):
        """Test LLMCodeFilterTool in precise mode"""
        result = asyncio.run(self.recall_with_mode("precise"))
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        logger.info(f"Precise mode results count: {len(result)}")
        if result:
            logger.info(f"Sample result: {result[0]}")


class TestLLMCodeMappingTool(unittest.TestCase):
    """Test multi_strategy_code_mapping functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.repo_url = TEST_REPOS[0]
        self.repo_path = get_data_dir() / "repos" / get_repo_id(self.repo_url)
        self.codebase = prepare_codebase(self.repo_url, self.repo_path)
        self.test_dir = "/"
        self.test_prompt = "Extract the function call sink that may result in arbitrary code execution"
    
    async def recall_with_mode(self, mode: Literal["fast", "balance", "precise", "custom"]):
        """Helper method to run the tool with a specific mode"""
        result, llm_output = await multi_strategy_code_mapping(
            codebase=self.codebase,
            subdirs_or_files=[self.test_dir],
            prompt=self.test_prompt,
            granularity="symbol_content",
            mode=mode,
        )
        return result
    
    def test_initialization(self):
        """Test initialization of test environment"""
        self.assertIsNotNone(self.codebase)
        self.assertIsNotNone(self.test_dir)
    
    def test_fast_mode(self):
        """Test LLMCodeMappingTool in fast mode"""
        result = asyncio.run(self.recall_with_mode("fast"))
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        logger.info(f"Fast mode results count: {len(result)}")
        if result:
            logger.info(f"Sample result: {result[0]}")
    
    def test_balance_mode(self):
        """Test LLMCodeMappingTool in balance mode"""
        result = asyncio.run(self.recall_with_mode("balance"))
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        logger.info(f"Balance mode results count: {len(result)}")
        if result:
            logger.info(f"Sample result: {result[0]}")
    
    def test_precise_mode(self):
        """Test LLMCodeMappingTool in precise mode"""
        result = asyncio.run(self.recall_with_mode("precise"))
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        logger.info(f"Precise mode results count: {len(result)}")
        if result:
            logger.info(f"Sample result: {result[0]}")


# Run tests if specified
if __name__ == "__main__":
    # Run unittest tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
