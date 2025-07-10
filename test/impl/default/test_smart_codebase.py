from dotenv import load_dotenv
load_dotenv()
import json
from pathlib import Path
from coderetrx.impl.default import CodebaseFactory
from coderetrx.impl.default import SmartCodebase
import os
import asyncio
import unittest
from coderetrx.utils.embedding import create_documents_embedding
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from coderetrx.utils.path import get_data_dir
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


class TestSmartCodebase(unittest.TestCase):
    """Test SmartCodebase functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.repo_url = TEST_REPOS[0]
        self.repo_path = get_data_dir() / "repos" / get_repo_id(self.repo_url)
    
    def test_codebase_initialization(self):
        """Test codebase initialization"""
        codebase = prepare_codebase(self.repo_url, self.repo_path)
        self.assertIsInstance(codebase, SmartCodebase)
        self.assertEqual(codebase.id, get_repo_id(self.repo_url))
    
    def test_keyword_extraction(self):
        """Test keyword extraction functionality"""
        os.environ["KEYWORD_EMBEDDING"] = "True"
        codebase = prepare_codebase(self.repo_url, self.repo_path)
        
        # Verify keywords were extracted
        self.assertGreater(len(codebase.keywords), 0)
        logger.info(f"Extracted {len(codebase.keywords)} keywords")
        logger.info(f"Sample keywords: {[k.content for k in codebase.keywords[:10]]}")
    
    def test_keyword_search(self):
        """Test keyword search functionality"""
        os.environ["KEYWORD_EMBEDDING"] = "True"
        codebase = prepare_codebase(self.repo_url, self.repo_path)
        
        try:
            # Generate embeddings
            test_query = "Is the code snippet used for user authentication?"
            logger.info(f"\nTesting keyword search with query: '{test_query}'")
            
            # Create a searcher
            results = asyncio.run(codebase.similarity_search(
                target_types=["keyword"], query=test_query
            ))
            
            logger.info("Search results:")
            logger.info(results)
            
            # Verify results
            self.assertIsNotNone(results)
            
        except Exception as e:
            self.fail(f"Error during keyword search test: {e}")
    
    def test_symbol_extraction(self):
        """Test symbol extraction functionality"""
        os.environ["SYMBOL_NAME_EMBEDDING"] = "True"
        codebase = prepare_codebase(self.repo_url, self.repo_path)
        
        # Verify symbols were extracted
        self.assertGreater(len(codebase.symbols), 0)
        logger.info(f"Sample symbols: {[k.name for k in codebase.symbols[:10]]}")
    
    def test_symbol_search(self):
        """Test symbol search functionality"""
        os.environ["SYMBOL_NAME_EMBEDDING"] = "True"
        codebase = prepare_codebase(self.repo_url, self.repo_path)
        
        try:
            # Test search with a sample query
            test_query = "Is the code snippet used for user authentication?"
            logger.info(f"\nTesting symbol search with query: '{test_query}'")
            
            # Create a searcher
            results = asyncio.run(codebase.similarity_search(
                target_types=["symbol_name"], query=test_query
            ))
            
            logger.info("Search results:")
            logger.info(results)
            
            # Verify results
            self.assertIsNotNone(results)
            
        except Exception as e:
            self.fail(f"Error during symbol search test: {e}")


# Run tests if specified
if __name__ == "__main__":
    # Run unittest tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)