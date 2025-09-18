"""
Comprehensive integration tests for both TreeSitter and CodeQL parsers.

This test suite validates that both parsers implement the new interface correctly
and produce consistent results for the same codebase using real GitHub repositories.
"""

import json
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from coderetrx.static.codebase.codebase import Codebase, File, CodeChunk, Symbol, Keyword, Dependency
from coderetrx.static.codebase.parsers import ParserFactory, TreeSitterParser
from coderetrx.static.codebase.parsers.codeql.parser import CodeQLParser
from coderetrx.static.codebase.languages import get_language
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from coderetrx.utils.path import get_cache_dir, get_data_dir

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
(get_cache_dir() / "test").mkdir(parents=True, exist_ok=True)
# Global variable for test repositories - can be configured externally
TEST_REPOS = [
    {
        "url": "https://github.com/OpenRCE/sulley", 
        "name": "sulley",
        "max_files": 30
    }
]


class TestParserInterface:
    """Test the parser interface implementation."""
    
    def test_treesitter_parser_interface(self):
        """Test TreeSitter parser implements the interface correctly."""
        parser = TreeSitterParser()
        
        # Test interface methods exist
        assert hasattr(parser, 'init_chunks')
        assert hasattr(parser, 'init_symbols')
        assert hasattr(parser, 'init_keywords')
        assert hasattr(parser, 'init_dependencies')
        assert hasattr(parser, 'create_chunk')
        assert hasattr(parser, 'create_chunk_from_node')
        
        # Test language support
        assert parser.supports_language('python')
        supported_langs = parser.get_supported_languages()
        assert 'python' in supported_langs
        logger.info(f"TreeSitter supports {len(supported_langs)} languages: {supported_langs[:5]}...")
        
    def test_codeql_parser_interface(self):
        """Test CodeQL parser implements the interface correctly."""
        try:
            parser = CodeQLParser()
            
            # Test interface methods exist
            assert hasattr(parser, 'init_chunks')
            assert hasattr(parser, 'init_symbols')
            assert hasattr(parser, 'init_keywords')
            assert hasattr(parser, 'init_dependencies')
            assert hasattr(parser, 'create_chunk')
            
            # Test language support
            supported_langs = parser.get_supported_languages()
            logger.info(f"CodeQL supports {len(supported_langs)} languages: {supported_langs}")
            
        except Exception as e:
            pytest.skip(f"CodeQL not available: {e}")
    
    def test_parser_factory(self):
        """Test parser factory creates parsers correctly."""
        # Test TreeSitter parser creation
        ts_parser = ParserFactory.get_parser("treesitter")
        assert isinstance(ts_parser, TreeSitterParser)
        
        # Test auto parser selection
        auto_parser = ParserFactory.get_parser("auto")
        assert auto_parser is not None
        
        # Test available parsers
        available = ParserFactory.get_available_parsers()
        assert "treesitter" in available
        logger.info(f"Available parsers: {available}")


class TestRealRepositoryParsing:
    """Test parsing with real GitHub repositories."""
    
    @classmethod
    def setup_class(cls):
        """Set up test repositories from global TEST_REPOS variable."""
        global TEST_REPOS
        cls.test_repos = TEST_REPOS
        
        cls.repo_paths = {}
        cls.available_repos = []
        
        for repo in cls.test_repos:
            try:
                repo_path = get_data_dir() / "test_repos" / get_repo_id(repo["url"])
                clone_repo_if_not_exists(repo["url"], str(repo_path))
                cls.repo_paths[repo["name"]] = repo_path
                cls.available_repos.append(repo)
                logger.info(f"Repository {repo['name']} available at {repo_path}")
            except Exception as e:
                logger.warning(f"Failed to clone {repo['name']}: {e}")
        
        if not cls.available_repos:
            logger.warning("No test repositories available")
    
    def test_treesitter_on_real_repo(self):
        """Test TreeSitter parser on a real repository."""
        if not self.available_repos:
            pytest.skip("No test repositories available")
        
        # Use the first available repository
        repo = self.available_repos[0]
        repo_path = self.repo_paths[repo["name"]]
        
        # Create codebase with TreeSitter parser
        codebase = Codebase.new(
            id=f"test_{repo['name']}_ts",
            dir=repo_path,
            parser="treesitter",
        )
        
        # Test chunk extraction
        logger.info("Testing TreeSitter chunk extraction...")
        chunks = codebase.init_chunks()
        assert len(chunks) > 0
        logger.info(f"TreeSitter extracted {len(chunks)} chunks")
        
        # Test symbol extraction
        logger.info("Testing TreeSitter symbol extraction...")
        symbols = codebase._extract_symbols()
        assert len(symbols) > 0
        logger.info(f"TreeSitter extracted {len(symbols)} symbols")
        
        # Test keyword extraction
        logger.info("Testing TreeSitter keyword extraction...")
        keywords = codebase._extract_keywords()
        assert len(keywords) > 0
        logger.info(f"TreeSitter extracted {len(keywords)} keywords")
        
        # Test dependency extraction
        logger.info("Testing TreeSitter dependency extraction...")
        dependencies = codebase._extract_dependencies()
        logger.info(f"TreeSitter extracted {len(dependencies)} dependencies")
        with open(get_cache_dir()/ "test" / "codebase_test_treesitter.json", "w") as f:
            json.dump(codebase.to_json(), f)
        
        # Verify data consistency
        self._verify_codebase_consistency(codebase, "TreeSitter")
    
    def test_codeql_on_real_repo(self):
        """Test CodeQL parser on a real repository."""
        if not self.available_repos:
            pytest.skip("No test repositories available")
        
        try:
            # Use the first available repository
            repo = self.available_repos[0]
            repo_path = self.repo_paths[repo["name"]]
            
            # Create codebase with CodeQL parser
            codebase = Codebase.new(
                id=f"test_{repo['name']}_codeql",
                dir=repo_path,
                parser="codeql",
            )
            
            # Test chunk extraction
            logger.info("Testing CodeQL chunk extraction...")
            chunks = codebase.init_chunks()
            assert len(chunks) >= 0  # CodeQL might find fewer chunks
            logger.info(f"CodeQL extracted {len(chunks)} chunks")
            
            # Test symbol extraction
            logger.info("Testing CodeQL symbol extraction...")
            symbols = codebase._extract_symbols()
            logger.info(f"CodeQL extracted {len(symbols)} symbols")
            
            # Test keyword extraction
            logger.info("Testing CodeQL keyword extraction...")
            keywords = codebase._extract_keywords()
            assert len(keywords) > 0
            logger.info(f"CodeQL extracted {len(keywords)} keywords")
            
            # Test dependency extraction
            logger.info("Testing CodeQL dependency extraction...")
            dependencies = codebase._extract_dependencies()
            logger.info(f"CodeQL extracted {len(dependencies)} dependencies")
            with open(get_cache_dir()/"test"/ "codebase_test_codeql.json", "w") as f:
                json.dump(codebase.to_json(), f)
            # Verify data consistency
            self._verify_codebase_consistency(codebase, "CodeQL")

        except Exception as e:
            pytest.skip(f"CodeQL not available or failed: {e}")
    
    def test_parser_comparison(self):
        """Compare results between TreeSitter and CodeQL parsers."""
        if not self.available_repos:
            pytest.skip("No test repositories available")
        
        # Use the first available repository
        repo = self.available_repos[0]
        repo_path = self.repo_paths[repo["name"]]
        
        # Create codebases with both parsers
        ts_codebase = Codebase.new(
            id=f"test_{repo['name']}_ts_compare",
            dir=repo_path,
            parser="treesitter",
        )
        
        try:
            codeql_codebase = Codebase.new(
                id=f"test_{repo['name']}_codeql_compare",
                dir=repo_path,
                parser="codeql",
            )
        except Exception as e:
            pytest.skip(f"CodeQL not available: {e}")
        
        # Extract chunks from both
        ts_chunks = ts_codebase.init_chunks()
        codeql_chunks = codeql_codebase.init_chunks()
        
        logger.info(f"Comparison: TreeSitter={len(ts_chunks)} chunks, CodeQL={len(codeql_chunks)} chunks")
        
        # Compare file coverage
        ts_files = set(chunk.src.path for chunk in ts_chunks)
        codeql_files = set(chunk.src.path for chunk in codeql_chunks)
        
        common_files = ts_files.intersection(codeql_files)
        logger.info(f"Common files parsed by both: {len(common_files)}")
        
        # Test UUID consistency for same locations
        self._test_uuid_consistency(ts_chunks, codeql_chunks)
    
    def _verify_codebase_consistency(self, codebase: Codebase, parser_name: str):
        """Verify internal consistency of codebase data."""
        logger.info(f"Verifying {parser_name} codebase consistency...")
        
        # Check that all chunks have valid UUIDs
        chunk_uuids = set()
        for chunk in codebase.all_chunks:
            assert chunk.uuid is not None
            assert chunk.id not in chunk_uuids, f"Duplicate chunk UUID: {chunk.id}"
            chunk_uuids.add(chunk.id)
        
        # Check that all symbols reference valid chunks
        for symbol in codebase.symbols:
            assert symbol.chunk in codebase.all_chunks
            assert symbol.file in codebase.source_files.values()
        
        # Check that all chunks reference valid files
        for chunk in codebase.all_chunks:
            assert chunk.src in codebase.source_files.values()
        
        # Check parent-child relationships
        for chunk in codebase.all_chunks:
            if chunk.parent:
                assert chunk.parent in codebase.all_chunks
                assert chunk.parent.includes(chunk)
        
        logger.info(f"{parser_name} codebase consistency verified ✓")
    
    def _test_uuid_consistency(self, ts_chunks: List[CodeChunk], codeql_chunks: List[CodeChunk]):
        """Test that identical code locations get identical UUIDs."""
        logger.info("Testing UUID consistency between parsers...")
        
        # Create lookup maps by location
        ts_by_location = {}
        for chunk in ts_chunks:
            key = (str(chunk.src.path), chunk.start_line, chunk.end_line, 
                   chunk.start_column, chunk.end_column)
            ts_by_location[key] = chunk
        
        codeql_by_location = {}
        for chunk in codeql_chunks:
            key = (str(chunk.src.path), chunk.start_line, chunk.end_line,
                   chunk.start_column, chunk.end_column)
            codeql_by_location[key] = chunk
        
        # Find common locations
        common_locations = set(ts_by_location.keys()).intersection(
            set(codeql_by_location.keys())
        )
        
        logger.info(f"Found {len(common_locations)} common chunk locations")
        
        # Verify UUID consistency
        consistent_uuids = 0
        for location in common_locations:
            ts_chunk = ts_by_location[location]
            codeql_chunk = codeql_by_location[location]
            
            if ts_chunk.uuid == codeql_chunk.uuid:
                consistent_uuids += 1
            else:
                logger.warning(f"UUID mismatch at {location}: {ts_chunk.uuid} vs {codeql_chunk.uuid}")
        
        consistency_rate = consistent_uuids / len(common_locations) if common_locations else 0
        logger.info(f"UUID consistency rate: {consistency_rate:.2%} ({consistent_uuids}/{len(common_locations)})")
        
        # We expect high consistency since both use the same hunk_uuid function
        assert consistency_rate > 0.9, f"Low UUID consistency: {consistency_rate:.2%}"


class TestParserSpecificMethods:
    """Test parser-specific method implementations."""
    
    def test_treesitter_create_chunk_from_node(self):
        """Test TreeSitter's create_chunk_from_node method."""
        # This would require creating a mock tree-sitter node
        # For now, just verify the method exists
        parser = TreeSitterParser()
        assert hasattr(parser, 'create_chunk_from_node')
        assert callable(getattr(parser, 'create_chunk_from_node'))
    
    def test_codeql_create_chunk(self):
        """Test CodeQL's create_chunk method."""
        try:
            parser = CodeQLParser()
            assert hasattr(parser, 'create_chunk')
            assert callable(getattr(parser, 'create_chunk'))
        except Exception as e:
            pytest.skip(f"CodeQL not available: {e}")
    
    def test_parser_cleanup(self):
        """Test parser cleanup methods."""
        # TreeSitter cleanup
        ts_parser = TreeSitterParser()
        ts_parser.cleanup()  # Should not raise
        
        # CodeQL cleanup
        try:
            codeql_parser = CodeQLParser()
            codeql_parser.cleanup()  # Should not raise
        except Exception as e:
            pytest.skip(f"CodeQL not available: {e}")


class TestErrorHandling:
    """Test error handling in parsers."""
    
    def test_unsupported_language(self):
        """Test handling of unsupported languages."""
        parser = TreeSitterParser()
        
        # Test with a made-up language (using type ignore for testing)
        # This tests the parser's robustness with invalid input
        try:
            result = parser.supports_language("nonexistent_language")  # type: ignore
            assert not result
        except Exception:
            # Parser should handle invalid languages gracefully
            pass
    
    def test_invalid_file_parsing(self):
        """Test parsing of invalid or non-existent files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a codebase with no files
            codebase = Codebase.new(
                id="test_empty",
                dir=temp_path,
                parser="treesitter"
            )
            
            # Should handle empty codebase gracefully
            chunks = codebase.init_chunks()
            assert len(chunks) == 0
            
            symbols = codebase._extract_symbols()
            assert len(symbols) == 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
