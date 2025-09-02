import asyncio
import pytest
from coderetrx.tools.find_references import GetReferenceTool
from coderetrx.tools.view_file import ViewFileTool
from coderetrx.tools.find_file_by_name import FindFileByNameTool
from coderetrx.tools.keyword_search import KeywordSearchTool
from coderetrx.tools.list_dir import ListDirTool
import logging
logger =logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
TEST_REPO = "https://github.com/apache/flink.git"

class TestGetReferenceTool:
    def test(self):
        """Test finding references to a symbol"""

        logger.info("Testing GetReferenceTool...")
        tool = GetReferenceTool(TEST_REPO)
        result = asyncio.run(tool._run(symbol_name="upload"))
        logger.info(f"GetReferenceTool result: {result}")


class TestViewFileTool:
    def test(self):
        """Test viewing file content"""
        logger.info("Testing ViewFileTool...")
        tool = ViewFileTool(TEST_REPO)
        result = asyncio.run(tool._run(file_path="./README.md", start_line=0, end_line=10))
        logger.info(f"ViewFileTool result: {result}")


class TestFindFileByNameTool:
    def test(self):
        """Test finding files by name pattern"""
        logger.info("Testing FindFileByNameTool...")
        tool = FindFileByNameTool(TEST_REPO)
        results = asyncio.run(tool._run(dir_path="/", pattern="*.md"))
        logger.info(f"FindFileByNameTool result: {results}")


class TestKeywordSearchTool:
    def test(self):
        """Test keyword search functionality"""
        logger.info("Testing KeywordSearchTool...")
        tool = KeywordSearchTool(TEST_REPO)
        result = asyncio.run(
            tool._run(
                query="README",
                dir_path="/",
                case_insensitive=False,
                include_content=False,
            )
        )
        logger.info(f"KeywordSearchTool result: {result}")


class TestListDirTool:
    def test(self):
        """Test listing directory contents"""
        logger.info("Testing ListDirTool...")
        tool = ListDirTool(TEST_REPO)
        result = asyncio.run(tool._run(directory_path="."))
        logger.info(f"ListDirTool result: {result}")

def test_all_tools():
    """Test all tools"""
    tool_testers = [
        TestGetReferenceTool(),
        TestViewFileTool(),
        TestFindFileByNameTool(),
        TestKeywordSearchTool(),
        TestListDirTool(),
    ]
    for tester in tool_testers:
        tester.test()
if __name__ == "__main__":
    test_all_tools()