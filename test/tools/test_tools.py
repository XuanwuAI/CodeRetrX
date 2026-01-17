import asyncio
import pytest
from coderetrx.tools.get_references import GetReferenceTool
from coderetrx.tools.view_file import ViewFileTool
from coderetrx.tools.find_file_by_name import FindFileByNameTool
from coderetrx.tools.keyword_search import KeywordSearchTool
from coderetrx.tools.list_dir import ListDirTool
from coderetrx.tools.codeql_query import CodeQLQueryTool
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
TEST_REPO = "https://github.com/pallets/flask.git"

class TestGetReferenceTool:
    def test(self):
        """Test finding references to a symbol"""
        logger.info("Testing GetReferenceTool...")
        tool = GetReferenceTool(TEST_REPO)
        result = asyncio.run(tool._run(symbol_name="upload"))
        logger.info(f"GetReferenceTool result: {result}")
        assert isinstance(result, (list, dict)), "Result should be a list or dict"


class TestViewFileTool:
    def test(self):
        """Test viewing file content"""
        logger.info("Testing ViewFileTool...")
        tool = ViewFileTool(TEST_REPO)
        result = asyncio.run(tool._run(file_path="./README.md", start_line=0, end_line=10))
        logger.info(f"ViewFileTool result: {result}")
        assert isinstance(result, str), "Result should be a string (file content)"
        assert len(result) > 0, "File content should not be empty"


class TestFindFileByNameTool:
    def test(self):
        """Test finding files by name pattern"""
        logger.info("Testing FindFileByNameTool...")
        tool = FindFileByNameTool(TEST_REPO)
        results = asyncio.run(tool._run(dir_path="/", pattern="*.md"))
        logger.info(f"FindFileByNameTool result: {results}")
        assert isinstance(results, list), "Result should be a list of file paths"


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
        assert isinstance(result, list), "Result should be a list of matches"


class TestListDirTool:
    def test(self):
        """Test listing directory contents"""
        logger.info("Testing ListDirTool...")
        tool = ListDirTool(TEST_REPO)
        result = asyncio.run(tool._run(directory_path="."))
        logger.info(f"ListDirTool result: {result}")
        assert isinstance(result, list), "Result should be a list of directory entries"


class TestCodeQLQueryTool:
    def test(self):
        """Test CodeQL query execution"""
        logger.info("Testing CodeQLQueryTool...")
        tool = CodeQLQueryTool(TEST_REPO)
        query = "import python\nfrom Function f\nselect f.getName()"
        result = asyncio.run(tool._run(query=query, language="python"))
        logger.info(f"CodeQLQueryTool result: {result}")
        assert isinstance(result, list), "Result should be a list"


def test_all_tools():
    """Test all tools"""
    tool_testers = [
        TestGetReferenceTool(),
        TestViewFileTool(),
        TestFindFileByNameTool(),
        TestKeywordSearchTool(),
        TestListDirTool(),
        TestCodeQLQueryTool(),
    ]
    for tester in tool_testers:
        tester.test()

class TestToolsSettings:
    def test_default_disabled_tools(self):
        """Test that codeql_query is disabled by default"""
        from coderetrx.tools.settings import Settings
        settings = Settings()
        assert "codeql_query" in settings.disabled_tools

    def test_tool_classes_excludes_disabled(self):
        """Test that tool_classes excludes disabled tools"""
        from coderetrx.tools import tool_classes
        tool_names = [cls.name for cls in tool_classes]
        assert "codeql_query" not in tool_names

    def test_env_override_disabled_tools(self):
        """Test that env var can override disabled_tools"""
        import os
        os.environ["CODERETRX_DISABLED_TOOLS"] = "[]"
        from coderetrx.tools.settings import Settings
        settings = Settings()
        assert settings.disabled_tools == []
        del os.environ["CODERETRX_DISABLED_TOOLS"]


if __name__ == "__main__":
    test_all_tools()