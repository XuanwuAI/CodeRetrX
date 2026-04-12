import asyncio
import pytest
from coderetrx.tools.get_references import GetReferenceTool
from coderetrx.tools.view_file import ViewFileTool
from coderetrx.tools.find_file_by_name import FindFileByNameTool
from coderetrx.tools.keyword_search import KeywordSearchTool
from coderetrx.tools.keyword_search import KeywordSearchResult
from coderetrx.tools.list_dir import ListDirTool
from coderetrx.tools.codeql_query import CodeQLQueryTool
from coderetrx.tools.llm_filter import LLMCodeFilterTool
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
        result = asyncio.run(tool._run(file_path="./README.md", show_line_numbers=True, start_line=0, end_line=10))
        logger.info(f"ViewFileTool result: {result}")
        assert isinstance(result, str), "Result should be a string (file content)"
        assert len(result) > 0, "File content should not be empty"

    def test_with_line_numbers(self):
        """Test viewing file with line numbers"""
        logger.info("Testing ViewFileTool with line numbers...")
        tool = ViewFileTool(TEST_REPO)
        result = asyncio.run(tool._run(
            file_path="./README.md",
            start_line=0,
            end_line=10,
            show_line_numbers=True
        ))
        logger.info(f"ViewFileTool result with line numbers: {result}")
        assert isinstance(result, str), "Result should be a string"
        assert " | " in result, "Result should contain line number separator"
        # Verify first line starts with 0 (0-indexed)
        assert result.startswith("0 | ") or result.strip().split("\n")[0].strip().startswith("0 | "), "First line should start with 0"

    def test_half_open_range_contract_uses_visible_lines(self, tmp_path):
        """view_file should use 0-based half-open ranges without a phantom EOF line."""
        sample = tmp_path / "sample.txt"
        sample.write_text("alpha\nbeta\n", encoding="utf-8")

        tool = object.__new__(ViewFileTool)
        tool.repo_path = tmp_path

        result = asyncio.run(
            tool._run(
                file_path="sample.txt",
                start_line=0,
                end_line=2,
                show_line_numbers=True,
            )
        )

        assert "0 | alpha" in result
        assert "1 | beta" in result
        assert "2 |" not in result
        assert "total 2 lines" in result
        assert "[start_line, end_line)" in result

    def test_invalid_range_reports_exclusive_upper_bound(self, tmp_path):
        """Invalid view_file ranges should report the exclusive upper bound clearly."""
        sample = tmp_path / "sample.txt"
        sample.write_text("alpha\nbeta\n", encoding="utf-8")

        tool = object.__new__(ViewFileTool)
        tool.repo_path = tmp_path

        result = asyncio.run(
            tool._run(
                file_path="sample.txt",
                start_line=0,
                end_line=3,
                show_line_numbers=True,
            )
        )

        assert "0 <= start_line <= end_line <= 2" in result
        assert "end_line is exclusive" in result

    def test_1000_line_range_is_allowed(self, tmp_path):
        """view_file should allow a range with exactly 1000 lines."""
        sample = tmp_path / "sample.txt"
        sample.write_text("\n".join(str(i) for i in range(1000)), encoding="utf-8")

        tool = object.__new__(ViewFileTool)
        tool.repo_path = tmp_path

        result = asyncio.run(
            tool._run(
                file_path="sample.txt",
                start_line=0,
                end_line=1000,
                show_line_numbers=False,
            )
        )

        assert "File is too large" not in result
        assert "999" in result


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

    def test_repr_shows_zero_based_line_zero(self):
        """KeywordSearchResult repr should keep line 0 visible for agent consumers."""
        result = KeywordSearchResult.repr(
            [
                KeywordSearchResult(
                    path="main.go",
                    start_line=0,
                    end_line=0,
                    content="package main",
                )
            ]
        )
        assert "Line 0 (0-based, inclusive)" in result


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

class TestLLMCodeFilterTool:
    def test(self):
        """Test LLM code filter functionality"""
        logger.info("Testing LLMCodeFilterTool...")
        tool = LLMCodeFilterTool("YCrypt")
        result = asyncio.run(
            tool._run(
                filter_prompt="functions that handle cryptographic operations",
                subdirs_or_files=[],
                return_content=True,
            )
        )
        logger.info(f"LLMCodeFilterTool result: {result}")
        assert isinstance(result, (list, str)), "Result should be a list or string"

def test_all_tools():
    """Test all tools"""
    tool_testers = [
        TestGetReferenceTool(),
        TestViewFileTool(),
        TestFindFileByNameTool(),
        TestKeywordSearchTool(),
        TestListDirTool(),
        TestCodeQLQueryTool(),
        TestLLMCodeFilterTool(),
    ]
    for tester in tool_testers:
        tester.test()

class TestToolsSettings:
    def test_available_tools_is_list(self):
        """Test that available_tools is a list"""
        from coderetrx.tools.settings import Settings
        settings = Settings()
        assert isinstance(settings.available_tools, list)

    def test_tool_classes_respects_available(self):
        """Test that tool_classes only includes tools in available_tools"""
        from coderetrx.tools.settings import Settings
        settings = Settings()
        from coderetrx.tools import tool_classes
        tool_names = [cls.name for cls in tool_classes]
        for name in tool_names:
            assert name in settings.available_tools


if __name__ == "__main__":
    test_all_tools()
