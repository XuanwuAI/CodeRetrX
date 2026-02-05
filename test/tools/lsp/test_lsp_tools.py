"""Tests for LSP MCP tools."""

import os
import pytest
from pathlib import Path

from coderetrx.tools.lsp import ListSymbolTool, GetDefinitionTool, GetReferencesTool


@pytest.fixture
def test_repo_path():
    """Get the path to test fixtures."""
    return str(Path(__file__).parent / "fixtures" / "python_sample")


@pytest.fixture
def test_repo_url(test_repo_path):
    """Mock repository URL using file:// scheme."""
    return f"file://{test_repo_path}"


class TestListSymbolTool:
    """Tests for ListSymbolTool."""

    @pytest.mark.asyncio
    async def test_list_symbols_basic(self, test_repo_url):
        """Test basic symbol listing."""
        tool = ListSymbolTool(test_repo_url)

        # List symbols in calculator.py
        results = await tool._run(
            file_path="calculator.py",
            max_depth=2,
            zero_based=False,
        )

        # Should find Calculator class and functions
        assert len(results) > 0

        # Check for Calculator class
        class_symbols = [r for r in results if r.name == "Calculator" and r.kind == "class"]
        assert len(class_symbols) == 1
        assert class_symbols[0].depth == 0

        # Check for methods (should be nested under Calculator)
        method_symbols = [r for r in results if r.kind == "method"]
        assert len(method_symbols) > 0

        # Check for top-level functions
        function_symbols = [
            r for r in results if r.kind == "function" and r.depth == 0
        ]
        assert len(function_symbols) >= 3  # multiply, divide, main

    @pytest.mark.asyncio
    async def test_list_symbols_depth_limit(self, test_repo_url):
        """Test depth limiting."""
        tool = ListSymbolTool(test_repo_url)

        # With depth=1, should only get top-level symbols
        results = await tool._run(
            file_path="calculator.py",
            max_depth=1,
            zero_based=False,
        )

        # All results should have depth 0
        assert all(r.depth == 0 for r in results)

    @pytest.mark.asyncio
    async def test_list_symbols_zero_based(self, test_repo_url):
        """Test zero-based indexing."""
        tool = ListSymbolTool(test_repo_url)

        # Test with 1-based (default)
        results_1based = await tool._run(
            file_path="calculator.py",
            max_depth=1,
            zero_based=False,
        )

        # Test with 0-based
        results_0based = await tool._run(
            file_path="calculator.py",
            max_depth=1,
            zero_based=True,
        )

        # Line numbers should differ by 1
        assert len(results_1based) == len(results_0based)
        if results_1based:
            assert results_1based[0].line == results_0based[0].line + 1

    @pytest.mark.asyncio
    async def test_list_symbols_nonexistent_file(self, test_repo_url):
        """Test with non-existent file."""
        tool = ListSymbolTool(test_repo_url)

        results = await tool._run(
            file_path="nonexistent.py",
            max_depth=3,
            zero_based=False,
        )

        assert results == []

    def test_repr_output(self):
        """Test formatted output."""
        from coderetrx.tools.lsp.list_symbol import ListSymbolResult

        results = [
            ListSymbolResult(
                name="MyClass",
                kind="class",
                line=5,
                column=1,
                detail="",
                depth=0,
            ),
            ListSymbolResult(
                name="my_method",
                kind="method",
                line=8,
                column=5,
                detail="",
                depth=1,
            ),
        ]

        output = ListSymbolResult.repr(results)
        assert "MyClass" in output
        assert "my_method" in output
        assert "5:1" in output
        assert "8:5" in output


class TestGetDefinitionTool:
    """Tests for GetDefinitionTool."""

    @pytest.mark.asyncio
    async def test_get_definition_class(self, test_repo_url):
        """Test finding class definition."""
        tool = GetDefinitionTool(test_repo_url)

        # Point to Calculator usage in test_calc.py
        results = await tool._run(
            file_path="test_calc.py",
            line=8,  # calc = Calculator()
            column=12,  # On "Calculator"
            zero_based=False,
        )

        # Should find definition in calculator.py
        assert len(results) >= 1
        assert results[0].symbol_name == "Calculator"
        assert "calculator.py" in results[0].definition_file

    @pytest.mark.asyncio
    async def test_get_definition_function(self, test_repo_url):
        """Test finding function definition."""
        tool = GetDefinitionTool(test_repo_url)

        # Point to multiply usage in test_calc.py
        results = await tool._run(
            file_path="test_calc.py",
            line=18,  # product = multiply(5, 6)
            column=16,  # On "multiply"
            zero_based=False,
        )

        # Should find definition
        assert len(results) >= 1
        assert results[0].symbol_name == "multiply"

    @pytest.mark.asyncio
    async def test_get_definition_zero_based(self, test_repo_url):
        """Test zero-based indexing."""
        tool = GetDefinitionTool(test_repo_url)

        # Test with 0-based indexing
        results = await tool._run(
            file_path="test_calc.py",
            line=7,  # 0-based: calc = Calculator()
            column=11,  # 0-based
            zero_based=True,
        )

        if results:
            # Line numbers in output should also be 0-based
            assert results[0].definition_line >= 0

    def test_repr_output(self):
        """Test formatted output."""
        from coderetrx.tools.lsp.get_definition import GetDefinitionResult

        results = [
            GetDefinitionResult(
                symbol_name="MyClass",
                definition_file="src/module.py",
                definition_line=10,
                definition_column=7,
            )
        ]

        output = GetDefinitionResult.repr(results)
        assert "MyClass" in output
        assert "src/module.py:10:7" in output


class TestGetReferencesTool:
    """Tests for GetReferencesTool."""

    @pytest.mark.asyncio
    async def test_get_references_class(self, test_repo_url):
        """Test finding class references."""
        tool = GetReferencesTool(test_repo_url)

        # Point to Calculator definition
        results = await tool._run(
            file_path="calculator.py",
            line=4,  # class Calculator:
            column=7,  # On "Calculator"
            include_declaration=True,
            include_context=True,
            zero_based=False,
        )

        # Should find references in test_calc.py
        assert len(results) >= 1
        ref_files = [r.reference_file for r in results]
        # Should include references from test_calc.py
        assert any("test_calc.py" in f for f in ref_files)

    @pytest.mark.asyncio
    async def test_get_references_function(self, test_repo_url):
        """Test finding function references."""
        tool = GetReferencesTool(test_repo_url)

        # Point to multiply definition
        results = await tool._run(
            file_path="calculator.py",
            line=22,  # def multiply
            column=5,
            include_declaration=True,
            include_context=True,
            zero_based=False,
        )

        # Should find reference in test_calc.py
        assert len(results) >= 1
        assert results[0].symbol_name == "multiply"

    @pytest.mark.asyncio
    async def test_get_references_with_context(self, test_repo_url):
        """Test context extraction."""
        tool = GetReferencesTool(test_repo_url)

        results = await tool._run(
            file_path="calculator.py",
            line=22,
            column=5,
            include_declaration=True,
            include_context=True,
            zero_based=False,
        )

        # Should have context for at least some results
        if results:
            # At least one result should have context
            assert any(r.context for r in results)

    @pytest.mark.asyncio
    async def test_get_references_without_context(self, test_repo_url):
        """Test without context extraction."""
        tool = GetReferencesTool(test_repo_url)

        results = await tool._run(
            file_path="calculator.py",
            line=22,
            column=5,
            include_declaration=True,
            include_context=False,
            zero_based=False,
        )

        # Context should be empty
        assert all(r.context == "" for r in results)

    def test_repr_output(self):
        """Test formatted output."""
        from coderetrx.tools.lsp.get_references import GetReferencesResult

        results = [
            GetReferencesResult(
                symbol_name="my_func",
                reference_file="src/a.py",
                reference_line=10,
                reference_column=5,
                context="    result = my_func()",
            ),
            GetReferencesResult(
                symbol_name="my_func",
                reference_file="src/b.py",
                reference_line=20,
                reference_column=8,
                context="    x = my_func(42)",
            ),
        ]

        output = GetReferencesResult.repr(results)
        assert "my_func" in output
        assert "src/a.py" in output
        assert "src/b.py" in output
        assert "10:5" in output
        assert "20:8" in output
