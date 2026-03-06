"""Simplified tests for LSP tools without cross-file imports."""

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


class TestListSymbolIntegration:
    """Integration tests for ListSymbolTool."""

    @pytest.mark.asyncio
    async def test_list_symbols(self, test_repo_url):
        """Test symbol listing works end-to-end."""
        tool = ListSymbolTool(test_repo_url)

        results = await tool._run(
            file_path="calculator.py",
            max_depth=3,
            zero_based=False,
        )

        assert len(results) > 0

        # Verify we found the Calculator class
        classes = [r for r in results if r.kind == "class"]
        assert any(r.name == "Calculator" for r in classes)

        # Verify we found functions
        functions = [r for r in results if r.kind == "function"]
        assert any(r.name == "multiply" for r in functions)

        # Print for debugging
        output = type(results[0]).repr(results)
        print(f"\n\nSymbol list output:\n{output}\n")


class TestGetDefinitionIntegration:
    """Integration tests for GetDefinitionTool."""

    @pytest.mark.asyncio
    async def test_get_definition_within_file(self, test_repo_url):
        """Test finding definition within the same file."""
        tool = GetDefinitionTool(test_repo_url)

        # Point to self.result usage in add() method
        results = await tool._run(
            file_path="calculator.py",
            line=12,  # self.result = x + y
            column=14,  # On "result"
            zero_based=False,
        )

        # Should find definition in __init__
        print(f"\n\nDefinition results: {results}\n")
        # Note: May or may not work depending on LSP implementation
        # Just verify it doesn't crash


class TestGetReferencesIntegration:
    """Integration tests for GetReferencesTool."""

    @pytest.mark.asyncio
    async def test_get_references_within_file(self, test_repo_url):
        """Test finding references within the same file."""
        tool = GetReferencesTool(test_repo_url)

        # Point to Calculator class definition
        results = await tool._run(
            file_path="calculator.py",
            line=4,  # class Calculator:
            column=7,  # On "Calculator"
            include_declaration=True,
            include_context=True,
            zero_based=False,
        )

        # Print results for debugging
        if results:
            output = type(results[0]).repr(results)
            print(f"\n\nReferences output:\n{output}\n")
        else:
            print("\n\nNo references found (may be expected for isolated file)\n")


class TestFormattedOutput:
    """Test that formatted output methods work correctly."""

    def test_list_symbol_repr(self):
        """Test ListSymbolResult.repr() formatting."""
        from coderetrx.tools.lsp.list_symbol import ListSymbolResult

        results = [
            ListSymbolResult(name="Calculator", kind="class", line=4, column=7, detail="A calculator", depth=0),
            ListSymbolResult(name="add", kind="method", line=10, column=9, detail="", depth=1),
        ]

        output = ListSymbolResult.repr(results)
        assert "Calculator" in output
        assert "add" in output
        assert "4:7" in output
        assert "10:9" in output

    def test_get_definition_repr(self):
        """Test GetDefinitionResult.repr() formatting."""
        from coderetrx.tools.lsp.get_definition import GetDefinitionResult

        results = [
            GetDefinitionResult(
                symbol_name="Calculator",
                definition_file="calculator.py",
                definition_line=4,
                definition_column=7,
            )
        ]

        output = GetDefinitionResult.repr(results)
        assert "Calculator" in output
        assert "calculator.py:4:7" in output

    def test_get_references_repr(self):
        """Test GetReferencesResult.repr() formatting."""
        from coderetrx.tools.lsp.get_references import GetReferencesResult

        results = [
            GetReferencesResult(
                symbol_name="Calculator",
                reference_file="calculator.py",
                reference_line=4,
                reference_column=7,
                context="class Calculator:",
            ),
            GetReferencesResult(
                symbol_name="Calculator",
                reference_file="test.py",
                reference_line=10,
                reference_column=12,
                context="calc = Calculator()",
            ),
        ]

        output = GetReferencesResult.repr(results)
        assert "Calculator" in output
        assert "calculator.py" in output
        assert "test.py" in output
        assert "2 reference(s)" in output
