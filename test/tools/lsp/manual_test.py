"""
Manual test/demo script for LSP MCP tools.

Usage:
    uv run python -m test.tools.lsp.manual_test
"""

import asyncio
from pathlib import Path

from coderetrx.tools.lsp import ListSymbolTool, GetDefinitionTool, GetReferencesTool


async def demo_list_symbol():
    """Demo the list_symbol tool."""
    print("=" * 60)
    print("DEMO: List Symbol Tool")
    print("=" * 60)

    # Get the test fixture path
    fixture_path = Path(__file__).parent / "fixtures" / "python_sample"
    repo_url = f"file://{fixture_path}"

    tool = ListSymbolTool(repo_url)

    print(f"\nListing symbols in calculator.py (max_depth=2):\n")
    output = await tool.run(file_path="calculator.py", max_depth=2, repr_output=True)
    print(output)

    print(f"\n\nListing symbols in calculator.py (max_depth=1, top-level only):\n")
    output = await tool.run(file_path="calculator.py", max_depth=1, repr_output=True)
    print(output)


async def demo_get_definition():
    """Demo the get_definition tool."""
    print("\n\n")
    print("=" * 60)
    print("DEMO: Get Definition Tool")
    print("=" * 60)

    fixture_path = Path(__file__).parent / "fixtures" / "python_sample"
    repo_url = f"file://{fixture_path}"

    tool = GetDefinitionTool(repo_url)

    print(f"\nFinding definition of 'multiply' function (line 39, column 16 in calculator.py):\n")
    output = await tool.run(
        file_path="calculator.py",
        line=39,
        column=16,
        repr_output=True
    )
    print(output)


async def demo_get_references():
    """Demo the get_references tool."""
    print("\n\n")
    print("=" * 60)
    print("DEMO: Get References Tool")
    print("=" * 60)

    fixture_path = Path(__file__).parent / "fixtures" / "python_sample"
    repo_url = f"file://{fixture_path}"

    tool = GetReferencesTool(repo_url)

    print(f"\nFinding references to 'Calculator' class (line 4, column 7 in calculator.py):\n")
    output = await tool.run(
        file_path="calculator.py",
        line=4,
        column=7,
        include_declaration=True,
        include_context=True,
        repr_output=True
    )
    print(output)


async def main():
    """Run all demos."""
    await demo_list_symbol()
    await demo_get_definition()
    await demo_get_references()

    print("\n\n")
    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
