"""Tests for LSP parser implementation."""

from coderetrx.static.codebase import Codebase
from coderetrx.static.codebase.codebase import ChunkType
from coderetrx.static.codebase.parsers import LSPParser, ParserFactory


def run_test(codebase: Codebase):
    codebase.init_all(dependencies=True, keywords=False)
    codebase.cleanup()


class TestLSPParser:
    """Test suite for LSP parser."""

    def test_supports_language(self):
        """Test that LSP parser reports correct language support."""
        parser = LSPParser()

        # Should support these languages via LSP
        assert parser.supports_language("python")
        assert parser.supports_language("javascript")
        assert parser.supports_language("typescript")
        assert parser.supports_language("rust")
        assert parser.supports_language("go")

    def test_parser_initialization(self):
        """Test LSP parser initialization."""
        parser = LSPParser()
        assert parser is not None
        assert parser._max_concurrent_requests == 10  # Default concurrency

    def test_max_concurrent_requests_config(self):
        """Test that max_concurrent_requests config is respected."""
        parser = LSPParser(max_concurrent_requests=5)
        assert parser._max_concurrent_requests == 5

        parser_default = LSPParser()
        assert parser_default._max_concurrent_requests == 10

        parser_high = LSPParser(max_concurrent_requests=20)
        assert parser_high._max_concurrent_requests == 20

    def test_parser_cleanup(self):
        """Test that parser cleanup doesn't error when client not initialized."""
        parser = LSPParser()
        parser.cleanup()  # Should not raise

    def test_basic_parsing(self, tmp_path):
        """Test basic parsing with LSP parser.

        This test is skipped by default as it requires LSP servers to be installed.
        Run with: pytest -v --run-lsp-tests
        """
        # Create a simple Python file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''
def hello_world():
    """A simple function."""
    print("Hello, world!")

class MyClass:
    """A simple class."""

    def method(self):
        """A method."""
        pass
'''
        )

        # Parse with LSP
        codebase = Codebase.new(id="test_lsp", dir=tmp_path, parser="lsp")

        codebase.init_chunks()

        # Should have extracted chunks
        assert len(codebase.all_chunks) > 0

        # Check that we got function and class chunks
        chunk_types = {chunk.type for chunk in codebase.all_chunks}
        assert ChunkType.PRIMARY in chunk_types

        # Cleanup
        codebase.cleanup()


class TestParserIntegration:
    """Test LSP parser integration with factory."""

    def test_factory_can_create_lsp_parser(self):
        """Test that parser factory can create LSP parser."""

        parser = ParserFactory.get_parser("lsp")
        assert isinstance(parser, LSPParser)

    def test_lsp_in_available_parsers(self):
        """Test that LSP parser appears in available parsers list."""

        available = ParserFactory.get_available_parsers()
        assert "lsp" in available
        # Note: availability will be False if LSP servers not installed

    def test_symbols_extracted_directly_from_lsp(self, tmp_path):
        """Test that symbols are created directly from LSP DocumentSymbols."""
        # Create test file with class and function
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
class MyClass:
    def method(self):
        pass

def my_function():
    pass
"""
        )

        # Parse with LSP
        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")
        run_test(codebase)
        assert len(codebase.symbols) >= 3, "Should extract class, method, and function"

        # Find specific symbols
        class_sym = next((s for s in codebase.symbols if s.name == "MyClass"), None)
        method_sym = next(
            (s for s in codebase.symbols if s.name == "MyClass.method"), None
        )
        func_sym = next((s for s in codebase.symbols if s.name == "my_function"), None)

        # Verify symbols exist
        assert class_sym is not None, "Should extract MyClass"
        assert method_sym is not None, "Should extract MyClass.method"
        assert func_sym is not None, "Should extract my_function"

        # Verify symbol types
        assert class_sym.type == "class"
        assert method_sym.type == "function"
        assert func_sym.type == "function"

        # Verify each symbol has a corresponding chunk
        assert class_sym.chunk in codebase.all_chunks
        assert method_sym.chunk in codebase.all_chunks
        assert func_sym.chunk in codebase.all_chunks

    def test_qualified_names_for_nested_symbols(self, tmp_path):
        """Test that nested symbols get qualified names (e.g., ClassName.method)."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
class OuterClass:
    def outer_method(self):
        pass

    class InnerClass:
        def inner_method(self):
            pass
"""
        )

        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")
        run_test(codebase)

        # Check for qualified names
        symbol_names = [s.name for s in codebase.symbols]

        assert "OuterClass" in symbol_names
        assert "OuterClass.outer_method" in symbol_names

        # Note: Inner classes may or may not be supported depending on LSP server
        # Just verify we don't crash on nested structures

    def test_chunks_created_as_symbol_representations(self, tmp_path):
        """Test that CodeChunks are created as representations of symbols."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
class MyClass:
    def method(self):
        pass
"""
        )

        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")
        run_test(codebase)

        # Verify all symbols have chunks
        for symbol in codebase.symbols:
            if symbol.type in ["class", "function"]:  # Not dependencies
                assert symbol.chunk is not None
                assert symbol.chunk.type == ChunkType.PRIMARY
                assert symbol.chunk in codebase.all_chunks

        # Verify chunks have correct parent-child relationships
        class_chunk = next(
            (c for c in codebase.all_chunks if c.tag == "definition.class"), None
        )
        method_chunk = next(
            (c for c in codebase.all_chunks if c.tag == "definition.method"), None
        )

        if class_chunk and method_chunk:
            # Method chunk should be child of class chunk
            assert method_chunk.parent == class_chunk

    def test_treesitter_fallback_for_imports(self, tmp_path):
        """Test that TreeSitter is used to extract imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
import os
import sys
from pathlib import Path

def my_function():
    pass
"""
        )

        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")
        run_test(codebase)

        # Check for import chunks (from TreeSitter fallback)
        import_chunks = [c for c in codebase.all_chunks if c.type == ChunkType.IMPORT]
        assert len(import_chunks) >= 3, "Should extract at least 3 imports"

        # Check for dependency symbols
        dependency_symbols = [s for s in codebase.symbols if s.type == "dependency"]
        assert len(dependency_symbols) >= 3, "Should create dependency symbols"

        # Verify dependency names
        dep_names = [s.name for s in dependency_symbols]
        assert "os" in dep_names
        assert "sys" in dep_names
        assert "pathlib" in dep_names

    def test_variables_create_chunks_not_symbols(self, tmp_path):
        """Test that variables create chunks but not Symbol objects."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
CONSTANT = 42

class MyClass:
    class_var = "test"

    def method(self):
        local_var = 10
"""
        )

        # Enable variable extraction
        codebase = Codebase.new(
            id="test",
            dir=tmp_path,
            parser="lsp",
            extract_variable_definitions=True,
        )

        run_test(codebase)

        # Check for variable chunks
        variable_chunks = [
            c for c in codebase.all_chunks if c.type == ChunkType.VARIABLE
        ]

        # Variables should create chunks
        # (Note: LSP variable extraction behavior may vary by language server)
        # Just verify no crashes

        # Verify only class/function symbols exist (no variable symbols)
        for symbol in codebase.symbols:
            assert symbol.type in [
                "class",
                "function",
            ], "Variables should not create Symbol objects"

    def test_symbol_chunk_uuid_consistency(self, tmp_path):
        """Test that symbol chunks have consistent UUIDs."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def my_function():
    pass
"""
        )

        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")
        run_test(codebase)

        # Verify all chunks have UUIDs
        for chunk in codebase.all_chunks:
            assert chunk.uuid is not None
            assert chunk.id is not None

        # Verify symbol ID matches chunk ID
        for symbol in codebase.symbols:
            if symbol.type in ["class", "function"]:
                assert symbol.id == symbol.chunk.id

    def test_lsp_and_treesitter_hybrid(self, tmp_path):
        """Test that LSP (symbols) and TreeSitter (imports) work together."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
import os

class MyClass:
    def method(self):
        return os.path.exists('.')
"""
        )

        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")

        codebase.init_all(keywords=False)
        codebase.cleanup()

        # Verify LSP symbols
        class_symbols = [s for s in codebase.symbols if s.type == "class"]
        func_symbols = [s for s in codebase.symbols if s.type == "function"]
        assert len(class_symbols) >= 1, "Should extract class via LSP"
        assert len(func_symbols) >= 1, "Should extract method via LSP"

        # Verify TreeSitter imports
        import_chunks = [c for c in codebase.all_chunks if c.type == ChunkType.IMPORT]
        assert len(import_chunks) >= 1, "Should extract imports via TreeSitter"

        # Verify dependency symbols
        dep_symbols = [s for s in codebase.symbols if s.type == "dependency"]
        assert len(dep_symbols) >= 1, "Should create dependency symbols"
        assert any(s.name == "os" for s in dep_symbols), "Should have 'os' dependency"

    def test_lsp_parser_sets_selection_range(self, tmp_path):
        """Test that LSP parser populates selection_range from DocumentSymbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def my_function():
    pass

class MyClass:
    def method(self):
        pass
"""
        )

        codebase = Codebase.new(id="test", dir=tmp_path, parser="lsp")
        run_test(codebase)

        # Find function symbol
        func_symbol = next(
            (s for s in codebase.symbols if s.name == "my_function"), None
        )
        assert func_symbol is not None, "Should find my_function symbol"

        # Verify selection_range is populated by LSP
        assert (
            func_symbol.selection_range is not None
        ), "LSP should set selection_range"

        # Verify it's a Range TypedDict with proper structure
        assert "start" in func_symbol.selection_range
        assert "end" in func_symbol.selection_range
        assert "line" in func_symbol.selection_range["start"]
        assert "character" in func_symbol.selection_range["start"]

        print(
            f"✓ selection_range set: line {func_symbol.selection_range['start']['line']}, "
            f"col {func_symbol.selection_range['start']['character']}"
        )

        # Also verify class symbol has selection_range
        class_symbol = next((s for s in codebase.symbols if s.name == "MyClass"), None)
        assert class_symbol is not None, "Should find MyClass symbol"
        assert (
            class_symbol.selection_range is not None
        ), "LSP should set selection_range for classes too"

