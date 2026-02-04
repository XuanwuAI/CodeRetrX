"""Tests for LSP parser implementation."""

from coderetrx.static.codebase import Codebase
from coderetrx.static.codebase.parsers import LSPParser
from coderetrx.static.codebase.codebase import ChunkType


class TestLSPParser:
    """Test suite for LSP parser."""

    def test_supports_language(self):
        """Test that LSP parser reports correct language support."""
        parser = LSPParser()

        # Should support these languages
        assert parser.supports_language("python")
        assert parser.supports_language("javascript")
        assert parser.supports_language("typescript")
        assert parser.supports_language("rust")
        assert parser.supports_language("go")

        # Should not support unsupported languages
        assert not parser.supports_language("php")
        assert not parser.supports_language("elixir")

    def test_parser_initialization(self):
        """Test LSP parser initialization."""
        parser = LSPParser()
        assert parser is not None
        assert parser._client is None  # Client should be lazy-initialized
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
        test_file.write_text('''
def hello_world():
    """A simple function."""
    print("Hello, world!")

class MyClass:
    """A simple class."""

    def method(self):
        """A method."""
        pass
''')

        # Parse with LSP
        codebase = Codebase.new(
            id="test_lsp",
            dir=tmp_path,
            parser="lsp"
        )

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
        from coderetrx.static.codebase.parsers import ParserFactory

        parser = ParserFactory.get_parser("lsp")
        assert isinstance(parser, LSPParser)

    def test_lsp_in_available_parsers(self):
        """Test that LSP parser appears in available parsers list."""
        from coderetrx.static.codebase.parsers import ParserFactory

        available = ParserFactory.get_available_parsers()
        assert "lsp" in available
        # Note: availability will be False if LSP servers not installed
