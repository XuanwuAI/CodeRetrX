"""LSP-based parser for CodeRetrX.

This module provides an LSP (Language Server Protocol) based parser that uses
the lspyc client library to extract code structures from source files.
"""

from .parser import LSPParser

__all__ = ["LSPParser"]
