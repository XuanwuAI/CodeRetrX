"""Mapping between LSP SymbolKind and CodeRetrX tags.

This module provides functions to convert LSP symbol kinds to CodeRetrX tags
and determine chunk types for LSP symbols.
"""

from typing import TYPE_CHECKING, Optional

from ...languages import IDXSupportedTag

if TYPE_CHECKING:
    from ...codebase import ChunkType


# LSP SymbolKind enumeration (LSP spec 3.17)
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
SYMBOLKIND_FILE = 1
SYMBOLKIND_MODULE = 2
SYMBOLKIND_NAMESPACE = 3
SYMBOLKIND_PACKAGE = 4
SYMBOLKIND_CLASS = 5
SYMBOLKIND_METHOD = 6
SYMBOLKIND_PROPERTY = 7
SYMBOLKIND_FIELD = 8
SYMBOLKIND_CONSTRUCTOR = 9
SYMBOLKIND_ENUM = 10
SYMBOLKIND_INTERFACE = 11
SYMBOLKIND_FUNCTION = 12
SYMBOLKIND_VARIABLE = 13
SYMBOLKIND_CONSTANT = 14
SYMBOLKIND_STRING = 15
SYMBOLKIND_NUMBER = 16
SYMBOLKIND_BOOLEAN = 17
SYMBOLKIND_ARRAY = 18
SYMBOLKIND_OBJECT = 19
SYMBOLKIND_KEY = 20
SYMBOLKIND_NULL = 21
SYMBOLKIND_ENUMMEMBER = 22
SYMBOLKIND_STRUCT = 23
SYMBOLKIND_EVENT = 24
SYMBOLKIND_OPERATOR = 25
SYMBOLKIND_TYPEPARAMETER = 26


# Mapping from LSP SymbolKind to CodeRetrX IDXSupportedTag
SYMBOLKIND_TO_TAG: dict[int, IDXSupportedTag] = {
    # PRIMARY types - main code structures
    SYMBOLKIND_CLASS: "definition.class",
    SYMBOLKIND_METHOD: "definition.method",
    SYMBOLKIND_CONSTRUCTOR: "definition.method",
    SYMBOLKIND_ENUM: "definition.type",
    SYMBOLKIND_INTERFACE: "definition.interface",
    SYMBOLKIND_FUNCTION: "definition.function",
    SYMBOLKIND_MODULE: "definition.module",
    SYMBOLKIND_NAMESPACE: "definition.module",
    SYMBOLKIND_STRUCT: "definition.class",
    # VARIABLE types
    SYMBOLKIND_VARIABLE: "definition.variable",
    SYMBOLKIND_CONSTANT: "definition.variable",
    SYMBOLKIND_FIELD: "definition.variable",
    SYMBOLKIND_PROPERTY: "definition.variable",
    SYMBOLKIND_ENUMMEMBER: "definition.variable",
}


# SymbolKind values that should be extracted as PRIMARY chunks
PRIMARY_SYMBOLKINDS = {
    SYMBOLKIND_CLASS,
    SYMBOLKIND_METHOD,
    SYMBOLKIND_CONSTRUCTOR,
    SYMBOLKIND_ENUM,
    SYMBOLKIND_INTERFACE,
    SYMBOLKIND_FUNCTION,
    SYMBOLKIND_MODULE,
    SYMBOLKIND_NAMESPACE,
    SYMBOLKIND_STRUCT,
}


# SymbolKind values that should be extracted as VARIABLE chunks
VARIABLE_SYMBOLKINDS = {
    SYMBOLKIND_VARIABLE,
    SYMBOLKIND_CONSTANT,
    SYMBOLKIND_FIELD,
    SYMBOLKIND_PROPERTY,
    SYMBOLKIND_ENUMMEMBER,
}


def symbolkind_to_tag(
    kind: int, extract_variables: bool = False
) -> Optional[IDXSupportedTag]:
    """Convert LSP SymbolKind to CodeRetrX IDXSupportedTag.

    Args:
        kind: LSP SymbolKind integer value
        extract_variables: Whether to extract variable definitions

    Returns:
        Corresponding IDXSupportedTag, or None if the kind should be skipped
    """
    # Always extract primary symbols
    if kind in PRIMARY_SYMBOLKINDS:
        return SYMBOLKIND_TO_TAG.get(kind)

    # Only extract variables if requested
    if extract_variables and kind in VARIABLE_SYMBOLKINDS:
        return SYMBOLKIND_TO_TAG.get(kind)

    return None


def symbolkind_to_chunk_type(kind: int) -> "ChunkType":
    """Determine the ChunkType for an LSP SymbolKind.

    Args:
        kind: LSP SymbolKind integer value

    Returns:
        Appropriate ChunkType for the symbol kind
    """
    from ...codebase import ChunkType

    if kind in PRIMARY_SYMBOLKINDS:
        return ChunkType.PRIMARY
    elif kind in VARIABLE_SYMBOLKINDS:
        return ChunkType.VARIABLE
    else:
        return ChunkType.OTHER


def is_primary_symbol(kind: int) -> bool:
    """Check if an LSP SymbolKind represents a primary code structure.

    Args:
        kind: LSP SymbolKind integer value

    Returns:
        True if the kind is a primary symbol (function, class, etc.)
    """
    return kind in PRIMARY_SYMBOLKINDS
