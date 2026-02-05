"""Shared constants for LSP tools."""

# LSP SymbolKind enum mapping to readable names
# From LSP specification: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
SYMBOL_KIND_NAMES = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum_member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type_parameter",
}


def get_symbol_kind_name(kind: int) -> str:
    """Get the readable name for a SymbolKind value.

    Args:
        kind: LSP SymbolKind integer value

    Returns:
        Readable string name for the symbol kind
    """
    return SYMBOL_KIND_NAMES.get(kind, f"unknown({kind})")
