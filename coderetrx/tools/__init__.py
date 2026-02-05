"""
This module contains tools exposed by mcp server.
"""

from typing import Type

from .base import BaseTool
from .find_file_by_name import FindFileByNameTool
from .get_references import GetReferenceTool
from .keyword_search import KeywordSearchTool
from .list_dir import ListDirTool
from .lsp import GetDefinitionTool, GetReferencesTool, ListSymbolTool
from .view_file import ViewFileTool

# Export the tools as the default
__all__ = [
    "FindFileByNameTool",
    "GetReferenceTool",
    "KeywordSearchTool",
    "ListDirTool",
    "ViewFileTool",
    "ListSymbolTool",
    "GetDefinitionTool",
    "GetReferencesTool",
]
tool_classes = [
    FindFileByNameTool,
    GetReferenceTool,
    KeywordSearchTool,
    ListDirTool,
    ViewFileTool,
    ListSymbolTool,
    GetDefinitionTool,
    GetReferencesTool,
]
tool_map: dict[str, dict[str, BaseTool]] = {}


def get_tool_class(name: str) -> Type[BaseTool]:
    for cls in tool_classes:
        if getattr(cls, "name", None) == name:
            return cls
    raise ValueError(f"{name} is not a valid tool")


def get_tool(repo_url: str, name: str) -> BaseTool:
    if tool_map.get(repo_url) is None:
        tool_map[repo_url] = {}
    if tool_map[repo_url].get(name) is None:
        tool_map[repo_url][name] = get_tool_class(name)(repo_url)
    return tool_map[repo_url][name]


def list_tool_class() -> list[BaseTool]:
    return tool_classes
