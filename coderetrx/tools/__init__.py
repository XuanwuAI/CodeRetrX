"""
Local tools for coderetrx agents.

This module contains both the original CrewAI-based tools and the new smolagents-based tools.
"""

# New smolagents-based tools
from .base import BaseTool
from .find_file_by_name import FindFileByNameTool
from .find_references import GetReferenceTool
from .keyword_search import KeywordSearchTool
from .list_dir import ListDirTool
from .view_file import ViewFileTool
from typing import Type
import sys

# Export the tools as the default
__all__ = [
    "FindFileByNameTool",
    "GetReferenceTool",
    "KeywordSearchTool",
    "ListDirTool",
    "ViewFileTool",
]
tool_classes = [FindFileByNameTool, GetReferenceTool, KeywordSearchTool, ListDirTool, ViewFileTool]
tool_map: dict[str, dict[str, BaseTool]] = {}


def get_tool_class(name: str) -> Type[BaseTool]:
    current_module = sys.modules[__name__]
    cls = getattr(current_module, name, None)
    if not issubclass(cls, BaseTool):  # type: ignore
        raise ValueError(f"{name} is not a valid tool")
    return cls

def get_tool(repo_url: str, name: str) -> BaseTool:
    if tool_map.get(repo_url) is None:
        tool_map[repo_url] = {}
    if tool_map[repo_url].get(name) is None:
        tool_map[repo_url][name] = get_tool_class(name)(repo_url)
    return tool_map[repo_url][name]

def list_tool_class()-> list[BaseTool]:
    return tool_classes 