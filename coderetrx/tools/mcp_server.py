"""
MCP Server for Code Analysis Tools

This module implements a Model-Code-Prompt (MCP) server that exposes various
code analysis tools through a standardized interface. It supports both stdio and
SSE transport mechanisms for communication.
"""

import argparse
import anyio
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.server.sse
from pydantic import AnyUrl, BaseModel as PydanticBaseModel
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from coderetrx.tools.base import BaseTool
from coderetrx.tools import list_tool_class, get_tool
import logging


logger = logging.getLogger("mcp")

# Create MCP server instance
server = Server("code_tool")

# Type definitions for improved type hinting
ToolType = TypeVar("ToolType", bound=BaseTool)
JsonSchema = Dict[str, Any]

# Global dictionary to map tool names to tool instances


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="MCP Server for Code Analysis Tools")

    # Required arguments
    parser.add_argument(
        "repo_url", type=str, help="repo_urlsitory URL or identifier to analyze"
    )

    # Optional arguments
    parser.add_argument(
        "--use-sse",
        action="store_true",
        help="Use Server-Sent Events (SSE) instead of stdio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10001,
        help="Port number for SSE server (default: 10001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for SSE server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def model_to_json_schema(model_class: Type[PydanticBaseModel]) -> JsonSchema:
    """
    Convert a Pydantic model to a JSON schema.

    Args:
        model_class: The Pydantic model class to convert

    Returns:
        A dictionary representing the JSON schema
    """
    properties = {}
    required = []

    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        field_description = field_info.description

        # Track required fields
        if field_info.is_required():
            required.append(field_name)

        # Map field type to JSON schema type
        field_schema = {"type": get_json_type(field_type)}

        # Add description if available
        if field_description:
            field_schema["description"] = field_description

        properties[field_name] = field_schema

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def get_json_type(python_type) -> str:
    """
    Map Python types to JSON schema types.

    Args:
        python_type: The Python type to map

    Returns:
        The corresponding JSON schema type as a string
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    if python_type in type_map:
        return type_map[python_type]

    origin = get_origin(python_type)
    if origin is not None:
        if origin in (list, set, tuple):
            return "array"
        elif origin is dict:
            return "object"
        elif origin is Union:
            args = get_args(python_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = next(arg for arg in args if arg is not type(None))
                return get_json_type(non_none_type)

    return "string"


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    return ""


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """
    Handle the list_tools request by providing metadata for all available tools.

    Returns:
        List of Tool objects containing metadata for each available tool
    """
    mcp_tools = []
    try:
        for tool_class in list_tool_class():
            mcp_tools.append(
                types.Tool(
                    name=tool_class.name,
                    description=tool_class.description,
                    inputSchema=model_to_json_schema(tool_class.args_schema),
                )
            )
        return mcp_tools
    except Exception as e:
        logging.error(f"Error listing tools: {str(e)}")
        raise


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Optional[Dict[str, Any]] = None
) -> List[types.TextContent]:
    """
    Handle tool execution requests.

    Args:
        name: The name of the tool to call
        arguments: Tool arguments as a dictionary

    Returns:
        List of content objects containing tool execution results
    """
    if arguments is None:
        arguments = {}

    tool = get_tool(repo_url, name)
    if not tool:
        raise ValueError(f"Unknown tool name: {name}")

    try:
        result = await tool.run(**arguments)
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        logging.error(f"Error executing tool '{name}': {str(e)}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def run_stdio_server() -> None:
    """Run the server using stdio transport."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="code_analysis_server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_sse_server(host: str, port: int, debug: bool) -> None:
    """Run the server using SSE transport."""
    sse = mcp.server.sse.SseServerTransport("/messages/")

    async def handle_sse(request) -> None:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    app = Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Main entry point for the MCP server."""
    args = parse_arguments()
    global repo_url
    repo_url = args.repo_url
    if args.use_sse:
        run_sse_server(args.host, args.port, args.debug)
    else:
        anyio.run(run_stdio_server)


if __name__ == "__main__":
    main()
