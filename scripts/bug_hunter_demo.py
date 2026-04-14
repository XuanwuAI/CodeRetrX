"""
Bug Hunter Agent - Uses Claude Code SDK for automated security analysis.

Leverages Claude Code's native capabilities and CodeRetrX MCP tools for bug hunting.
CodeQL documentation is provided via Claude Skills (.claude/skills/codeql/),
so codeql_learning tool is no longer needed.

Usage:
    uv run scripts/bug_hunter.py <repo_url> [--language auto] [--model opus]
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from claude_code_sdk import ClaudeCodeOptions

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).parent.parent)

# Supported languages and their file extensions
LANGUAGE_EXTENSIONS = {
    "python": ["*.py", "*.pyi"],
    "javascript": ["*.js", "*.jsx", "*.mjs"],
    "typescript": ["*.ts", "*.tsx"],
    "java": ["*.java"],
    "go": ["*.go"],
    "cpp": ["*.cpp", "*.cc", "*.cxx", "*.c", "*.h", "*.hpp"],
    "csharp": ["*.cs"],
    "ruby": ["*.rb"],
    "rust": ["*.rs"],
    "swift": ["*.swift"],
    "kotlin": ["*.kt", "*.kts"],
    "php": ["*.php"],
}

CODEQL_LANGUAGES = [
    "python", "javascript", "typescript", "java",
    "go", "cpp", "csharp", "ruby", "swift",
]

MCP_TOOL_NAMES = [
    "mcp__coderetrx__list_dir",
    "mcp__coderetrx__find_file_by_name",
    "mcp__coderetrx__keyword_search",
    "mcp__coderetrx__view_file",
    "mcp__coderetrx__codeql_query",
    "mcp__coderetrx__llm_code_filter",
    "mcp__coderetrx__list_symbol",
    "mcp__coderetrx__get_definition",
    "mcp__coderetrx__get_references",
]


def build_system_prompt(repo_url: str, repo_name: str, language: str) -> str:
    """Build system prompt for the bug hunter agent."""
    if language == "auto":
        all_extensions = []
        for exts in LANGUAGE_EXTENSIONS.values():
            all_extensions.extend(exts)
        ext_pattern = "|".join(all_extensions)
        language_section = f"""## Language Detection
Detect languages in this codebase first.
Use mcp__coderetrx__find_file_by_name with pattern "{ext_pattern}" to find source files.

Supported languages: {", ".join(LANGUAGE_EXTENSIONS.keys())}
CodeQL supported: {", ".join(CODEQL_LANGUAGES)}"""
    else:
        exts = LANGUAGE_EXTENSIONS.get(language, [f"*.{language}"])
        language_section = f"""## Target Language: {language}
File patterns: {", ".join(exts)}"""

    return f"""You are a Bug Hunter Agent specialized in finding security vulnerabilities and bugs.

Target repository: {repo_name}

{language_section}

## Workflow
1. Explore codebase structure via mcp__coderetrx__list_dir
2. Find source files via mcp__coderetrx__find_file_by_name
3. Search for dangerous patterns via mcp__coderetrx__keyword_search
4. Use mcp__coderetrx__llm_code_filter for semantic code search — it finds code by intent (e.g., filter_prompt="functions handling buffer operations", "cryptographic key cleanup logic"), complementing keyword_search which only matches text patterns
5. Use mcp__coderetrx__list_symbol to list all symbols (functions, classes, methods) in a file — gives a structural overview with precise line:column positions
6. Use mcp__coderetrx__get_definition to jump to where a symbol is defined (go-to-definition via LSP)
7. Use mcp__coderetrx__get_references to find all usages of a symbol across the codebase (semantic find-references via LSP)
8. Run CodeQL for deep analysis via mcp__coderetrx__codeql_query
9. View suspicious files via mcp__coderetrx__view_file
10. Validate findings by tracing call chains (mcp__coderetrx__get_references, mcp__coderetrx__get_definition, mcp__coderetrx__view_file)
11. Report findings with file paths, line numbers, and call chain evidence

## CodeQL Analysis
There are two ways to run CodeQL:
- **Full scan**: Use the Skill tool to invoke the "codeql" skill. This runs the complete pipeline (build database, create data extensions, run standard security rulesets). Best for comprehensive scanning.
- **Custom query**: Use mcp__coderetrx__codeql_query to execute a specific QL query you write. Before writing custom queries, read the official CodeQL qlpacks at ~/.local/codeql/qlpacks/codeql/ to learn the correct API and syntax. You can also pass a .ql file path directly to mcp__coderetrx__codeql_query.

## Rules
- All file paths are relative to repo root
- mcp__coderetrx__view_file uses 0-based half-open ranges: [start_line, end_line). start_line is inclusive and end_line is exclusive.
- Tools that print `Lines a-b` generally report inclusive ranges. When passing those ranges to mcp__coderetrx__view_file, convert them to [a, b+1).
- Combine multiple regex patterns with | to reduce calls
- Only report vulnerabilities with verified call chains from entry point to sink
- Avoid false positives
- Prefer MCP tools (mcp__coderetrx__*) for code exploration and search — they are optimized for the target repository
- Use LSP tools (list_symbol, get_definition, get_references) for precise semantic navigation — they understand code structure beyond text matching
"""


def build_task_prompt(language: str) -> str:
    """Build the initial task prompt."""
    if language == "auto":
        return """Perform a comprehensive security analysis on this codebase:

1. List the root directory to understand the project structure
2. Find source files to detect what languages are used
3. For each language found, search for dangerous patterns via keyword_search
4. Use llm_code_filter for semantic search of security-sensitive code (e.g., input handling, memory operations, crypto logic)
5. Run CodeQL queries for the primary language(s) found
6. View suspicious files for context
7. Summarize all security concerns found

Focus on real vulnerabilities, not style issues. You MUST be sure that each vulnerability reported is exploitable."""
    else:
        return f"""Perform a security analysis on this {language} codebase:

1. List the root directory to understand the project structure
2. Find all {language} source files
3. Search for dangerous patterns relevant to {language} via keyword_search
4. Use llm_code_filter for semantic search of security-sensitive code
5. Run CodeQL queries for {language}
6. View suspicious files for context
7. Summarize any security concerns found

Focus on real vulnerabilities, not style issues. You MUST be sure that each vulnerability reported is exploitable."""


def build_options(
    repo_url: str,
    repo_name: str,
    language: str,
    model: str,
    max_turns: int,
) -> "ClaudeCodeOptions":
    """Build ClaudeCodeOptions for the agent."""
    _, ClaudeCodeOptions = _require_claude_sdk()
    return ClaudeCodeOptions(
        system_prompt=build_system_prompt(repo_url, repo_name, language),
        max_turns=max_turns,
        allowed_tools=MCP_TOOL_NAMES + [
            "Skill", "Glob", "Grep", "Read", "Write", "Bash", "TodoWrite",
        ],
        model=model,
        cwd=PROJECT_ROOT,
        permission_mode="bypassPermissions",
        env={"MAX_MCP_OUTPUT_TOKENS": "100000"},
        extra_args={"setting-sources": "user,project,local"},
        mcp_servers={
            "coderetrx": {
                "command": "uv",
                "args": [
                    "run", "python", "-m",
                    "coderetrx.tools.mcp_server", repo_url,
                ],
                "cwd": PROJECT_ROOT,
            }
        },
    )


_tool_count = 0
_turn_count = 0


def _require_claude_sdk():
    """Import the Claude Code SDK only when the demo is executed."""
    try:
        from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient
    except ImportError as exc:
        raise RuntimeError(
            "claude-code-sdk is required for this demo but is not importable in "
            "the current environment. Rebuild the environment with `uv sync`, "
            "then rerun `uv run scripts/bug_hunter_demo.py <repo_url>`."
        ) from exc

    return ClaudeSDKClient, ClaudeCodeOptions


def _iter_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract structured content blocks from a raw Claude CLI message."""
    content = message.get("message", {}).get("content", [])
    return content if isinstance(content, list) else []


def _format_rate_limit_event(message: dict[str, Any]) -> str:
    """Format rate limit updates emitted by newer Claude CLI versions."""
    info = message.get("rate_limit_info", {})
    if not isinstance(info, dict):
        return "status=unknown"

    status = info.get("status", "unknown")
    parts = [f"status={status}"]

    utilization = info.get("utilization")
    if isinstance(utilization, (int, float)):
        parts.append(f"utilization={utilization:.0%}")

    rate_limit_type = info.get("rate_limit_type")
    if rate_limit_type:
        parts.append(f"type={rate_limit_type}")

    resets_at = info.get("resetsAt") or info.get("resets_at")
    if resets_at is not None:
        parts.append(f"resets_at={resets_at}")

    overage_status = info.get("overage_status")
    if overage_status:
        parts.append(f"overage_status={overage_status}")

    return ", ".join(parts)


def print_message(message: dict[str, Any]):
    """Print a streamed raw message from Claude Code CLI."""
    global _tool_count, _turn_count

    message_type = message.get("type")

    if message_type == "assistant":
        _turn_count += 1
        for block in _iter_content_blocks(message):
            block_type = block.get("type")
            if block_type == "text":
                print(f"\n{block.get('text', '')}")
            elif block_type == "tool_use":
                _tool_count += 1
                print(f"  [{_tool_count}] {block.get('name', '<unknown tool>')}")

    elif message_type == "user":
        for block in _iter_content_blocks(message):
            if block.get("type") == "tool_result":
                content = block.get("content")
                if content:
                    logger.debug("    -> %s", str(content)[:500])

    elif message_type == "rate_limit_event":
        logger.debug("Claude rate limit event: %s", _format_rate_limit_event(message))

    elif message_type == "result":
        print(f"\n{'=' * 60}")
        print(f"Done. {_turn_count} turns, {_tool_count} tool calls.")
        total_cost_usd = message.get("total_cost_usd")
        if total_cost_usd is not None:
            print(f"Cost: ${total_cost_usd:.4f}")
        print("=" * 60)

    else:
        logger.debug("Ignoring unsupported Claude CLI message: %s", message_type)


async def receive_response_tolerant(client: Any) -> "AsyncIterator[dict[str, Any]]":
    """
    Yield raw Claude CLI messages until a result frame is received.

    The deprecated Python Claude Code SDK parser raises on newer frame types
    such as `rate_limit_event`, so this demo reads the raw stream and handles
    only the message types it needs.
    """
    query = getattr(client, "_query", None)
    if query is None:
        raise RuntimeError("Claude SDK client is not connected.")

    async for message in query.receive_messages():
        yield message
        if message.get("type") == "result":
            return


async def run_bug_hunt(
    repo_url: str,
    language: str = "auto",
    model: str = "claude-opus-4-6",
    max_turns: int = 50,
):
    """Run automated bug hunt using Claude Code SDK."""
    ClaudeSDKClient, _ = _require_claude_sdk()
    repo_name = Path(repo_url).name
    print("=" * 60)
    print("BUG HUNTER AGENT Demo")
    print("=" * 60)
    print(f"Repository: {repo_name}")
    print(f"Language:   {language}")
    print(f"Model:      {model}")
    print("=" * 60)

    options = build_options(repo_url, repo_name, language, model, max_turns)
    task = build_task_prompt(language)

    async with ClaudeSDKClient(options=options) as client:
        await client.connect()
        await client.query(task)
        async for message in receive_response_tolerant(client):
            if message is not None:
                print_message(message)


async def interactive_mode(
    repo_url: str,
    language: str = "auto",
    model: str = "claude-opus-4-6",
):
    """Interactive mode for manual exploration."""
    ClaudeSDKClient, _ = _require_claude_sdk()
    repo_name = Path(repo_url).name
    print("=" * 60)
    print("BUG HUNTER AGENT Demo - Interactive Mode")
    print("=" * 60)
    print(f"Repository: {repo_name}")
    print(f"Language:   {language}")
    print(f"Model:      {model}")
    print("\nType your requests. 'quit' to exit, 'auto' for automated scan.")
    print("=" * 60)

    options = build_options(repo_url, repo_name, language, model, max_turns=50)

    async with ClaudeSDKClient(options=options) as client:
        # First connect without an initial prompt
        await client.connect()

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if user_input.lower() == "auto":
                    task = build_task_prompt(language)
                    await client.query(task)
                else:
                    await client.query(user_input)

                async for message in receive_response_tolerant(client):
                    if message is not None:
                        print_message(message)

            except (KeyboardInterrupt, EOFError):
                break


def main():
    parser = argparse.ArgumentParser(description="Bug Hunter Agent Demo")
    parser.add_argument("repo_url", help="Repository URL to analyze")
    parser.add_argument(
        "--language", "-l", default="auto",
        help="Language to analyze (default: auto-detect)",
    )
    parser.add_argument(
        "--model", "-m", default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-turns", "-t", type=int, default=50,
        help="Maximum agent turns (default: 50)",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Enable debug logging (shows tool outputs)",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.interactive:
        asyncio.run(interactive_mode(args.repo_url, args.language, args.model))
    else:
        asyncio.run(run_bug_hunt(args.repo_url, args.language, args.model, args.max_turns))


if __name__ == "__main__":
    main()
