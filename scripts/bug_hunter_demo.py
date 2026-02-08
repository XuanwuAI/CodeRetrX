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

from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

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
    "mcp__coderetrx__get_reference",
    "mcp__coderetrx__codeql_query",
    "mcp__coderetrx__llm_code_filter",
]


def build_system_prompt(repo_url: str, language: str) -> str:
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

Target repository: {repo_url}

{language_section}

## Workflow
1. Explore codebase structure via mcp__coderetrx__list_dir
2. Find source files via mcp__coderetrx__find_file_by_name
3. Search for dangerous patterns via mcp__coderetrx__keyword_search
4. Use mcp__coderetrx__llm_code_filter for semantic code search — it finds code by intent (e.g., filter_prompt="functions handling buffer operations", "cryptographic key cleanup logic"), complementing keyword_search which only matches text patterns
5. Run CodeQL for deep analysis via mcp__coderetrx__codeql_query
6. View suspicious files via mcp__coderetrx__view_file
7. Validate findings by tracing call chains (mcp__coderetrx__get_reference, mcp__coderetrx__view_file)
8. Report findings with file paths, line numbers, and call chain evidence

## CodeQL Analysis
There are two ways to run CodeQL:
- **Full scan**: Use the Skill tool to invoke the "codeql" skill. This runs the complete pipeline (build database, create data extensions, run standard security rulesets). Best for comprehensive scanning.
- **Custom query**: Use mcp__coderetrx__codeql_query to execute a specific QL query you write. If unsure about syntax, read the reference files under .claude/skills/codeql/references/ first (especially diagnostic-query-templates.md and language-details.md).

## Rules
- All file paths are relative to repo root
- Combine multiple regex patterns with | to reduce calls
- Only report vulnerabilities with verified call chains from entry point to sink
- Avoid false positives
- Prefer MCP tools (mcp__coderetrx__*) for code exploration and search — they are optimized for the target repository"""


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

Focus on real vulnerabilities, not style issues."""
    else:
        return f"""Perform a security analysis on this {language} codebase:

1. List the root directory to understand the project structure
2. Find all {language} source files
3. Search for dangerous patterns relevant to {language} via keyword_search
4. Use llm_code_filter for semantic search of security-sensitive code
5. Run CodeQL queries for {language}
6. View suspicious files for context
7. Summarize any security concerns found

Focus on real vulnerabilities, not style issues."""


def build_options(repo_url: str, language: str, model: str, max_turns: int) -> ClaudeCodeOptions:
    """Build ClaudeCodeOptions for the agent."""
    return ClaudeCodeOptions(
        system_prompt=build_system_prompt(repo_url, language),
        max_turns=max_turns,
        allowed_tools=MCP_TOOL_NAMES + [
            "Skill", "Glob", "Grep", "Read", "Write", "Bash",
        ],
        model=model,
        cwd=PROJECT_ROOT,
        permission_mode="bypassPermissions",
        extra_args={"setting-sources": "user,project,local"},
        mcp_servers={
            "coderetrx": {
                "command": "uv",
                "args": [
                    "run", "python", "-m",
                    "coderetrx.tools.mcp_server", repo_url,
                ],
            }
        },
    )


_tool_count = 0
_turn_count = 0


def print_message(message):
    """Print a streamed message from Claude Code SDK."""
    global _tool_count, _turn_count

    if isinstance(message, AssistantMessage):
        _turn_count += 1
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"\n{block.text}")
            elif isinstance(block, ToolUseBlock):
                _tool_count += 1
                print(f"  [{_tool_count}] {block.name}")

    elif isinstance(message, ResultMessage):
        print(f"\n{'=' * 60}")
        print(f"Done. {_turn_count} turns, {_tool_count} tool calls.")
        if getattr(message, "total_cost_usd", None):
            print(f"Cost: ${message.total_cost_usd:.4f}")
        print("=" * 60)


async def run_bug_hunt(
    repo_url: str,
    language: str = "auto",
    model: str = "claude-opus-4-6",
    max_turns: int = 50,
):
    """Run automated bug hunt using Claude Code SDK."""
    print("=" * 60)
    print("BUG HUNTER AGENT (Claude Code SDK)")
    print("=" * 60)
    print(f"Repository: {repo_url}")
    print(f"Language:   {language}")
    print(f"Model:      {model}")
    print("=" * 60)

    options = build_options(repo_url, language, model, max_turns)
    task = build_task_prompt(language)

    async with ClaudeSDKClient(options=options) as client:
        await client.connect()
        await client.query(task)
        async for message in client.receive_response():
            print_message(message)


async def interactive_mode(
    repo_url: str,
    language: str = "auto",
    model: str = "claude-opus-4-6",
):
    """Interactive mode for manual exploration."""
    print("=" * 60)
    print("BUG HUNTER AGENT - Interactive Mode (Claude Code SDK)")
    print("=" * 60)
    print(f"Repository: {repo_url}")
    print(f"Language:   {language}")
    print(f"Model:      {model}")
    print("\nType your requests. 'quit' to exit, 'auto' for automated scan.")
    print("=" * 60)

    options = build_options(repo_url, language, model, max_turns=50)

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

                async for message in client.receive_response():
                    print_message(message)

            except (KeyboardInterrupt, EOFError):
                break


def main():
    parser = argparse.ArgumentParser(description="Bug Hunter Agent (Claude Code SDK)")
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

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode(args.repo_url, args.language, args.model))
    else:
        asyncio.run(run_bug_hunt(args.repo_url, args.language, args.model, args.max_turns))


if __name__ == "__main__":
    main()