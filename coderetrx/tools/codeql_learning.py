"""
CodeQL Learning Tool - Fetch CodeQL documentation for query writing.

Use this tool when unsure about CodeQL syntax or types.
"""

import re
import httpx
from typing import ClassVar, Type
from pydantic import BaseModel, Field

from coderetrx.tools.base import BaseTool


# CodeQL documentation URLs by language
CODEQL_DOCS = {
    "go": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-go-code/",
    "python": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-python-code/",
    "javascript": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-javascript-code/",
    "java": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-java-code/",
    "cpp": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-cpp-code/",
    "csharp": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-csharp-code/",
    "ruby": "https://codeql.github.com/docs/codeql-language-guides/basic-query-for-ruby-code/",
    "reference": "https://codeql.github.com/docs/ql-language-reference/",
}


def html_to_text(html: str) -> str:
    """Simple HTML to text conversion."""
    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # Decode common HTML entities
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&amp;', '&')
    html = html.replace('&quot;', '"')
    # Collapse whitespace
    html = re.sub(r'\s+', ' ', html)
    return html.strip()


class CodeQLLearningArgs(BaseModel):
    language: str = Field(
        description="Language to look up CodeQL docs for. Options: go, python, javascript, java, cpp, csharp, ruby, reference"
    )


class CodeQLLearningResult(BaseModel):
    success: bool
    content: str
    language: str
    url: str

    @classmethod
    def repr(cls, entries: list["CodeQLLearningResult"]) -> str:
        if not entries:
            return "No results."
        result = entries[0]
        if not result.success:
            return f"Error fetching docs for {result.language}: {result.content}"
        # Truncate long content
        content = result.content
        if len(content) > 10000:
            content = content[:10000] + "\n\n... (truncated)"
        return f"CodeQL Documentation for {result.language}:\n\n{content}"


class CodeQLLearningTool(BaseTool):
    name = "codeql_learning"
    description = (
        "Query CodeQL documentation when unsure about syntax or types.\n\n"
        "Available languages: go, python, javascript, java, c, cpp, csharp, ruby\n"
        "Use 'reference' for general QL language reference."
    )
    args_schema: ClassVar[Type[CodeQLLearningArgs]] = CodeQLLearningArgs

    def __init__(self, repo_url: str, uuid: str = None):
        # This tool doesn't need repo access, skip cloning
        self.repo_url = repo_url
        self.repo_id = None
        self.uuid = uuid
        self.repo_path = None

    def forward(self, language: str) -> str:
        return self.run_sync(language=language)

    async def _run(self, language: str) -> list[CodeQLLearningResult]:
        language = language.lower().strip()

        # Map common aliases
        aliases = {
            "c": "cpp",
            "c++": "cpp",
            "js": "javascript",
            "ts": "javascript",
            "typescript": "javascript",
            "c#": "csharp",
            "ref": "reference",
        }
        language = aliases.get(language, language)

        url = CODEQL_DOCS.get(language)
        if not url:
            available = ", ".join(CODEQL_DOCS.keys())
            return [CodeQLLearningResult(
                success=False,
                content=f"Unknown language '{language}'. Available: {available}",
                language=language,
                url=""
            )]

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; CodeRetrX/1.0)"
                })
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    text = html_to_text(response.text)
                else:
                    text = response.text

                return [CodeQLLearningResult(
                    success=True,
                    content=text,
                    language=language,
                    url=url
                )]
        except httpx.HTTPStatusError as e:
            return [CodeQLLearningResult(
                success=False,
                content=f"HTTP {e.response.status_code}",
                language=language,
                url=url
            )]
        except Exception as e:
            return [CodeQLLearningResult(
                success=False,
                content=str(e),
                language=language,
                url=url
            )]
