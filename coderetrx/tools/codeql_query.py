import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Type, List, Any, Dict, Optional

from pydantic import BaseModel, Field

from coderetrx.tools.base import BaseTool
from coderetrx.static.codeql.codeql import CodeQLWrapper, CodeQLDatabase, CodeQLRawResult
from coderetrx.utils.path import get_data_dir


def find_codeql_cli() -> Optional[str]:
    """
    Find CodeQL CLI path by checking:
    1. CODEQL_CLI_PATH environment variable
    2. Common installation paths
    3. System PATH
    """
    # Check environment variable first
    env_path = os.environ.get("CODEQL_CLI_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Check common installation paths
    common_paths = [
        Path.home() / ".local" / "codeql" / "codeql",
        Path("/opt/codeql/codeql"),
        Path("/usr/local/codeql/codeql"),
    ]
    for path in common_paths:
        if path.exists():
            return str(path)

    # Check if codeql is in PATH
    if shutil.which("codeql"):
        return "codeql"

    return None


class CodeQLQueryArgs(BaseModel):
    query: str = Field(
        description="The CodeQL query string or path to a .ql file"
    )
    language: str = Field(
        description="The language of the codebase (e.g., python, javascript, java, go, cpp)"
    )


class CodeQLQueryResult(BaseModel):
    columns: List[str] = Field(description="Column names from the query result")
    tuples: List[List[Any]] = Field(description="Result rows as list of tuples")

    @classmethod
    def repr(cls, entries: list["CodeQLQueryResult"]) -> str:
        if not entries:
            return "No results found."
        result = entries[0]
        if not result.tuples:
            return "Query returned no results."

        output = f"Columns: {result.columns}\n\n"
        for i, row in enumerate(result.tuples[:100]):  # Limit to 100 rows
            output += f"{i+1}. {row}\n"
        if len(result.tuples) > 100:
            output += f"\n... and {len(result.tuples) - 100} more rows"
        return output


class CodeQLQueryTool(BaseTool):
    name = "codeql_query"
    description = (
        "Run arbitrary CodeQL queries against the codebase.\n"
        "Features:\n"
        "1. Execute any CodeQL query string or .ql file\n"
        "2. Automatically creates/reuses CodeQL database for the repo\n"
        "3. Returns raw query results as columns and tuples\n\n"
        "Example query:\n"
        "  import python\n"
        "  from Function f\n"
        "  select f.getName(), f.getLocation().getFile().getRelativePath(), f.getLocation().getStartLine()"
    )
    args_schema: ClassVar[Type[CodeQLQueryArgs]] = CodeQLQueryArgs

    def forward(self, query: str, language: str) -> str:
        return self.run_sync(query=query, language=language)

    async def _run(self, query: str, language: str) -> list[CodeQLQueryResult]:
        # Find CodeQL CLI path
        codeql_path = find_codeql_cli()
        if not codeql_path:
            install_hint = (
                "CodeQL CLI not found. To install:\n\n"
                "  python -m coderetrx.static.codeql.installer --install-path ~/.local/codeql\n\n"
                "Then either:\n"
                "  1. Add to PATH: export PATH=$PATH:~/.local/codeql\n"
                "  2. Or set environment variable: export CODEQL_CLI_PATH=~/.local/codeql/codeql"
            )
            return [CodeQLQueryResult(
                columns=["error"],
                tuples=[[install_hint]]
            )]

        try:
            wrapper = CodeQLWrapper(codeql_cli_path=codeql_path)
        except RuntimeError as e:
            return [CodeQLQueryResult(
                columns=["error"],
                tuples=[[str(e)]]
            )]

        with wrapper:
            if not wrapper.supports_language(language):
                return [CodeQLQueryResult(
                    columns=["error"],
                    tuples=[[f"Language '{language}' is not supported by CodeQL"]]
                )]

            # Get or create database
            data_dir = get_data_dir()
            db_path = data_dir / "codeql_db" / self.repo_id / language

            if db_path.exists():
                database = CodeQLDatabase(
                    database_path=db_path,
                    language=language,
                    source_dir=self.repo_path,
                )
                database.mark_created()
            else:
                database = wrapper.create_database(
                    source_dir=self.repo_path,
                    language=language,
                    project_name=self.repo_id,
                    database_name=language,
                )

            # Check if query is a file path or query string
            # Avoid Path() on long strings or strings with newlines (query content)
            is_file_path = (
                len(query) < 256
                and '\n' not in query
                and not query.strip().startswith('import ')
            )
            if is_file_path:
                query_path = Path(query)
                is_file_path = query_path.exists() and query_path.suffix == ".ql"

            if is_file_path:
                raw_result = wrapper.run_query_raw(database, query_path, query_name="custom_query")
            else:
                # Query is a string, write to temp file in qlpack directory
                # so that standard library imports work
                qlpack_dir = Path(__file__).parent.parent / "static" / "codebase" / "queries" / "codeql" / language
                if not qlpack_dir.exists():
                    # Fallback to temp directory if no qlpack for this language
                    qlpack_dir = Path(tempfile.gettempdir())

                # Create unique temp file in qlpack directory
                import uuid
                tmp_filename = f"_tmp_query_{uuid.uuid4().hex[:8]}.ql"
                tmp_path = qlpack_dir / tmp_filename

                try:
                    tmp_path.write_text(query)
                    raw_result = wrapper.run_query_raw(database, tmp_path, query_name="custom_query")
                finally:
                    tmp_path.unlink(missing_ok=True)

            # Convert column dicts to names
            column_names = [col.get("name", f"col_{i}") for i, col in enumerate(raw_result.columns)]

            return [CodeQLQueryResult(
                columns=column_names,
                tuples=raw_result.tuples,
            )]