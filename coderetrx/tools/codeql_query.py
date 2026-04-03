import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Type, List, Any, Dict, Optional

from pydantic import BaseModel, Field

from coderetrx.tools.base import BaseTool
from coderetrx.static.codeql.codeql import CodeQLWrapper, CodeQLDatabase, CodeQLRawResult
from coderetrx.utils.path import get_data_dir

logger = logging.getLogger(__name__)

_DEFAULT_INSTALL_PATH = Path.home() / ".local" / "codeql"
_install_lock = asyncio.Lock()


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
        _DEFAULT_INSTALL_PATH / "codeql",
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


async def auto_install_codeql() -> Optional[str]:
    """Auto-install CodeQL CLI if not found. Returns path to CLI or None."""
    async with _install_lock:
        # Re-check after acquiring lock
        existing = find_codeql_cli()
        if existing:
            return existing

        logger.info("CodeQL CLI not found, auto-installing to %s ...", _DEFAULT_INSTALL_PATH)
        try:
            from coderetrx.static.codeql.installer import install_codeql
            result = await install_codeql(_DEFAULT_INSTALL_PATH)
            if result and result.exists():
                logger.info("CodeQL CLI installed at %s", result)
                return str(result)
            logger.warning("CodeQL auto-install returned no valid path")
        except Exception as e:
            logger.warning("CodeQL auto-install failed: %s", e)
        return None


def _get_query_qlpack_dir(language: str) -> Path:
    """Find the installed *-queries qlpack directory for the given language.

    Ad-hoc queries are written as temp files inside the installed queries qlpack
    (e.g. `~/.local/codeql/qlpacks/codeql/go-queries/1.4.5/`). This way the
    query inherits all standard-library dependencies that the qlpack already
    declares — no manual dependency management or `codeql pack install` needed.
    """
    codeql_path = find_codeql_cli()
    if codeql_path:
        codeql_root = Path(codeql_path).resolve().parent
        qlpacks_dir = codeql_root / "qlpacks" / "codeql"
        queries_pack = qlpacks_dir / f"{language}-queries"

        if queries_pack.is_dir():
            # Pick the latest version directory
            version_dirs = [
                d for d in queries_pack.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            if version_dirs:
                return max(version_dirs, key=lambda d: d.name)

    # Fallback: language not supported or codeql not installed
    fallback = Path(tempfile.gettempdir()) / "coderetrx_qlpacks" / language
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "No installed queries qlpack found for '%s', using fallback %s. "
        "Imports may not resolve.",
        language, fallback,
    )
    return fallback


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
        # Find CodeQL CLI path, auto-install if missing
        codeql_path = find_codeql_cli()
        if not codeql_path:
            codeql_path = await auto_install_codeql()
        if not codeql_path:
            return [CodeQLQueryResult(
                columns=["error"],
                tuples=[["CodeQL CLI auto-install failed. Manual install:\n"
                         "  python -m coderetrx.static.codeql.installer --install-path ~/.local/codeql"]]
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
                # Write query into a temporary qlpack so standard library
                # imports (e.g. `import go`) resolve correctly.
                qlpack_dir = _get_query_qlpack_dir(language)

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