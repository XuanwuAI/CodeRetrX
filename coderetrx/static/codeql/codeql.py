import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from abc import ABC, abstractmethod

from ..codebase.languages import IDXSupportedLanguage
from coderetrx.utils.path import get_data_dir

logger = logging.getLogger(__name__)


@dataclass
class CodeQLRawResult:
    """Raw JSON result from CodeQL query."""

    query_name: str
    json_data: Dict[str, Any]
    columns: List[str]
    tuples: List[List[Any]]


class CodeQLResultParser(ABC):
    """Abstract base class for custom result parsers."""

    @abstractmethod
    def parse(self, raw_result: CodeQLRawResult) -> List[Any]:
        """Parse raw CodeQL result into structured format."""
        pass


class CodeQLDatabase:
    """
    Represents a CodeQL database for a codebase.

    Manages the lifecycle of a CodeQL database including creation and cleanup.
    """

    def __init__(
        self,
        database_path: Path,
        language: IDXSupportedLanguage,
        source_dir: Path,
        temp_dir: Optional[Path] = None,
    ):
        self.database_path = database_path
        self.language = language
        self.source_dir = source_dir
        self.temp_dir = temp_dir
        self._is_created = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up the database directory if it was created in a temp location."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary CodeQL database: {self.temp_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup CodeQL database {self.temp_dir}: {e}"
                )

    @property
    def is_created(self) -> bool:
        """Check if the database has been created."""
        return self._is_created and self.database_path.exists()

    def mark_created(self):
        """Mark the database as created."""
        self._is_created = True


class CodeQLWrapper:
    """
    Python wrapper around the CodeQL CLI.

    Provides methods to create databases, run queries, and manage CodeQL operations
    while handling errors and resource cleanup automatically.
    """

    # Language mapping from our internal names to CodeQL language names
    LANGUAGE_MAP = {
        "javascript": "javascript",
        "typescript": "javascript",  # TypeScript uses JavaScript extractor
        "python": "python",
        "java": "java",
        "csharp": "csharp",
        "cpp": "cpp",
        "c": "cpp",  # C uses C++ extractor
        "go": "go",
        # "rust": "rust",  # Rust support is limited in CodeQL
    }

    def __init__(
        self,
        codeql_cli_path: Optional[str] = None,
        max_workers: int = 4,
        query_timeout: int = 300,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the CodeQL wrapper.

        Args:
            codeql_cli_path: Path to CodeQL CLI executable. If None, assumes 'codeql' is in PATH
            max_workers: Maximum number of parallel workers for batch operations
            query_timeout: Timeout in seconds for individual queries
            cache_dir: Custom cache directory. If None, creates a unique temp directory
        """
        self.codeql_cli_path = codeql_cli_path or "codeql"
        self.max_workers = max_workers
        self.query_timeout = query_timeout

        # Create unique cache directory to avoid conflicts
        # Each wrapper instance gets its own cache to prevent lock conflicts
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Create a unique cache directory with timestamp and process ID
            cache_suffix = f"{os.getpid()}_{int(time.time() * 1000)}"
            self.cache_dir = Path(
                tempfile.mkdtemp(prefix=f"codeql_cache_{cache_suffix}_")
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._verify_installation()

    def _verify_installation(self):
        """Verify that CodeQL CLI is installed and accessible."""
        try:
            result = subprocess.run(
                [self.codeql_cli_path, "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"CodeQL CLI not working: {result.stderr}")
            logger.info(f"CodeQL CLI found: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(f"CodeQL CLI not found at: {self.codeql_cli_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("CodeQL CLI verification timed out")

    def supports_language(self, language: IDXSupportedLanguage) -> bool:
        """Check if CodeQL supports the given language."""
        return language in self.LANGUAGE_MAP

    def create_database(
        self,
        source_dir: Path,
        language: IDXSupportedLanguage,
        project_name: str,
        database_name: Optional[str] = None,
    ) -> CodeQLDatabase:
        """
        Create a CodeQL database for the given source directory and language.

        Args:
            source_dir: Path to source code directory
            language: Language to create database for
            project_name: Name of the project for persistent storage
            database_name: Optional name for database (defaults to language name)

        Returns:
            CodeQLDatabase object

        Raises:
            RuntimeError: If database creation fails
        """
        if not self.supports_language(language):
            raise RuntimeError(f"Language {language} not supported by CodeQL")

        codeql_language = self.LANGUAGE_MAP[language]

        # Use persistent storage: get_data_dir()/codeql_db/{project_name}/{language_name}
        data_dir = get_data_dir()
        db_dir = data_dir / "codeql_db" / project_name
        db_dir.mkdir(parents=True, exist_ok=True)

        db_name = database_name or language
        database_path = db_dir / db_name

        try:
            cmd = [
                self.codeql_cli_path,
                "database",
                "create",
                str(database_path),
                f"--language={codeql_language}",
                f"--source-root={source_dir}",
                "--overwrite",
            ]

            # Set environment variables to isolate processes
            env = os.environ.copy()
            env["TMPDIR"] = str(self.cache_dir)
            # Force single-threaded operation to avoid cache conflicts
            env["CODEQL_THREADS"] = "1"

            logger.info(f"Creating CodeQL database for {language} at {database_path}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for database creation
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to create CodeQL database: {result.stderr}")

            # No temp_dir since we're using persistent storage
            database = CodeQLDatabase(
                database_path, language, source_dir, temp_dir=None
            )
            database.mark_created()
            logger.info(f"CodeQL database created successfully: {database_path}")
            return database

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Database creation timed out for {source_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to create database: {str(e)}")

    def _execute_query(
        self,
        database: CodeQLDatabase,
        query_file_path: Union[str, Path],
        query_name: str = "query",
    ) -> CodeQLRawResult:
        """
        Execute a CodeQL query and return raw result.

        This is the core query execution logic shared by other methods.

        Args:
            database: CodeQL database to query
            query_file_path: CodeQL query Path to ql file
            query_name: Name for the query (for debugging)

        Returns:
            CodeQLRawResult object containing parsed JSON data

        Raises:
            RuntimeError: If query execution fails
        """
        if not database.is_created:
            raise RuntimeError("Database is not created")

        # Create temporary file for BQRS output
        bqrs_file = tempfile.NamedTemporaryFile(suffix=".bqrs", delete=False)
        bqrs_path = bqrs_file.name
        bqrs_file.close()

        try:
            # Run query and save BQRS output to file
            cmd = [
                self.codeql_cli_path,
                "query",
                "run",
                str(query_file_path),
                "--database",
                str(database.database_path),
                "--output",
                bqrs_path,
            ]

            # Set environment variables to isolate processes
            env = os.environ.copy()
            env["TMPDIR"] = str(self.cache_dir)
            # Force single-threaded operation to avoid cache conflicts
            env["CODEQL_THREADS"] = "1"

            logger.debug(f"Running query: {query_name}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.query_timeout, env=env
            )

            if result.returncode != 0:
                logger.error(f"Query '{query_name}' failed: {result.stderr}")
                raise RuntimeError(f"Query '{query_name}' failed: {result.stderr}")

            return self._decode_bqrs(Path(bqrs_path), query_name)

        except subprocess.TimeoutExpired:
            logger.error(f"Query '{query_name}' timed out")
            raise RuntimeError(f"Query '{query_name}' timed out")
        except Exception as e:
            logger.error(f"Query '{query_name}' error: {str(e)}")
            raise RuntimeError(f"Query '{query_name}' error: {str(e)}")
        except:
            # Clean up on any failure
            try:
                os.unlink(bqrs_path)
            except Exception:
                pass
            raise
        finally:
            try:
                os.unlink(bqrs_path)
            except Exception:
                pass

    def _decode_bqrs(self, bqrs_path: Path, query_name: str) -> CodeQLRawResult:
        """
        Decode a BQRS file to CodeQLRawResult format.

        Args:
            bqrs_path: Path to the BQRS file
            query_name: Name of the query for debugging

        Returns:
            CodeQLRawResult object containing parsed JSON data

        Raises:
            RuntimeError: If decoding fails
        """
        decode_cmd = [
            self.codeql_cli_path,
            "bqrs",
            "decode",
            "--format=json",
            str(bqrs_path),
        ]

        # Set environment variables to isolate processes
        env = os.environ.copy()
        env["TMPDIR"] = str(self.cache_dir)
        env["CODEQL_THREADS"] = "1"

        logger.debug(f"Decoding BQRS output for query: {query_name}")
        decode_result = subprocess.run(
            decode_cmd,
            capture_output=True,
            text=True,
            timeout=self.query_timeout,
            env=env,
        )

        if decode_result.returncode != 0:
            logger.error(
                f"Failed to decode BQRS for query '{query_name}': {decode_result.stderr}"
            )
            raise RuntimeError(
                f"Failed to decode BQRS for query '{query_name}': {decode_result.stderr}"
            )

        # Parse JSON and create CodeQLRawResult
        try:
            if not decode_result.stdout.strip():
                # Return empty result for empty output
                return CodeQLRawResult(
                    query_name=query_name, json_data={}, columns=[], tuples=[]
                )

            data = json.loads(decode_result.stdout)
            logger.debug(f"Query '{query_name}' JSON data: {data}")

            # Extract the #select section which contains the query results
            select_data = data.get("#select", {})
            columns = select_data.get("columns", [])
            tuples = select_data.get("tuples", [])

            return CodeQLRawResult(
                query_name=query_name, json_data=data, columns=columns, tuples=tuples
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output for query '{query_name}': {e}")
            raise RuntimeError(
                f"Failed to parse JSON output for query '{query_name}': {e}"
            )

    def run_query_raw(
        self,
        database: CodeQLDatabase,
        query_file_path: Union[str, Path],
        query_name: str = "query",
    ) -> CodeQLRawResult:
        """
        Run a single CodeQL query against a database and return raw result.

        Args:
            database: CodeQL database to query
            query_file_path: CodeQL query Path to ql file
            query_name: Name for the query (for debugging)

        Returns:
            CodeQLRawResult object containing parsed JSON data

        Raises:
            RuntimeError: If query execution fails
        """

        raw_result = self._execute_query(database, query_file_path, query_name)
        return raw_result

    def run_query(
        self,
        database: CodeQLDatabase,
        query_file_path: Union[str, Path],
        query_name: str,
        parser: CodeQLResultParser,
    ) -> List[Any]:
        """
        Run a single CodeQL query against a database.

        Args:
            database: CodeQL database to query
            query_file_path: CodeQL query Path to ql file
            query_name: Name for the query (for debugging)

        Returns:
            List of CodeQL results

        Raises:
            RuntimeError: If query execution fails
        """
        try:
            # Use the new raw method and parse with default parser
            raw_result = self.run_query_raw(database, query_file_path, query_name)
            return parser.parse(raw_result)
        except Exception as e:
            logger.error(f"Query '{query_name}' error: {str(e)}")
            return []  # Return empty list for backward compatibility

    def run_queries(
        self,
        database: CodeQLDatabase,
        queries: Union[Dict[str, str], Dict[str, Path]],
        parser: CodeQLResultParser,
    ) -> Dict[str, List[Any]]:
        """
        Run custom CodeQL queries against a database.

        Args:
            database: CodeQL database to query
            queries: Dictionary of query_name -> query_file_path

        Returns:
            Dictionary of query_name -> results

        Raises:
            RuntimeError: If query execution fails
        """
        if not database.is_created:
            raise RuntimeError("Database is not created")

        results = {}

        # Run queries sequentially to avoid cache conflicts
        for query_name, query_file_path in queries.items():
            if query_file_path is None or not Path(query_file_path).exists():
                logger.warning(
                    f"Query file for '{query_name}' does not exist: {query_file_path}"
                )
                results[query_name] = []
                continue
            try:
                query_results = self.run_query(
                    database, query_file_path, query_name, parser
                )
                results[query_name] = query_results
                logger.debug(
                    f"Query '{query_name}' completed with {len(query_results)} results"
                )
            except Exception as e:
                logger.error(f"Query '{query_name}' failed: {str(e)}")
                results[query_name] = []

        return results

    def cleanup(self):
        """Clean up the cache directory."""
        try:
            if self.cache_dir and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                logger.debug(f"Cleaned up CodeQL cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup CodeQL cache {self.cache_dir}: {e}")

    def __enter__(self):
        """Support for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        self.cleanup()
