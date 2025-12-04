import json
import os
import asyncio
import tempfile
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel

from .installer import install_rg


class RipgrepMessageType(str, Enum):
    """Types of messages in ripgrep JSON output."""

    BEGIN = "begin"
    END = "end"
    MATCH = "match"
    CONTEXT = "context"
    SUMMARY = "summary"


class ArbitraryData(BaseModel):
    """Model for arbitrary data that might be UTF-8 text or base64 encoded bytes."""

    text: Optional[str] = None
    bytes: Optional[str] = None


class Submatch(BaseModel):
    """Model for a submatch within a match."""

    match: ArbitraryData
    start: int
    end: int


class Duration(BaseModel):
    """Model for duration information."""

    secs: int
    nanos: int
    human: str


class Stats(BaseModel):
    """Model for search statistics."""

    elapsed: Duration
    searches: int
    searches_with_match: int
    bytes_searched: int
    bytes_printed: int
    matched_lines: int
    matches: int


class SummaryData(BaseModel):
    """Data for a 'summary' message."""

    elapsed_total: Duration
    stats: Stats


class BeginData(BaseModel):
    """Data for a 'begin' message."""

    path: Optional[ArbitraryData] = None


class EndData(BaseModel):
    """Data for an 'end' message."""

    path: Optional[ArbitraryData] = None
    binary_offset: Optional[int] = None
    stats: Stats


class MatchData(BaseModel):
    """Data for a 'match' message."""

    path: Optional[ArbitraryData] = None
    lines: ArbitraryData
    line_number: Optional[int] = None
    absolute_offset: int
    submatches: List[Submatch]


class ContextData(BaseModel):
    """Data for a 'context' message."""

    path: Optional[ArbitraryData] = None
    lines: ArbitraryData
    line_number: Optional[int] = None
    absolute_offset: int
    submatches: List[Submatch]


class RipgrepMessage(BaseModel):
    """Model for a ripgrep JSON message."""

    type: RipgrepMessageType
    data: Union[BeginData, EndData, MatchData, ContextData, SummaryData]

    def get_data(
        self,
    ) -> Union[BeginData, EndData, MatchData, ContextData, SummaryData]:
        """Get the correctly typed data based on message type."""
        data_dict = self.data
        if isinstance(data_dict, dict):
            if self.type == RipgrepMessageType.BEGIN:
                return BeginData.model_validate(data_dict)
            elif self.type == RipgrepMessageType.END:
                return EndData.model_validate(data_dict)
            elif self.type == RipgrepMessageType.MATCH:
                return MatchData.model_validate(data_dict)
            elif self.type == RipgrepMessageType.CONTEXT:
                return ContextData.model_validate(data_dict)
            elif self.type == RipgrepMessageType.SUMMARY:
                return SummaryData.model_validate(data_dict)
        return self.data


class GrepMatchResult(BaseModel):
    """A simplified result from a ripgrep search."""

    file_path: Path
    line_number: int
    line_text: str
    matches: List[str]


def parse_ripgrep_output(json_lines: str) -> List[GrepMatchResult]:
    """
    Parse ripgrep JSON output into a list of GrepMatchResult objects.

    Args:
        json_lines: String containing JSON lines output from ripgrep

    Returns:
        List of GrepMatchResult objects
    """
    results = []

    # Split the output by lines and parse each line as a JSON object
    for line in json_lines.strip().split("\n"):
        if not line:
            continue

        try:
            # Parse the JSON message
            message_dict = json.loads(line)
            message = RipgrepMessage.model_validate(message_dict)

            # We're only interested in match messages
            if message.type == RipgrepMessageType.MATCH:
                match_data = message.get_data()
                if isinstance(match_data, MatchData):
                    # Extract the file path
                    file_path = ""
                    if match_data.path and match_data.path.text:
                        file_path = match_data.path.text

                    # Extract the line text
                    line_text = ""
                    if match_data.lines.text:
                        line_text = match_data.lines.text.rstrip("\n")

                    # Extract the matches
                    matches = []
                    for submatch in match_data.submatches:
                        if submatch.match.text:
                            matches.append(submatch.match.text)

                    # Create a GrepMatchResult
                    result = GrepMatchResult(
                        file_path=Path(file_path),
                        line_number=(
                            match_data.line_number - 1
                            if match_data.line_number is not None
                            else 0
                        ),
                        line_text=line_text,
                        matches=matches,
                    )
                    results.append(result)
        except json.JSONDecodeError:
            # Skip lines that aren't valid JSON
            continue
        except Exception as e:
            # Log other errors but continue processing
            print(f"Error parsing ripgrep output: {e}")
            continue

    return results


async def check_ripgrep_available() -> Optional[Path]:
    """Check if ripgrep is available on the system."""
    if os.name != "nt" and (Path(__file__).parent / "rg").exists():
        return Path(__file__).parent / "rg"
    elif os.name == "nt" and (Path(__file__).parent / "rg.exe").exists():
        return Path(__file__).parent / "rg.exe"
    rg_path = await install_rg(Path(__file__).parent)
    if rg_path is None:
        print("Ripgrep installation failed. Please install ripgrep manually.")
    return rg_path


async def ripgrep_glob(
    search_dir: str | PathLike, pattern: str, extra_argv: List[str] = []
) -> list[str]:
    """
    Check if a file matching the glob pattern exists in the directory.

    Args:
        search_dir: Directory to search in
        pattern: Glob pattern to match files

    Returns:
        List of matching file paths

    Raises:
        RuntimeError: If ripgrep is not installed
    """
    # Check if ripgrep is installed
    rg_path = await check_ripgrep_available()
    if not rg_path:
        raise RuntimeError("Ripgrep (rg) binary not found. Please install ripgrep.")

    # Build the ripgrep command
    cmd = [rg_path.absolute(), "--files", "--no-messages"]
    for item in pattern.split("|"):
        cmd.extend(["-g", item])
    if extra_argv:
        cmd.extend(extra_argv)
    cmd.append(".")

    # Run the command asynchronously
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=search_dir,
        )

        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        return []
    if (
        proc.returncode != 0 and proc.returncode != 1
    ):  # rg returns 1 when no matches found
        raise RuntimeError(f"ripgrep failed with error {proc.returncode}: {stderr.decode()}")

    return stdout.decode().splitlines()


async def ripgrep_search(
    search_dir: str | PathLike,
    regexes: List[str],
    extra_argvs: List[str] = [],
    case_sensitive: bool = True,
    include_file_pattern: Optional[str] = None,
    exclude_file_pattern: Optional[str] = None,
    search_arg: Optional[str] = None,
) -> List[GrepMatchResult]:
    """
    Search for regexes in files using ripgrep.

    Args:
        search_dir: Directory to search in
        regexes: List of regexes to search for
        case_sensitive: Whether the search should be case sensitive
        include_file_pattern: Optional glob pattern to include files
        exclude_file_pattern: Optional glob pattern to exclude files

    Returns:
        List of GrepMatchResult objects containing matches

    Raises:
        RuntimeError: If ripgrep is not installed
    """
    # Check if ripgrep is installed
    rg_path = await check_ripgrep_available()
    if not rg_path:
        raise RuntimeError("Ripgrep (rg) binary not found. Please install ripgrep.")

    # Create a temporary file with the search patterns
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        tmpfile_path = f.name
        # Add word boundary markers to each symbol for more precise matching
        f.write("\n".join(regexes))

    try:
        # Build the ripgrep command
        cmd = [
            str(rg_path.absolute()),
            "-f",
            tmpfile_path,
            "--no-messages",
            "--line-number",
            "--json",
        ]

        # Add case sensitivity option
        if not case_sensitive:
            cmd.append("-i")

        # Add file pattern if provided

        if include_file_pattern:
            for pattern in include_file_pattern.split("|"):
                cmd.extend(["-g", pattern])

        if exclude_file_pattern:
            for pattern in exclude_file_pattern.split("|"):
                cmd.extend(["-g", f"!{pattern}"])

        # Add search directory
        if extra_argvs:
            cmd.extend(extra_argvs)
        cmd.append(search_arg or ".")

        # Run the command asynchronously
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(search_dir),
        )

        stdout, stderr = await proc.communicate()

        if (
            proc.returncode != 0 and proc.returncode != 1
        ):  # rg returns 1 when no matches found
            raise RuntimeError(f"ripgrep failed with error {proc.returncode}: {stderr.decode()}")

        # Parse the output
        return parse_ripgrep_output(stdout.decode("utf-8"))

    except FileNotFoundError:
        return []

    finally:
        # Clean up the temporary file
        Path(tmpfile_path).unlink(missing_ok=True)

def rgx_warp_symbol(symbol: str) -> str:
    return f"\\b{symbol}\\b"

async def ripgrep_search_symbols(
    search_dir: str | PathLike,
    symbols: List[str],
    extra_argvs: List[str] = [],
    case_sensitive: bool = True,
    include_file_pattern: Optional[str] = None,
    exclude_file_pattern: Optional[str] = None,
    search_arg: Optional[str] = None,
) -> List[GrepMatchResult]:
    return await ripgrep_search(
        search_dir,
        [rgx_warp_symbol(symbol) for symbol in symbols],
        extra_argvs,
        case_sensitive,
        include_file_pattern,
        exclude_file_pattern,
        search_arg,
    )


async def ripgrep_raw(
    search_dir: Union[str, PathLike],
    symbols: List[str],
    search_arg: Optional[str] = None,
    case_sensitive: bool = True,
    file_pattern: Optional[str] = None,
) -> str:
    """
    Run ripgrep and return the raw stdout output.

    Args:
        search_dir: Directory to search in
        symbols: List of regex patterns to search for
        search_arg: Optional specific file or directory to search within search_dir
        case_sensitive: Whether the search should be case sensitive
        file_pattern: Optional glob pattern to filter files

    Returns:
        Raw stdout output from ripgrep as a string

    Raises:
        RuntimeError: If ripgrep is not installed
    """
    # Check if ripgrep is installed
    rg_path = await check_ripgrep_available()
    if not rg_path:
        raise RuntimeError("Ripgrep (rg) binary not found. Please install ripgrep.")

    # Create a temporary file with the search patterns
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        tmpfile_path = f.name
        # Add word boundary markers to each symbol for more precise matching
        f.write("\n".join(map(lambda q: f"\\b{q}\\b", symbols)))

    try:
        # Build the ripgrep command
        cmd = [
            str(rg_path.absolute()),
            "-f",
            tmpfile_path,
            "--no-messages",
            "--line-number",
        ]

        # Add case sensitivity option
        if not case_sensitive:
            cmd.append("-i")

        # Add file pattern if provided
        if file_pattern:
            cmd.extend(["-g", file_pattern])

        # Add search directory
        if search_arg:
            if not (Path(search_dir) / search_arg).exists():
                raise RuntimeError(f"File or directory {search_arg} not found.")
            cmd.append(search_arg)
        else:
            cmd.append(".")

        # Run the command asynchronously
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(search_dir),
        )

        stdout, stderr = await proc.communicate()

        if (
            proc.returncode != 0 and proc.returncode != 1
        ):  # rg returns 1 when no matches found
            raise RuntimeError(f"ripgrep failed with error {proc.returncode}: {stderr.decode()}")

        # Return the raw stdout
        return stdout.decode("utf-8")

    finally:
        # Clean up the temporary file
        Path(tmpfile_path).unlink(missing_ok=True)


async def main():
    # Example usage
    try:
        # Check if ripgrep is available
        rg_path = await check_ripgrep_available()
        print(f"Ripgrep path: {rg_path}")

        # Example glob search
        print("Example glob search:")
        glob_results = await ripgrep_glob(".", "*.py")
        print(glob_results)

        # Example search
        print("\nExample search:")
        search_results = await ripgrep_search(
            ".", ["async", "await"], case_sensitive=True, include_file_pattern="*.py"
        )
        for result in search_results:
            print(f"{result.file_path}:{result.line_number}: {result.line_text}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
