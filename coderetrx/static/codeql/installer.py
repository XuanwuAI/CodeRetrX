import os
import platform
import sys
import asyncio
import shutil
import tarfile
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional, Tuple, Dict

import httpx

# CodeQL release information
CODEQL_VERSION = "v2.23.1"
CODEQL_REPO = "github/codeql-action"
CODEQL_BASE_URL = (
    f"https://github.com/{CODEQL_REPO}/releases/download/codeql-bundle-{CODEQL_VERSION}"
)


def get_platform_info() -> Optional[str]:
    """Get platform information and return the corresponding download filename

    Returns:
        Optional[str]: Matching platform filename or None if unsupported
    """
    system = platform.system().lower()

    # Map platform to CodeQL release filename
    platform_map: Dict[str, str] = {
        "darwin": f"codeql-bundle-osx64.tar.gz",
        "linux": f"codeql-bundle-linux64.tar.gz",
    }

    return platform_map.get(system)


async def download_file(url: str, dest_path: Path) -> None:
    """Download a file from a URL to a destination path."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)


def extract_codeql_bundle(archive_path: Path, dest_dir: Path) -> bool:
    """Extract the entire CodeQL bundle to destination directory

    Args:
        archive_path: Path to the archive file
        dest_dir: Destination directory for extraction

    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extract all contents
            tar.extractall(path=dest_dir)
            return True
    except Exception as e:
        print(f"Error extracting CodeQL bundle: {e}")
        return False


async def install_codeql(install_path: Path) -> Optional[Path]:
    """Main installation function

    Args:
        install_path: Path where CodeQL should be installed (e.g., /opt/codeql)

    Returns:
        Optional[Path]: Path to the installed CodeQL CLI binary or None if failed
    """
    # Get appropriate CodeQL filename for current platform
    codeql_file = get_platform_info()
    if not codeql_file:
        print("Error: Unsupported platform")
        print(f"System: {platform.system()}")
        print("CodeQL installer only supports Linux and macOS")
        return None

    # Create install directory if it doesn't exist
    install_path.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        try:
            # Download the file
            download_url = f"{CODEQL_BASE_URL}/{codeql_file}"
            archive_path = Path(temp_dir) / codeql_file

            print(f"Downloading CodeQL from: {download_url}")
            await download_file(download_url, archive_path)
            print("Download completed")

            # Extract the bundle to temp directory
            temp_extract_dir = Path(temp_dir) / "extracted"
            temp_extract_dir.mkdir()

            print("Extracting CodeQL bundle...")
            if not extract_codeql_bundle(archive_path, temp_extract_dir):
                print("Error: Could not extract CodeQL bundle")
                return None

            # Find the codeql directory in extracted contents
            # The bundle typically extracts to a 'codeql' directory
            codeql_dir = temp_extract_dir / "codeql"
            if not codeql_dir.exists():
                # Look for any directory that might contain CodeQL
                extracted_dirs = [d for d in temp_extract_dir.iterdir() if d.is_dir()]
                if extracted_dirs:
                    codeql_dir = extracted_dirs[0]
                else:
                    print("Error: Could not find CodeQL directory in extracted bundle")
                    return None

            # Verify that the codeql binary exists
            binary_name = "codeql.exe" if os.name == "nt" else "codeql"
            codeql_binary = codeql_dir / binary_name
            if not codeql_binary.exists():
                print(f"Error: Could not find {binary_name} in the extracted bundle")
                return None

            # Remove existing installation if it exists
            if install_path.exists():
                print(f"Removing existing CodeQL installation at {install_path}")
                shutil.rmtree(install_path)

            # Move the entire codeql directory to the install location
            shutil.move(str(codeql_dir), str(install_path))

            # Set executable permissions for the binary (Unix-like systems)
            final_binary = install_path / binary_name
            if os.name != "nt":
                os.chmod(final_binary, 0o755)

            print("Installation complete!")
            print(f"CodeQL has been installed to: {install_path}")
            print(f"CodeQL CLI binary: {final_binary}")
            print(
                f"To use CodeQL, add {install_path} to your PATH or use the full path: {final_binary}"
            )

            return final_binary

        except httpx.HTTPError as e:
            print(f"Download failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install CodeQL CLI")
    parser.add_argument(
        "--install-path",
        type=Path,
        default=Path("/opt/codeql"),
        help="Installation path for CodeQL (default: /opt/codeql)",
    )

    args = parser.parse_args()

    # Check if we have write permissions to the install path
    try:
        args.install_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(args.install_path.parent, os.W_OK):
            print(f"Error: No write permission to {args.install_path.parent}")
            print(
                "You may need to run this script with sudo or choose a different install path"
            )
            sys.exit(1)
    except Exception as e:
        print(f"Error checking install path: {e}")
        sys.exit(1)

    installed_path = asyncio.run(install_codeql(args.install_path))
    if installed_path:
        print(f"Successfully installed CodeQL CLI at: {installed_path}")
    else:
        print("Failed to install CodeQL")
        sys.exit(1)
