import os
import platform
import sys
import asyncio
import shutil
import tarfile
from tempfile import TemporaryDirectory
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict

import httpx

# Ripgrep release information
RG_VERSION = "14.1.1"
RG_REPO = "BurntSushi/ripgrep"
RG_BASE_URL = f"https://github.com/{RG_REPO}/releases/download/{RG_VERSION}"


def get_platform_info() -> Optional[str]:
    """Get platform information and return the corresponding download filename

    Returns:
        Optional[str]: Matching platform filename or None if unsupported
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map platform to ripgrep release filename
    platform_map: Dict[Tuple[str, str], str] = {
        ("darwin", "x86_64"): f"ripgrep-{RG_VERSION}-x86_64-apple-darwin.tar.gz",
        ("darwin", "arm64"): f"ripgrep-{RG_VERSION}-aarch64-apple-darwin.tar.gz",
        ("linux", "x86_64"): f"ripgrep-{RG_VERSION}-x86_64-unknown-linux-musl.tar.gz",
        ("linux", "i686"): f"ripgrep-{RG_VERSION}-i686-unknown-linux-gnu.tar.gz",
        ("linux", "aarch64"): f"ripgrep-{RG_VERSION}-aarch64-unknown-linux-gnu.tar.gz",
        (
            "linux",
            "armv7l",
        ): f"ripgrep-{RG_VERSION}-armv7-unknown-linux-gnueabihf.tar.gz",
        ("windows", "amd64"): f"ripgrep-{RG_VERSION}-x86_64-pc-windows-msvc.zip",
        ("windows", "x86"): f"ripgrep-{RG_VERSION}-i686-pc-windows-msvc.zip",
    }

    return platform_map.get((system, machine))


async def download_file(url: str, dest_path: Path) -> None:
    """Download a file from a URL to a destination path."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)


def extract_single_file(archive_path: Path, filename: str, dest: Path) -> bool:
    """Extract a single file from archive without extracting everything

    Args:
        archive_path: Path to the archive file
        filename: Name of the file to extract
        dest: Destination path for the extracted file

    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    try:
        if (
            archive_path.suffixes[-2:] == [".tar", ".gz"]
            or archive_path.suffix[-1] == ".tgz"
        ):
            with tarfile.open(archive_path, "r:gz") as tar:
                member = next(
                    (m for m in tar.getmembers() if m.name.endswith(f"/{filename}")),
                    None,
                )
                if member:
                    extracted_file = tar.extractfile(member)
                    if extracted_file is None:
                        return False
                    with extracted_file as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return True
        elif archive_path.suffix[-1] == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                member = next(
                    (m for m in zip_ref.infolist() if m.filename.endswith(filename)),
                    None,
                )
                if member:
                    with zip_ref.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return True
        return False
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        return False


async def install_rg(install_path: Path) -> Optional[Path]:
    """Main installation function

    Returns:
        Optional[Path]: Path to the installed rg binary or None if failed
    """
    # Get appropriate ripgrep filename for current platform
    rg_file = get_platform_info()
    if not rg_file:
        print("Error: Unsupported platform")
        print(f"System: {platform.system()}, Machine: {platform.machine()}")
        return None

    install_path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp_dir:  # ty: ignore[no-matching-overload]
        try:
            # Download the file
            download_url = f"{RG_BASE_URL}/{rg_file}"
            archive_path = Path(temp_dir) / rg_file
            await download_file(download_url, archive_path)

            # Determine binary filename based on platform
            binary_name = "rg.exe" if os.name == "nt" else "rg"
            temp_binary = Path(temp_dir) / binary_name

            # Extract just the binary file
            if not extract_single_file(archive_path, binary_name, temp_binary):
                print(f"Error: Could not find {binary_name} in the downloaded archive")
                return None

            # Set executable permissions (Unix-like systems)
            if os.name != "nt":
                os.chmod(temp_binary, 0o755)

            # Install to destination
            dest_path = install_path / binary_name

            # Move the binary to final location
            shutil.move(temp_binary, dest_path)

            print("Installation complete!")
            print(f"ripgrep has been installed to: {dest_path}")
            print(f"{install_path}")

            return dest_path

        except httpx.HTTPError as e:
            print(f"Download failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    return None


if __name__ == "__main__":
    from pathlib import Path

    path = Path(".")
    installed_path = asyncio.run(install_rg(path))
    if installed_path:
        print(f"Successfully installed rg at: {installed_path}")
    else:
        print("Failed to install ripgrep")
        sys.exit(1)
