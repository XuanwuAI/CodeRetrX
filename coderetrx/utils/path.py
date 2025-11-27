from pathlib import Path
from coderetrx.utils.git import get_repo_id
import os


def get_data_dir():
    """Returns the path to the data directory."""
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = Path(__file__).parent.parent.parent / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_repo_path(repo_url):
    repo_path = get_data_dir() / "repos" / get_repo_id(repo_url)
    return repo_path

def get_cache_dir():
    cache_dir = os.getenv("CACHE_DIR")
    if cache_dir:
        cache_dir = Path(cache_dir)
    else:
        cache_dir = Path(__file__).parent.parent.parent / ".cache"
    (cache_dir / "llm").mkdir(parents=True, exist_ok=True)
    (cache_dir / "embedding").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def safe_join(path1: str | Path, path2: str | Path) -> Path:
    if isinstance(path1, str):
        path1 = Path(path1)
    if isinstance(path2, str):
        path2 = Path(path2)
    result = path1 / path2
    # Basic check: disallow absolute path2 that would ignore path1
    if not result.is_relative_to(path1):
        raise ValueError(f"Path {path2} is not relative to the base directory {path1}")
    # Symlink-aware check: resolve both sides (non-strict to allow non-existing leaf)
    resolved_base = path1.resolve(strict=False)
    resolved_result = (path1 / path2).resolve(strict=False)
    if not resolved_result.is_relative_to(resolved_base):
        raise ValueError(
            f"Path {path2} escapes base directory {path1} via symlink resolution"
        )
    return result
