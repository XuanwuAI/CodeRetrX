from pathlib import Path
import os


def get_data_dir():
    """Returns the path to the data directory."""
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        data_dir = Path(data_dir)
    data_dir = Path(__file__).parent.parent.parent / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_cache_dir():
    cache_dir = os.getenv("CACHE_DIR")
    if cache_dir:
        cache_dir = Path(cache_dir)
    cache_dir = Path(__file__).parent.parent.parent / ".cache"
    (cache_dir / "llm").mkdir(parents=True, exist_ok=True)
    (cache_dir / "embedding").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
