from pathlib import Path
import os
def get_data_dir():
    """Returns the path to the data directory."""
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        return Path(data_dir)
    return Path(__file__).parent.parent.parent / ".data"

def get_cache_dir():
    cache_dir = os.getenv("CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path(__file__).parent.parent.parent / ".cache" 
