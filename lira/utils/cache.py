from pathlib import Path


def get_cache_dir():
    """Returns the path to the project's cache directory."""
    cache_dir = Path.home() / ".cache" / "lira"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
