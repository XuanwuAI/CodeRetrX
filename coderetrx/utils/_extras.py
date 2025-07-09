import importlib


def require_extra(pkg_name: str, extra: str):
    """
    Raises an ImportError if pkg_name isn't importable, pointing users
    to install the given extra.
    """
    try:
        importlib.import_module(pkg_name)
    except ImportError as e:
        raise ImportError(
            f"This feature requires the '{extra}' extra. "
            f"Install with: uv add coderetrx[{extra}]"
        ) from e
