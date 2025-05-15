from functools import lru_cache
from os import PathLike
from typing import List, Literal, Optional
from pathlib import Path


IDXSupportedLanguage = Literal[
    "javascript",
    "typescript",
    "python",
    "rust",
    "c",
    "cpp",
    "csharp",
    "go",
    "elixir",
    "java",
]

IDXSupportedTag = Literal[
    "definition.function",
    "definition.method",
    "definition.class",
    "definition.interface",
    "definition.module",
    "definition.reexport",
    "reference.implementation",
    "reference.call",
    "reference.class",
    "import",
]

EXTENSION_MAP: dict[str, IDXSupportedLanguage] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rs": "rust",
    "c": "c",
    "cpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "ex": "elixir",
    "exs": "elixir",
    "java": "java",
}

DEP_FILES: List[str] = [
    # JS / TS
    "package.json",
    # Python
    "requirements.txt",
    "setup.py",
    "Pipfile",
    "pyproject.toml",
    # Rust
    "Cargo.toml",
    # C/CPP
    "Makefile",
    # Golang
    "go.mod",
    # Elixir
    "mix.exs",
    "pom.xml",
    # Java
    "build.gradle",
    "build.gradle.kts",
    "build.sbt",
    "build.gradle",
    "build.gradle.kts",
    "build.sbt",
]

BUILTIN_CRYPTO_LIBS: dict[IDXSupportedLanguage, List[str]] = {
    "javascript": ["crypto", "node:crypto", "webcrypto"],
    "typescript": ["crypto", "node:crypto", "webcrypto"],
    "python": ["hashlib", "hmac", "secrets", "ssl"],
    "rust": [],
    "c": [],
    "cpp": [],
    "csharp": ["System.Security.Cryptography"],
    "go": ["crypto"],
    "elixir": [":crypto"],
    "java": ["java.security", "javax.crypto"],
}

FUNCLIKE_TAGS: List[IDXSupportedTag] = [
    "definition.function",
    "definition.method",
]

OBJLIKE_TAGS: List[IDXSupportedTag] = [
    "definition.class",
    "definition.interface",
    "definition.module",
    "definition.reexport",
    "reference.implementation",
]

PRIMARY_TAGS: List[IDXSupportedTag] = [
    "definition.class",
    "definition.function",
    "definition.interface",
    "definition.method",
    "definition.module",
    # Special case: imports something and introduces a new symbol to the global pool
    "definition.reexport",
    "reference.implementation",
]

REFERENCE_TAGS: List[IDXSupportedTag] = [
    "reference.call",
    "reference.class",
]

IMPORT_TAGS: List[IDXSupportedTag] = [
    "import",
]


def get_extension(filepath: PathLike | str) -> str:
    return str(filepath).split(".")[-1].lower()


def is_sourcecode(filepath: PathLike | str) -> bool:
    extension = get_extension(filepath)
    return extension in EXTENSION_MAP


def is_dependency(filepath: PathLike | str) -> bool:
    return str(filepath).split("/")[-1] in DEP_FILES


def get_language(filepath: PathLike | str) -> Optional[IDXSupportedLanguage]:
    extension = get_extension(filepath)
    return EXTENSION_MAP.get(extension)


@lru_cache()
def get_query(
    language: IDXSupportedLanguage,
    query_type: Literal["tags", "tests", "fine_imports"] = "tags",
) -> str:
    scm_loc = Path(__file__).parent / "queries" / query_type / f"{language}.scm"
    if not scm_loc.exists():
        raise FileNotFoundError(
            f"Query file for {query_type} not found for language {language}"
        )
    with open(scm_loc, "r") as f:
        return f.read()
