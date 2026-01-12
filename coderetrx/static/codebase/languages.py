from functools import lru_cache
from os import PathLike
from typing import List, Literal, Optional
from pathlib import Path
import fnmatch


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
    "php",
    "objc",
]

IDXSupportedTag = Literal[
    "definition.function",
    "definition.type",
    "definition.method",
    "definition.class",
    "definition.interface",
    "definition.module",
    "definition.reexport",
    "definition.variable",
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
    "h": "c",
    "hpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "ex": "elixir",
    "exs": "elixir",
    "java": "java",
    "php": "php",
    "m": "objc",
}

BLOCKED_PATTERNS = ["*.min.js", "*_test.go"]

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
    "php": ["openssl", "hash", "sodium"],
    "objc": ["CommonCrypto", "Security"],
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
    "definition.type",
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

VARIABLE_TAGS: List[IDXSupportedTag] = [
    "definition.variable",
]


def get_extension(filepath: PathLike | str) -> str:
    return str(filepath).split(".")[-1].lower()


def is_blocked_file(filepath: PathLike | str) -> bool:
    return any(fnmatch.fnmatch(str(filepath), pattern) for pattern in BLOCKED_PATTERNS)


def is_sourcecode(filepath: PathLike | str) -> bool:
    if is_blocked_file(filepath):
        return False
    extension = get_extension(filepath)
    return extension in EXTENSION_MAP


def is_dependency(filepath: PathLike | str) -> bool:
    return str(filepath).split("/")[-1] in DEP_FILES


def get_language(filepath: PathLike | str) -> Optional[IDXSupportedLanguage]:
    extension = get_extension(filepath)
    return EXTENSION_MAP.get(extension)
