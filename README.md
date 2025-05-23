# Code Analysis Library for XuanWu Lab / TSpark Security

This library is structured as follows:

- Static Analysis Tools
  - Structures for representing codebases and code elements
  - Python wrappers for static analysis tools like `ripgrep`
- LLM Code Retrieval Tools

## Extras

- `stats`: for codebase statistics
- `builtin-impl`: for builtin LLM code retrieval tools

e.g. specify `codelib[builtin-impl]` in `pyproject.toml` to have builtin LLM code retrieval tools.

## Local development

For development related to extras, please run `uv sync --all-extras` to install all dependencies.
