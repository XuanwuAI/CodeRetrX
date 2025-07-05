# API Reference

## Table of Contents

- [Core Classes](#core-classes)
  - [Codebase](#codebase)
  - [SmartCodebase](#smartcodebase)
  - [File](#file)
  - [Symbol](#symbol)
- [Static Analysis](#static-analysis)
  - [Ripgrep Functions](#ripgrep-functions)
  - [Search Results](#search-results)
- [LLM Retrieval](#llm-retrieval)
  - [Multi-Strategy Functions](#multi-strategy-functions)
  - [Topic Extraction](#topic-extraction)
- [Data Models](#data-models)
- [Utilities](#utilities)

## Core Classes

### Codebase

The main class for representing and analyzing codebases.

#### Class Definition

```python
class Codebase:
    """
    Represents a codebase with files, symbols, and dependencies.
    Provides methods for analysis and search operations.
    """
```

#### Methods

##### `from_path(path: str | Path) -> Codebase`

Load a codebase from a filesystem path.

**Parameters:**
- `path`: Path to the codebase directory

**Returns:**
- `Codebase` instance

**Example:**
```python
from codelib import Codebase

codebase = Codebase.from_path("/path/to/project")
```

##### `search(pattern: str, **kwargs) -> List[GrepMatchResult]`

Search for code patterns using ripgrep.

**Parameters:**
- `pattern`: Regular expression pattern to search for
- `**kwargs`: Additional options for ripgrep

**Returns:**
- List of `GrepMatchResult` objects

**Example:**
```python
# Search for async functions
results = codebase.search(r"async def \w+")

# Search with file type filter
results = codebase.search(r"class \w+", file_types=["*.py"])
```

##### `find_files(pattern: str = "*") -> List[File]`

Find files matching a glob pattern.

**Parameters:**
- `pattern`: Glob pattern for file matching

**Returns:**
- List of `File` objects

**Example:**
```python
# Find all Python files
python_files = codebase.find_files("*.py")

# Find test files
test_files = codebase.find_files("test_*.py")
```

##### `get_symbols() -> List[Symbol]`

Extract all symbols from the codebase.

**Returns:**
- List of `Symbol` objects

**Example:**
```python
symbols = codebase.get_symbols()
functions = [s for s in symbols if s.type == "function"]
```

##### `get_dependencies() -> List[Dependency]`

Get dependency information for the codebase.

**Returns:**
- List of `Dependency` objects

**Example:**
```python
deps = codebase.get_dependencies()
external_deps = [d for d in deps if d.is_external]
```

#### Properties

- `files: List[File]` - List of all files in the codebase
- `symbols: List[Symbol]` - List of all symbols
- `keywords: List[Keyword]` - List of all keywords
- `dependencies: List[Dependency]` - List of all dependencies
- `path: Path` - Root path of the codebase

### SmartCodebase

LLM-enhanced codebase analysis capabilities.

#### Class Definition

```python
class SmartCodebase:
    """
    Enhanced codebase analysis using LLM capabilities.
    Provides semantic search and intelligent filtering.
    """
```

#### Methods

##### `from_path(path: str | Path, **kwargs) -> SmartCodebase`

Create a SmartCodebase instance from a filesystem path.

**Parameters:**
- `path`: Path to the codebase directory
- `**kwargs`: Additional configuration options

**Returns:**
- `SmartCodebase` instance

**Example:**
```python
from codelib.retrieval import SmartCodebase

smart_codebase = SmartCodebase.from_path(
    "/path/to/project",
    llm_model="gpt-4",
    chunk_size=1000
)
```

##### `search(query: str, strategy: str = "hybrid") -> List[CodeElement]`

Perform intelligent code search using LLM.

**Parameters:**
- `query`: Natural language search query
- `strategy`: Search strategy ("semantic", "keyword", "hybrid")

**Returns:**
- List of `CodeElement` objects

**Example:**
```python
# Semantic search
results = smart_codebase.search(
    "functions that handle user authentication",
    strategy="semantic"
)

# Hybrid search
results = smart_codebase.search(
    "error handling in API endpoints",
    strategy="hybrid"
)
```

##### `extract_topics(limit: int = 10) -> List[str]`

Extract main topics from the codebase.

**Parameters:**
- `limit`: Maximum number of topics to return

**Returns:**
- List of topic strings

**Example:**
```python
topics = smart_codebase.extract_topics(limit=5)
print("Main topics:", topics)
```

##### `filter_by_relevance(query: str, results: List[CodeElement]) -> List[CodeElement]`

Filter results by relevance to a query.

**Parameters:**
- `query`: Relevance query
- `results`: List of code elements to filter

**Returns:**
- Filtered list of `CodeElement` objects

**Example:**
```python
filtered = smart_codebase.filter_by_relevance(
    "security vulnerabilities",
    all_results
)
```

## Static Analysis

### Ripgrep Functions

#### `ripgrep_search(path: str, pattern: str, **kwargs) -> List[GrepMatchResult]`

Fast text search using ripgrep.

**Parameters:**
- `path`: Directory to search in
- `pattern`: Regular expression pattern
- `file_types`: List of file extensions to include (optional)
- `ignore_case`: Whether to ignore case (optional)
- `context`: Number of context lines (optional)

**Returns:**
- List of `GrepMatchResult` objects

**Example:**
```python
from codelib.static import ripgrep_search

results = ripgrep_search(
    "/path/to/project",
    r"TODO|FIXME",
    file_types=["*.py", "*.js"],
    ignore_case=True,
    context=2
)
```

#### `ripgrep_glob(path: str, pattern: str) -> List[str]`

Find files matching a glob pattern.

**Parameters:**
- `path`: Directory to search in
- `pattern`: Glob pattern

**Returns:**
- List of matching file paths

**Example:**
```python
from codelib.static import ripgrep_glob

config_files = ripgrep_glob("/path/to/project", "*.config.*")
```

#### `ripgrep_search_symbols(path: str, symbol_type: str) -> List[GrepMatchResult]`

Search for specific types of symbols.

**Parameters:**
- `path`: Directory to search in
- `symbol_type`: Type of symbol ("function", "class", "variable", etc.)

**Returns:**
- List of `GrepMatchResult` objects

**Example:**
```python
from codelib.static import ripgrep_search_symbols

classes = ripgrep_search_symbols("/path/to/project", "class")
```

### Search Results

#### `GrepMatchResult`

Represents a single search match result.

**Properties:**
- `file_path: str` - Path to the file containing the match
- `line_number: int` - Line number of the match
- `line_content: str` - Content of the matching line
- `match_start: int` - Start position of the match
- `match_end: int` - End position of the match
- `context_before: List[str]` - Lines before the match
- `context_after: List[str]` - Lines after the match

## LLM Retrieval

### Multi-Strategy Functions

#### `multi_strategy_code_mapping(results: List[Any], query: str) -> List[CodeElement]`

Apply multiple analysis strategies to code results.

**Parameters:**
- `results`: Input results to analyze
- `query`: Analysis query

**Returns:**
- List of `CodeElement` objects

**Example:**
```python
from codelib.retrieval import multi_strategy_code_mapping

enhanced_results = multi_strategy_code_mapping(
    static_results,
    "security-related code patterns"
)
```

#### `multi_strategy_code_filter(results: List[Any], query: str) -> List[CodeElement]`

Filter code results using multiple strategies.

**Parameters:**
- `results`: Input results to filter
- `query`: Filter query

**Returns:**
- Filtered list of `CodeElement` objects

**Example:**
```python
from codelib.retrieval import multi_strategy_code_filter

filtered = multi_strategy_code_filter(
    all_results,
    "remove test files and mock implementations"
)
```

### Topic Extraction

#### `TopicExtractor`

Extract topics and themes from code.

**Methods:**

##### `extract_from_codebase(codebase: Codebase) -> List[str]`

Extract topics from a codebase.

**Parameters:**
- `codebase`: Codebase to analyze

**Returns:**
- List of topic strings

**Example:**
```python
from codelib.retrieval import TopicExtractor

extractor = TopicExtractor()
topics = extractor.extract_from_codebase(codebase)
```

## Data Models

### File

Represents a file in the codebase.

**Properties:**
- `path: str` - File path
- `content: str` - File content
- `size: int` - File size in bytes
- `type: FileType` - File type enum
- `language: str` - Programming language
- `symbols: List[Symbol]` - Symbols in the file

### Symbol

Represents a code symbol (function, class, variable, etc.).

**Properties:**
- `name: str` - Symbol name
- `type: str` - Symbol type
- `file_path: str` - File containing the symbol
- `line_number: int` - Line number where defined
- `signature: str` - Symbol signature (if applicable)
- `docstring: str` - Associated documentation

### Keyword

Represents a keyword or important term.

**Properties:**
- `term: str` - The keyword term
- `frequency: int` - Frequency in codebase
- `context: List[str]` - Context where it appears

### Dependency

Represents a code dependency.

**Properties:**
- `name: str` - Dependency name
- `version: str` - Version specification
- `is_external: bool` - Whether it's an external dependency
- `file_path: str` - File where declared

## Utilities

### Configuration

#### Environment Variables

The library supports configuration through environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Default OpenAI model
- `OPENAI_API_BASE`: Custom OpenAI API base URL
- `CODELIB_CACHE_DIR`: Cache directory path

#### Configuration File

Create a `codelib.toml` file for project-specific configuration:

```toml
[codelib]
max_file_size = 1048576
ignore_patterns = [".git", "__pycache__"]

[codelib.llm]
model = "gpt-4"
temperature = 0.1
max_tokens = 2048

[codelib.static]
use_tree_sitter = true
extract_comments = true
```

### Error Handling

The library defines custom exceptions:

- `CodebaseError`: Base exception for codebase operations
- `SearchError`: Exception for search operations
- `LLMError`: Exception for LLM-related operations
- `ConfigurationError`: Exception for configuration issues

**Example:**
```python
from codelib.exceptions import CodebaseError

try:
    codebase = Codebase.from_path("/invalid/path")
except CodebaseError as e:
    print(f"Error loading codebase: {e}")
```

### Logging

The library uses Python's logging module. Configure logging in your application:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logger = logging.getLogger('codelib')
logger.setLevel(logging.INFO)
```

## Type Hints

The library provides comprehensive type hints. Import types as needed:

```python
from typing import List, Dict, Optional
from codelib.types import (
    CodeElement,
    SearchResult,
    AnalysisResult,
    LLMResponse
)
```

## Async Support

Some operations support async execution:

```python
import asyncio
from codelib.retrieval import SmartCodebase

async def analyze_codebase():
    smart_codebase = SmartCodebase.from_path("/path/to/project")
    results = await smart_codebase.search_async("async functions")
    return results

# Run async analysis
results = asyncio.run(analyze_codebase())
```