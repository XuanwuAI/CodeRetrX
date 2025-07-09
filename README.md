# Code Analysis Library for Security Research

An AI-powered code analysis library that combines **static analysis** with **LLM-based code retrieval strategies** to perform intelligent code search, bug detection, and security vulnerability analysis.

## 🏗️ System Architecture

CodeLib processes repositories through a multi-stage pipeline: **Static Analysis** → **Code Retrieval** (**Coarse Filter** → **Refined Filter**)

### 📝 Static Analysis Stage (`codelib/static/`)

The static analysis stage transforms unstructured source code into searchable, structured representations that enable both fast pattern matching and semantic understanding for security research.

#### Core processes

- **Code Parsing**: Uses tree-sitter to parse source code into abstract syntax trees across 9 programming languages
- **Symbol Extraction**: Extracts functions, classes with hierarchical relationships and dependency
- **Content Structuring**: Breaks code into semantic chunks (definitions, references, imports) with keyword extraction
- **Search Infrastructure**: Integrates ripgrep for fast pattern matching and prepares data for vector similarity search

### 🎯 Code Retrieval Stage (`codelib/retrieval` & `codelib/impl` )

The two-stage filtering approach balances recall (finding all relevant code) with precision (avoiding false positives) for effective code analysis and security research.

#### Coarse Filter Stage

To achieve efficient and comprehensive filtering, the coarse filter employs a high-recall, low-cost strategy to identify potentially relevant code elements. This is accomplished by utilizing vector-based retrieval, LLM-driven semantic analysis, and adaptive algorithm.

#### Refined Filter Stage

To improve precision and reduce false positives, the refined filter employs a high-precision, strict-validation strategy to identify truly relevant code elements. This is accomplished by utilizing stronger LLM-based semantic analysis with stricter filtering criteria, alongside optional secondary validation that re-evaluates the primary filter's results using better models.

## 🚀 Setup & Installation

### Prerequisites

- **uv** package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)), with python 3.12+
- **LLM provider API keys** (OpenAI, Anthropic, etc.)
- **Vector embedding model keys** (for similarity search)

### Local Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SparkSecurity/codelib
   cd codelib
   ```

2. **Install Python dependencies:**

   ```bash
   uv sync --all-extras
   ```

3. **Configure environment:**

   ```bash
   EMBEDDING_BASE_URL=https://your-embedding-service.com
   EMBEDDING_API_KEY=your_embedding_api_key_here
   OPENAI_BASE_URL=https://your-key-service.com/
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🖥️ Usage

### Programmatic API

CodeLib provides a powerful programmatic interface through the `coderecx_filter` and `llm_traversal_filter` API, which enables flexible code analysis and retrieval across different search strategies and filtering modes.

The core API combines multiple search strategies with two-stage filtering to balance recall and precision:

#### Using coderecx_filter (Fast & Efficient)

The `coderecx_filter` function provides fast, efficient code retrieval with configurable search strategies:

```python
from codelib.retrieval import coderecx_filter
from codelib.impl.default import CodebaseFactory

# Initialize codebase
codebase = CodebaseFactory.new("repo_name", "/path/to/your/repo")

# Basic symbol search
elements, llm_results = await coderecx_filter(
    codebase=codebase,
    prompt="your_filter_prompt",
    subdirs_or_files=["src/"],
    granularity="symbol_content",
    coarse_recall_strategy="symbol"
)

# Process results
for element in elements:
    print(f"Found: {element.name} in {element.file.path}")
```

**Advanced Configuration with coderecx_filter:**

```python
from codelib.retrieval import coderecx_filter
from codelib.retrieval.strategies import CodeRecallSettings

# Configure advanced settings
settings = CodeRecallSettings(
    llm_primary_recall_model_id="google/gemini-2.5-flash-lite-preview-06-17",
    llm_secondary_recall_model_id="openai/gpt-4.1-mini"
)

# Cost-efficient and complete search with line mode and secondary recall
elements, llm_results = await coderecx_filter(
    codebase=codebase,
    prompt="your_filter_prompt", 
    subdirs_or_files=["src/", "lib/"],
    granularity="symbol_content",
    coarse_recall_strategy="line",
    settings=settings,
    enable_secondary_recall=True
)
```

#### Using llm_traversal_filter (Ground Truth & Maximum Accuracy)

The `llm_traversal_filter` function provides the most comprehensive and accurate analysis, ideal for establishing ground truth:

```python
from codelib.retrieval import llm_traversal_filter

# Ground truth search - most comprehensive and accurate
elements, llm_results = await llm_traversal_filter(
    codebase=codebase,
    prompt="your_filter_prompt",
    subdirs_or_files=["src/", "lib/"],
    granularity="symbol_content",
    settings=settings
)
```


#### Search Strategies

**coderecx_filter** supports `filename`, `symbol`, `line`, and `auto` strategies for different speed/accuracy tradeoffs. **llm_traversal_filter** uses full LLM processing for maximum accuracy.

[See detailed strategy comparison](#-search-strategies)

#### Granularity Options

- **`symbol_content`**: Symbol code content (functions, classes, dependencies)
- **`class_content`**: Class code content  
- **`function_content`**: Function code content  
- **`dependency_name`**: Dependency names  
- **`keyword`**: Keywords/key phrases from code 

#### Settings Configuration

The `CodeRecallSettings` class allows fine-tuning of the search behavior:

```python
settings = CodeRecallSettings(
    llm_primary_recall_model_id="...",      # Model for initial filtering
    llm_secondary_recall_model_id="...",    # Model for refinement
    llm_selector_strategy_model_id="...",   # Model for strategy selection
    llm_call_mode="traditional"             # LLM call mode (default: "function_call")
)
```

#### Working with Results

```python
# Process different types of results
for element in elements:
    if hasattr(element, 'name'):  # Symbol
        print(f"Symbol: {element.name} in {element.file.path}")
        print(f"Content: {element.chunk.content}")
    elif hasattr(element, 'path'):  # File
        print(f"File: {element.path}")
    elif hasattr(element, 'text'):  # Keyword
        print(f"Keyword: {element.text}")
        
# Access LLM analysis results
for result in llm_results:
    print(f"Analysis: {result.analysis}")
    print(f"Confidence: {result.confidence}")
```

### Quick Start

Get help with available commands:

```bash
uv run -m scripts.code_retriever --help
```

For optimal results, use the line strategy with function call:

```bash
uv run -m scripts.code_retriever -f --mode auto
```

Analyze and compare results from different strategies:

```bash
uv run -m scripts.analyze_code_reports
```

This tool provides comprehensive evaluation capabilities, including **coverage analysis** to compare how many issues each strategy finds against the ground truth, **cost comparison** to assess token usage and LLM costs across strategies, and **performance metrics** to analyse the overall effectiveness of different approaches.

## 🔍 Search Strategies

CodeLib provides multiple search strategies optimized for different use cases:

### For coderecx_filter (coarse_recall_strategy parameter)

- **`filename`**: Fastest filename-based filtering, ideal for file discovery and structural queries
- **`symbol`**: Balanced symbol vector filtering, good for function/class search with moderate accuracy
- **`line`**: High-accuracy line-level vector search with LLM judgment, best for complex analysis
- **`auto`**: LLM automatically selects optimal strategy based on query complexity

### For llm_traversal_filter

- Full LLM processing for maximum accuracy and comprehensive analysis - no strategy parameter needed

### Performance Characteristics

- **Speed**: `filename` > `symbol` > `auto` > `line` > `precise`
- **Accuracy**: `precise` > `line` > `auto` > `symbol` > `filename`
- **Cost**: `filename` < `line` < `auto` < `symbol` < `precise`

Use `filename` for structural queries, `symbol` for API search, `line` for specific code related analysis, `auto` for general purpose, and `precise` for ground truth.

## 📚 Extras

- `stats`: for codebase statistics
- `builtin-impl`: for builtin LLM code retrieval tools
- `cli`: for command-line interface tools

e.g. specify `codelib[builtin-impl]` in `pyproject.toml` to have builtin LLM code retrieval tools.