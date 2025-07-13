# CodeRetrX: Code Analysis and Semantic Retrieval Library with Smart Strategies 

An AI-powered code analysis library that combines **static analysis** with **LLM-based code retrieval strategies** to perform intelligent code search.

## 🏗️ System Architecture

CodeRetrX processes repositories through a multi-stage pipeline: **Static Analysis** → **Code Retrieval** (**Coarse Filter** → **Refined Filter**)

### 📝 Static Analysis 

Static analysis extracts metadata, including file structures, dependencies, symbols (e.g., classes, functions), and other relevant details. The results of static analysis serve as the foundation for code retrieval and can also be utilized independently for tasks like repository exploration and visualization.

#### Core processes

- **Code Parsing**: Uses tree-sitter to parse source code into abstract syntax trees across 9 programming languages
- **Symbol Extraction**: Extracts functions, classes with hierarchical relationships and dependency
- **Content Structuring**: Breaks code into semantic chunks (definitions, references, imports) with keyword extraction
- **Search Infrastructure**: Integrates ripgrep for fast pattern matching and prepares data for vector similarity search

### 🎯 Code Retrieval 

The code retrieval process operates in two stages filtering: prioritizes high recall in the coarse stage (retrieving as many relevant code snippets as possible at minimal cost) and high precision in the refined stage (eliminating false positives), ensuring effective code analysis and security research.

#### Coarse Filter Stage

The coarse filter focuses on maximizing recall while keeping costs low, aiming to retrieve as many potentially relevant code snippets as possible. This stage prioritizes recall over precision, allowing for false positives to ensure comprehensive coverage. Techniques such as vector-based retrieval, LLM-driven semantic analysis, and adaptive algorithms are employed to achieve efficient, large-scale filtering.

#### Refined Filter Stage

The refined filter targets precision by removing the false positives introduced in the coarse stage. It applies a high-precision, strict-validation strategy to isolate truly relevant code snippets. This is achieved through advanced LLM-based semantic analysis with stricter filtering criteria, combined with optional secondary validation using enhanced models to re-evaluate and refine the results from the coarse stage.

## 🚀 Setup & Installation

### Prerequisites

- **uv** package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)), with python 3.12+
- **LLM provider API keys** (OpenAI, Anthropic, etc.)
- **Vector embedding model keys** (for similarity search)

### Local Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SparkSecurity/coderetrx
   cd coderetrx
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

CodeRetrX provides a powerful programmatic interface through the `coderetrx_filter` and `llm_traversal_filter` API, which enables flexible code analysis and retrieval across different search strategies and filtering modes.

The coderetrx_filter implements stragties designed by us. It offers a cost-effective semantic recall approach for large-scale repositories, achieving approximately 80% recall with only about 20% of the resource consumption in practical tests(line_per_symbol strategy) — the larger the repository, the greater the savings.

The llm_traversal_filter provides the most comprehensive and accurate analysis, ideal for establishing ground truth. For small-scale repositories, this strategy can also be a good choise.

#### Using coderetrx_filter with symbol_name strategy 

```python
from pathlib import Path
from coderetrx.retrieval import coderetrx_filter
from coderetrx.impl.default import CodebaseFactory

# Initialize codebase
codebase = CodebaseFactory.new("repo_name", Path("/path/to/your/repo"))

# Basic symbol search
elements, llm_results = await coderetrx_filter(
    codebase=codebase,
    prompt="your_filter_prompt",
    subdirs_or_files=["src/"],
    target_type="symbol_content",
    coarse_recall_strategy="symbol_name"
)

# Process results
for element in elements:
    print(f"Found: {element.name} in {element.file.path}")
```

**Using coderetrx_filter with line_per_symbol strategy**

```python
from coderetrx.retrieval import coderetrx_filter
from coderetrx.retrieval.code_recall import CodeRecallSettings

# Configure advanced settings
settings = CodeRecallSettings(
    llm_primary_recall_model_id="google/gemini-2.5-flash-lite-preview-06-17",
    llm_secondary_recall_model_id="openai/gpt-4o-mini"
)

# Cost-efficient and complete search with line_per_symbol mode and secondary recall
elements, llm_results = await coderetrx_filter(
    codebase=codebase,
    prompt="your_filter_prompt", 
    subdirs_or_files=["src/", "lib/"],
    target_type="symbol_content",
    coarse_recall_strategy="line_per_symbol",
    settings=settings,
    enable_secondary_recall=True
)
```

#### Using llm_traversal_filter (Ground Truth & Maximum Accuracy)

```python
from coderetrx.retrieval import llm_traversal_filter

# Ground truth search - most comprehensive and accurate
elements, llm_results = await llm_traversal_filter(
    codebase=codebase,
    prompt="your_filter_prompt",
    subdirs_or_files=["src/", "lib/"],
    target_type="symbol_content",
    settings=settings
)
```


#### Coarse Recall Strategy 

**coderetrx_filter** supports `file_name`, `symbol_name`, `line_per_symbol`, and `auto` strategies for different speed/accuracy tradeoffs. **llm_traversal_filter** uses full LLM processing for maximum accuracy.

[See detailed strategy comparison](#-coarse-recall-strategies)

#### Target type Options

Target type defines the retrieval target, determining the type of code object to be recalled and returned. For example, if the target_type is set to class_content, the result will include the relevant classes whose content match the query. Below are the available target_type options:

- **`symbol_name`**:Matches symbols (e.g., functions, classes) whose **name** satisfies the query. 
- **`symbol_content`**: Matches symbols whose **entire code content** satisfies the query.
- **`leaf_symbol_name`**: Matches **leaf symbols** (symbols without child elements, such as methods) whose **name** satisfy the query.
- **`leaf_symbol_content`**: Matches **leaf symbols** whose **code content** satisfies the query.
- **`root_symbol_name`**: Matches **root symbols** (symbols without parent elements, such as top-level classes, functions) whose **name** satisfy the query.
- **`root_symbol_content`**: Matches **root symbols** whose **entire code content** satisfies the query.
- **`class_name`**: Matches **classes** whose **name** that satisfy the query.
- **`class_content`**: Matches **classes** whose **entire code content** satisfies the query.
- **`function_name`**: Matches **functions**  whose **name**  satisfy the query.
- **`function_content`**: Matches **functions** whose **entire code content** satisfies the query.
- **`dependency_name`**: Matches **dependency names** (e.g., imported libraries or modules) that satisfy the query.

Note: The coderetrx_filter only supports the xxx_content series of target_type, while the llm_traversal_filter supports all target_type options.

#### Settings Configuration

The `CodeRecallSettings` class allows fine-tuning of the search behavior:

```python
settings = CodeRecallSettings(
    llm_primary_recall_model_id="...",      # Model used for coarse recall and the primary recall in the refined stage.
    llm_secondary_recall_model_id="...",    # Model used for secondary recall in the refined stage. If set (not None), secondary recall will be enabled.
    llm_selector_strategy_model_id="...",   # Model used for strategy selection in "auto" mode during the coarse recall stage.
    llm_call_mode="function_call"           # LLM call mode. If set to "function_call", the LLM will return results as a function call (recommended for models supporting this feature). 
                                            # If set to "traditional", the LLM will return results in plain text format.
)
```

#### Working with Results

```python
# Process different types of results
for element in elements:
    if hasattr(element, 'name'):  # Symbol
        print(f"Symbol: {element.name} in {element.file.path}")
        print(f"Content: {element.chunk.code()}")
    elif hasattr(element, 'path'):  # File
        print(f"File: {element.path}")
    elif hasattr(element, 'content'):  # Keyword
        print(f"Keyword: {element.content}")
        
# Access LLM analysis results
for result in llm_results:
    print(f"Index: {result.index}")
    print(f"Reason: {result.reason}")
    print(f"Result: {result.result}")
```

### Quick Start

A simple usage example:

```bash
uv run -m scripts.example
```

Analyze and compare results from different strategies:

```bash
uv run -m scripts.analyze_code_reports
```

This tool provides comprehensive evaluation capabilities, including **coverage analysis** to compare how many issues each strategy finds against the ground truth, **cost comparison** to assess token usage and LLM costs across strategies, and **performance metrics** to analyse the overall effectiveness of different approaches.

## 🔍 Coarse Recall Strategies 

In the coarse recall stage, multiple strategies are used to efficiently retrieve potentially relevant code snippets at low cost. The retrieved results will then be further processed in the refined filter stage to improve precision and accuracy.

### Available Strategies 

- **`file_name`**: The fastest filtering strategy, ideal for coarse recall by directly determining relevance based on filenames. This approach is highly effective for cases where the query can be matched through filenames alone, such as "retrieve all configuration files." Its simplicity makes it extremely efficient for structural queries and file discovery.

- **`symbol_name`**: A filtering approach that focuses on symbol names (e.g., function or class names) during coarse recall. This method excels in cases like "retrieve functions implementing cryptographic algorithms," where relevance can be inferred directly from the symbol name. It balances recall and precision for targeted queries at the symbol level.

- **`line_per_symbol`**: A high-accuracy filtering strategy that leverages line-level vector search combined with LLM analysis. It identifies relevance by focusing on the most relevant code lines (`top-k`) within a function body. This method is particularly effective for complex cases where understanding the function’s implementation is necessary, such as "retrieve code implementing authentication logic." While powerful, it’s also versatile enough to handle most queries effectively. Our benchmarking shows that the algorithm achieves over 80% recall with approximately 20% of the computational cost, making it both precise and efficient.

- **`auto`**: Automatically selects the optimal filtering strategy based on query complexity, routing requests to the most appropriate method. For the majority of cases, `line_per_symbol` is a well-balanced and reliable choice, but `file_name` or `symbol_name` can also be explicitly used for scenarios where they excel.

### Performance Characteristics

- **Speed**: `file_name` > `symbol_name` > `auto` > `line_per_symbol` > `precise`
- **Accuracy**: `precise` > `line_per_symbol` > `auto` > `symbol_name` > `file_name`
- **Cost**: `file_name` < `line_per_symbol` < `auto` < `symbol_name` < `precise`

Use `file_name` for structural queries, `symbol_name` for API search, `line_per_symbol` for specific code related analysis, `auto` for general purpose, and `precise` for ground truth.

## 🧪 Experiments

We conducted comprehensive experiments on the *Ollama* repository to validate the effectiveness of our code retrieval strategies. The analysis demonstrates how **`coderetrx_filter`** performs across various bug types and complexity levels.

For detailed results, see: [Ollama Analysis by Bug Type](bench/Ollama_Analysis_by_Bug_Type.md)

## 📚 Extras

- `stats`: for codebase statistics
- `builtin-impl`: for builtin LLM code retrieval tools
- `cli`: for command-line interface tools

e.g. specify `coderetrx[builtin-impl]` in `pyproject.toml` to have builtin LLM code retrieval tools.