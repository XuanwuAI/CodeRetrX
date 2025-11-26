# CodeRetrX: Code Analysis and Semantic Retrieval Library with Smart Strategies 

CodeRetrX is an AI-powered code analysis toolkit designed for agent-driven software engineering and security research. It simplifies building agentic bug-hunting environments by providing two core components: (1) a suite of code analysis tools exposed via the **Model Context Protocol (MCP)**, and (2) a high-recall, low-cost **semantic retrieval API**.

## 🏗️ System Architecture

CodeRetrX processes repositories through a multi-stage pipeline: **Static Analysis** → **Code Retrieval** (**Coarse Recall** → **Refined Recall**), and exposed as agentic tools.

### 📝 Static Analysis 

Static analysis extracts metadata, including file structures, dependencies, symbols (e.g., classes, functions), and other relevant details. The results of static analysis serve as the foundation for code retrieval and can also be utilized independently for tasks like repository exploration and visualization.

#### Core Processes

- **Code Parsing**: Uses tree-sitter to parse source code into abstract syntax trees across 10 programming languages (javascript, typescript, python, rust, c, cpp, csharp, go, elixir, java)
- **Symbol Extraction**: Extracts functions, classes with hierarchical relationships and dependency
- **Content Structuring**: Breaks code into semantic chunks (definitions, references, imports) with keyword extraction
- **Search Infrastructure**: Integrates ripgrep for fast pattern matching and prepares data for vector similarity search

### 🎯 Code Retrieval 

The code retrieval process operates in two stages recalling: prioritizes high recall in the coarse stage (retrieving as many relevant code snippets as possible at minimal cost) and high precision in the refined stage (eliminating false positives), ensuring effective code analysis and security research.

#### Coarse Recall Stage

The coarse recall focuses on maximizing recall while keeping costs low, aiming to retrieve as many potentially relevant code snippets as possible. This stage prioritizes recall over precision, allowing for false positives to ensure comprehensive coverage. Techniques such as vector-based retrieval, LLM-driven semantic analysis, and adaptive algorithms are employed to achieve efficient, large-scale filtering. The `line_per_symbol` method, used in our paper, is our best-performing approach and is chosen for optimal performance.

For detailed information about available strategies and their performance characteristics, see [STRATEGIES.md](STRATEGIES.md). See our [experimental results](#-experiments) for performance benchmarks.

#### Refined Recall Stage

The refined recall stage targets precision by removing the false positives introduced in the coarse stage. It applies a high-precision, strict-validation strategy to isolate truly relevant code snippets. This is achieved through advanced LLM-based semantic analysis with stricter filtering criteria, combined with optional secondary validation using enhanced models to re-evaluate and refine the results.

### 🔌 MCP Toolkit

#### Tools

CodeRetrX ships with a small set of repo-scoped MCP tools defined in `coderetrx/tools` and auto-registered via `coderetrx.tools.__init__.py`. These tools operate directly on the checked-out repository (ripgrep + raw file reads) rather than precomputed parser/CodeQL metadata. A `list_tools` call will return:

| Tool name | Purpose | Key arguments |
| --- | --- | --- |
| `list_dir` | Render a directory tree with file/dir markers and sizes for quick repo orientation | `directory_path`, `limit` |
| `find_file_by_name` | Find files or folders under a path (ripgrep glob behind the scenes) | `dir_path`, `pattern` |
| `keyword_search` | High-speed ripgrep search with regex support, file filters, and optional content inlining | `dir_path`, `query`, `query_with_regexp`, `glob_pattern_includes`, `glob_pattern_excludes`, `case_insensitive`, `include_content` |
| `get_reference` | Locate direct references to a symbol (case-sensitive word boundary search) | `symbol_name` |
| `view_file` | Read a file or slice by line range (0-indexed, with safety limits on range length) | `file_path`, `start_line`, `end_line` |

Example `call_tool` payloads:

```json
{"name":"list_dir","arguments":{"directory_path":"/","limit":120}}
{"name":"keyword_search","arguments":{"dir_path":"/src","query":"DatabaseClient","query_with_regexp":false,"glob_pattern_includes":"*.py","glob_pattern_excludes":"","case_insensitive":true,"include_content":true}}
{"name":"view_file","arguments":{"file_path":"/src/app/main.py","start_line":0,"end_line":120}}
```

#### MCP

The MCP server in `coderetrx/tools/mcp_server.py` exposes `list_tools` and `call_tool` handlers over stdio or SSE. The server instantiates the tool registry above on a per-repo basis and returns results as `mcp.types.TextContent`. Each call shares the same cloned repository on disk; the server does not pre-run tree-sitter or CodeQL.

Quick flow:

1) Launch the server against a repo: `uv run coderetrx.tools.mcp_server /path/to/repo` (or with `--use-sse`).
2) `list_tools` shows the five built-ins (`list_dir`, `find_file_by_name`, `keyword_search`, `get_reference`, `view_file`).
3) `call_tool` invokes them with JSON args (see examples in the tools section). Calls work over the same repo path on disk.


## 🛠️ Setup & Installation

### Prerequisites

- **uv** package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)), with python 3.12+
- **LLM provider API keys** (OpenAI, Anthropic, etc.)
- **Vector embedding model keys** (for similarity search)
- **Vector database**: defaults to Qdrant; point `QDRANT_BASE_URL` to a running instance or set `VECTOR_DB_PROVIDER=chroma` (install the `chromadb` extra). If you only need the MCP tools/ripgrep searches, disable embeddings via env flags (e.g., `SYMBOL_NAME_EMBEDDING=false` and `SYMBOL_CODELINE_EMBEDDING=false`).

### Local Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/XuanwuAI/CodeRetrX.git
   cd CodeRetrX
   ```

2. **Install Python dependencies:**

   ```bash
   uv sync --all-extras
   ```

3. **Configure environment variables:**

   ```bash
   EMBEDDING_BASE_URL=https://your-embedding-service.com
   EMBEDDING_API_KEY=your_embedding_api_key_here
   OPENAI_BASE_URL=https://your-key-service.com/
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🖥️ Usage

### 🤖 MCP Tooling

Start the MCP server against any repository (stdio transport by default):

```python
uv run coderetrx.tools.mcp_server /path/to/repo
```

Use SSE (helpful for hosted/tunneled setups):

```python
uv run coderetrx.tools.mcp_server /path/to/repo --use-sse --host 127.0.0.1 --port 10001
```

Once running, any MCP-compliant LLM agent can:

1. Call `list_tools` to discover built-in tools (`list_dir`, `find_file_by_name`, `keyword_search`, `get_reference`, `view_file`).
2. Invoke `call_tool` with JSON arguments, e.g. `{"name":"keyword_search","arguments":{"dir_path":"/","query":"DatabaseClient","query_with_regexp":false,"glob_pattern_includes":"","glob_pattern_excludes":"","case_insensitive":true,"include_content":true}}`.
3. Stream responses over stdio or connect to the SSE endpoint (`/sse`) you exposed above.

Each tool instance is scoped to the `repo_url` you passed when launching the server, so calls reuse the same cloned repository path on disk.

### 📦 Retrieval API

```python
from pathlib import Path
from coderetrx.retrieval import coderetrx_filter, CodebaseFactory

# Initialize codebase
codebase = CodebaseFactory.new("repo_name", Path("/path/to/your/repo"))

# Search for relevant code
elements, llm_results = await coderetrx_filter(
    codebase=codebase,
    prompt="your_filter_prompt",
    subdirs_or_files=["src/"],
    target_type="symbol_content",
    coarse_recall_strategy="line_per_symbol"
)

# Process results
for element in elements:
    print(f"Found: {element.name} in {element.file.path}")
```

For detailed usage examples and advanced configurations, see [USAGE.md](USAGE.md).

### 🚀 Quick Start

Run the retrieval script for a quick demonstration:

```bash
uv run -m scripts.code_retriever
```

Analyze and compare results from different strategies:

```bash
uv run -m scripts.analyze_code_reports
```

This tool provides comprehensive evaluation capabilities, including **coverage analysis** to compare how many issues each strategy finds against the ground truth, **cost comparison** to assess token usage and LLM costs across strategies, and **performance metrics** to analyse the overall effectiveness of different approaches.

## 🧪 Experiments

We conducted comprehensive experiments on large-scale benchmarks across multiple programming languages and repository sizes to validate the effectiveness of our code retrieval strategies. Our benchmarking shows that the `line_per_symbol` strategy achieves over 90% recall with approximately 25% of the computational cost.

![Figure 1: Recall Rate Comparisons across languages and repository sizes](bench/recall_rate_comparison.png)
![Table 1: Effectiveness and Efficiency Comparison](bench/effectiveness_efficiency_comparison.png)

For detailed experiment setup, methodology, and results, see [bench/EXPERIMENTS.md](bench/EXPERIMENTS.md).


## 📚 Extras

- `stats`: for codebase statistics
- `cli`: for command-line interface tools
