# Code Analysis Library for Security Research

An AI-powered code analysis library that combines **static analysis** with **Large Language Model (LLM) capabilities** to perform intelligent code search, bug detection, and security vulnerability analysis.

## 🏗️ System Architecture

CodeLib processes repositories through a multi-stage pipeline: **Static Analysis** → **Coarse Filter** → **Refined Filter**

### 📝 Static Analysis Tools
- **Structures for representing codebases and code elements**
- **Python wrappers for static analysis tools like `ripgrep`**
- **Tree-sitter based parsing for multiple programming languages**
- **Symbol and dependency extraction**

### 🤖 Code Retrieval Tools
- **Semantic code search and understanding**
- **Vector similarity matching for intelligent retrieval**
- **Hybrid approaches combining multiple search strategies**

## 🔍 Search Strategies

CodeLib provides multiple search strategies optimized for different use cases:

### Strategy Types
- **`fast`**: Basic filename filtering and pattern matching - fastest execution
- **`balance`**: Keyword vector search combined with filename filtering - relatively good balance of speed and accuracy
- **`precise`**: Full LLM processing for maximum accuracy - most thorough but slower
- **`smart`**: LLM automatically determines the optimal strategy based on query context
- **`intelligent`**: Advanced line-level vector search with LLM judgment for fine-grained analysis
- **`adaptive`**: Dynamic strategy adjustment based on result quality and performance

### Recall Strategies (Internal)

The system uses multiple recall strategies internally:

#### Basic Strategies
- **`FILTER_FILENAME_BY_LLM`**: Uses LLM to filter files by path patterns and naming conventions
- **`FILTER_KEYWORD_BY_VECTOR`**: Vector similarity search on extracted keywords from code content
- **`FILTER_SYMBOL_BY_VECTOR`**: Vector similarity search on symbol names (functions, classes, variables)
- **`FILTER_SYMBOL_BY_LLM`**: LLM-based filtering of symbols based on semantic understanding

#### Hybrid Strategies
- **`FILTER_KEYWORD_BY_VECTOR_AND_LLM`**: Combines vector search with LLM refinement on keywords
- **`FILTER_SYMBOL_BY_VECTOR_AND_LLM`**: Combines vector search with LLM refinement on symbols

#### Advanced Strategies
- **`ADAPTIVE_FILTER_KEYWORD_BY_VECTOR_AND_LLM`**: Adaptive keyword search with dynamic result expansion/contraction
- **`ADAPTIVE_FILTER_SYMBOL_BY_VECTOR_AND_LLM`**: Adaptive symbol search with quality-based early termination
- **`INTELLIGENT_FILTER`**: Line-level vector recall with LLM judgment for fine-grained analysis

### LLM Call Modes
- **`traditional`**: Standard LLM text generation with structured output
- **`function_call`**: Structured LLM output using function calling

## 🚀 Key Features

### Multi-Modal Code Analysis
- **Static Analysis**: Tree-sitter based parsing and symbol extraction
- **LLM Integration**: Semantic understanding of code patterns
- **Vector Similarity Search**: Embedding-based content matching
- **Hybrid Approaches**: Combining multiple search strategies for optimal results

### Security-Focused Analysis
- **Pattern Recognition**: Automated detection of security-relevant code patterns
- **Vulnerability Analysis**: Comprehensive analysis of potential security risks

### Performance & Cost Optimization
- **Adaptive Algorithms**: Dynamic strategy adjustment based on result quality
- **Resource Management**: Intelligent batching and concurrent processing
- **Cost Tracking**: Real-time monitoring of LLM usage and costs

## 🚀 Setup & Installation

### Prerequisites

- **uv** package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)), with python 3.12+
- **LLM provider API keys** (OpenAI, Anthropic, etc.)
- **Vector embedding models** (for similarity search)

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

   **Required Environment Variables:**

   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## 🖥️ Usage

### Command Line Interface

Get help with available commands:

```bash
uv run -m scripts.bug_finder --help
```

### Bug Finding

For optimal results, use the intelligent strategy:

```bash
uv run -m scripts.bug_finder -f --mode intelligent
```

### Bug Report Analysis

Analyze and compare results from different strategies:

```bash
uv run -m scripts.analyze_bug_reports
```

This tool provides:
- **Coverage Analysis**: Compare how many issues each strategy finds vs ground truth
- **Cost Comparison**: Token usage and LLM costs between different strategies  
- **Performance Metrics**: Effectiveness analysis of different approaches

## Extras

- `stats`: for codebase statistics
- `builtin-impl`: for builtin LLM code retrieval tools
- `cli`: for command-line interface tools

e.g. specify `codelib[builtin-impl]` in `pyproject.toml` to have builtin LLM code retrieval tools.
