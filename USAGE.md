# Usage Guide

## Programmatic API

CodeRetrX provides a powerful programmatic interface through the `coderetrx_filter` and `llm_traversal_filter` API, which enables flexible code analysis and retrieval across different search strategies and filtering modes.

The coderetrx_filter implements stragties designed by us. It offers a cost-effective semantic recall approach for large-scale repositories, achieving approximately 90% recall with only about 25% of the resource consumption in practical tests(line_per_symbol strategy) — the larger the repository, the greater the savings.

The llm_traversal_filter provides the most comprehensive and accurate analysis, ideal for establishing ground truth. For small-scale repositories, this strategy can also be a good choise.

### Using coderetrx_filter with symbol_name strategy 

```python
from pathlib import Path
from coderetrx.retrieval import coderetrx_filter, CodebaseFactory

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

### Using coderetrx_filter with line_per_symbol strategy

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

### Using llm_traversal_filter (Ground Truth & Maximum Accuracy)

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

## Coarse Recall Strategy 

**coderetrx_filter** supports `file_name`, `symbol_name`, `line_per_symbol`, and `auto` strategies for different speed/accuracy tradeoffs. **llm_traversal_filter** uses full LLM processing for maximum accuracy.

[See detailed strategy comparison in README.md](#-coarse-recall-strategies)

## Target type Options

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

## Settings Configuration

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

## Working with Results

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

