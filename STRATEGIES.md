# Coarse Recall Strategies

In the coarse recall stage, multiple strategies are used to efficiently retrieve potentially relevant code snippets at low cost. The retrieved results will then be further processed in the refined filter stage to improve precision and accuracy.

## Available Strategies

- **`file_name`**: The fastest filtering strategy, ideal for coarse recall by directly determining relevance based on filenames. This approach is highly effective for cases where the query can be matched through filenames alone, such as "retrieve all configuration files." Its simplicity makes it extremely efficient for structural queries and file discovery.

- **`symbol_name`**: A filtering approach that focuses on symbol names (e.g., function or class names) during coarse recall. This method excels in cases like "retrieve functions implementing cryptographic algorithms," where relevance can be inferred directly from the symbol name. It balances recall and precision for targeted queries at the symbol level.

- **`line_per_symbol`**: A high-accuracy filtering strategy that leverages line-level vector search combined with LLM analysis. It identifies relevance by focusing on the most relevant code lines (`top-k`) within a function body. This method is particularly effective for complex cases where understanding the function's implementation is necessary, such as "retrieve code implementing authentication logic." While powerful, it's also versatile enough to handle most queries effectively. Our benchmarking shows that the algorithm achieves over 90% recall with approximately 25% of the computational cost, making it both precise and efficient.

- **`auto`**: Automatically selects the optimal filtering strategy based on query complexity, routing requests to the most appropriate method. For the majority of cases, `line_per_symbol` is a well-balanced and reliable choice, but `file_name` or `symbol_name` can also be explicitly used for scenarios where they excel.

## Performance Characteristics

- **Speed**: `file_name` > `symbol_name` > `auto` > `line_per_symbol` > `precise`
- **Accuracy**: `precise` > `line_per_symbol` > `auto` > `symbol_name` > `file_name`
- **Cost**: `file_name` < `line_per_symbol` < `auto` < `symbol_name` < `precise`

Use `file_name` for structural queries, `symbol_name` for API search, `line_per_symbol` for specific code related analysis, `auto` for general purpose, and `precise` for ground truth.

