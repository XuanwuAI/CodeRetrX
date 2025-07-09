# Ollama Analysis by Bug Type

## Introduction

To assess the effectiveness of the **`coderecx_filter`** strategy, we conducted experiments on the Ollama repository using nine filter prompts, with **`llm_traversal_filter`** serving as the ground truth comparator.

We employed *text-embedding-3-large* for embedding generation. For task resolution, *google/gemini-2.5-flash-lite-preview-06-17* was utilised for simpler tasks, while *openai/gpt-4.1-mini* was used for more complex tasks. The complete configuration details can be found in `.env.example`.

## Experiments

### 1. Dynamic Code Execution

**Filter Prompt**: The code snippet contains a function call that dynamically executes code or system commands. Examples include Python's `eval()`, `exec()`, or functions like `os.system()`, `subprocess.run()` (especially with `shell=True`), `subprocess.call()` (with `shell=True`), or `popen()`. The critical feature is that the string representing the code or command to be executed is not a hardcoded literal; instead, it's derived from a variable, function argument, string concatenation/formatting, or an external source such as user input, network request, or LLM output.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 9       | 334,148   | 39,609     | 100.0%   | $0.1970 | 0.30       |
| llm_traversal_filter | 8       | 770,694   | 222,683    | 100.0%   | $0.6646 | 1          |

------

### 2. Pickle Deserialization

**Filter Prompt**: This resource flags all code snippets that perform deserialization of data using Python's `pickle` or `cloudpickle` libraries (e.g., `pickle.load()`, `pickle.loads()`, `cloudpickle.load()`, `cloudpickle.loads()`). Deserializing data from untrusted sources with these libraries is a significant security risk, potentially leading to arbitrary code execution. This configuration specifically targets instances where the input to these deserialization functions is not a hardcoded literal, indicating that the data might originate from an external or dynamic source.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 0       | 115,733   | 13,294     | 100.0%   | $0.0676 | 0.08       |
| llm_traversal_filter | 0       | 806,605   | 307,879    | 100.0%   | $0.8152 | 1          |

------

### 3. Magic Bytes Validation

**Filter Prompt**: This code snippet implements logic to determine or validate a file's type by reading and analyzing its initial bytes (e.g., magic bytes, file signature, or header). This is often part of a file upload handling mechanism or file processing pipeline where verifying the actual content type based on its leading bytes is critical.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 9       | 441,732   | 113,302    | 100.0%   | $0.3580 | 0.54       |
| llm_traversal_filter | 8       | 751,536   | 223,196    | 100.0%   | $0.6577 | 1          |

------

### 4. Shell Command Execution

**Filter Prompt**: This resource flags code locations that execute shell commands, external programs, or evaluate dynamic code using non-literal inputs. These are potential sinks for command injection or arbitrary code execution if external input influences the command or code being executed. This covers direct command execution (e.g., using functions like os.system, subprocess.run with shell=True, Runtime.exec) and dynamic code evaluation (e.g., using functions like eval, exec, ScriptEngine.eval()), where the executed content is not a hardcoded literal.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 11      | 183,046   | 27,683     | 100.0%   | $0.1510 | 0.22       |
| llm_traversal_filter | 11      | 796,650   | 229,366    | 100.0%   | $0.6856 | 1          |

------


### 5. CLI Command Injection

**Filter Prompt**: This code snippet executes operating system commands using functions like `os.system`, `subprocess.run`, `subprocess.Popen`, `subprocess.call`, `subprocess.check_output`, `commands.getoutput`, or `pty.spawn`. The command being executed is dynamically constructed using string operations (e.g., concatenation, f-strings, `.format()`) with variables that could hold data from external sources like command-line arguments or file content. Prioritize instances where `subprocess` functions are used with `shell=True` or where command components are assembled from non-literal string variables.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 2       | 153,161   | 23,862     | 50.0%    | $0.0994 | 0.15       |
| llm_traversal_filter | 4       | 768,531   | 216,167    | 100.0%   | $0.6533 | 1          |

------

### 6. Other Deserialization

**Filter Prompt**: This code snippet performs deserialization of data using PyTorch's `torch.load()` (or similar model loading functions in AI/ML frameworks), Python's `shelve` module (e.g., `shelve.open()`, `shelf[key]`), or JDBC connection mechanisms (e.g., constructing connection URLs or using drivers). The deserialization is flagged if the input data (such as a model file path or content, data from a shelve file, or components of a JDBC URL) is not a hardcoded literal and could originate from an untrusted external source.

### 6. Other Deserialization

**Filter Prompt**: Flags deserialization using PyTorch `torch.load()`, Python `shelve`, or JDBC connections where input is not hardcoded.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 14      | 476,499   | 85,661     | 93.3%    | $0.3277 | 0.47       |
| llm_traversal_filter | 15      | 769,149   | 244,207    | 100.0%   | $0.6984 | 1          |

------

### 7. Path Traversal/File Ops

**Filter Prompt**: Locate code snippets that perform file system operations (such as reading, writing, deleting, moving files or directories, extracting archives, or including files) or use file paths or names within system commands. Focus on cases where these file paths or names are derived from, or can be influenced by, external sources (e.g., user input, network data, API parameters, environment variables, or function arguments traceable to such sources) and where there is a potential lack of, or insufficient, sanitization or validation against path traversal techniques (e.g., sequences like '..', absolute paths, symbolic links, null bytes, or encoding tricks).

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 118     | 623,360   | 163,334    | 88.7%    | $0.5167 | 0.74       |
| llm_traversal_filter | 133     | 771,621   | 244,673    | 100.0%   | $0.7001 | 1          |

------

### 8. Arbitrary File Write

**Filter Prompt**: This code snippet involves a file system write operation (such as creating, writing to, or moving a file). The destination path, filename, or the content of the file appears to be constructed or influenced by data originating from an external source (e.g., user input, API request parameters, network data, configuration files, environment variables) and there is no clear evidence of robust sanitization, validation, or restriction of the path to a predefined safe directory.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 53      | 371,865   | 108,617    | 89.8%    | $0.3531 | 0.51       |
| llm_traversal_filter | 59      | 760,806   | 244,319    | 100.0%   | $0.6952 | 1          |

------

### 9. File Upload Processing

**Filter Prompt**: This code snippet is involved in processing files uploaded by users. This includes operations such as retrieving the original filename or file extension, determining the destination path or filename for storage, moving or saving the uploaded file to the server's filesystem, and/or implementing validation rules to restrict allowed file types. These validation rules might be based on the file's extension (e.g., checking against a list of permitted or forbidden extensions like '.php', '.jsp', '.asp', '.exe', '.gif', '.jpg'), its MIME type, or its initial bytes (magic bytes/file signatures).

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 13      | 255,868   | 54,491     | 76.5%    | $0.1936 | 0.29       |
| llm_traversal_filter | 17      | 767,913   | 219,938    | 100.0%   | $0.6591 | 1          |

------

## Overall Performance

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Cost Ratio |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ---------- |
| coderecx_filter      | 151     | 2,955,412 | 629,853    | 81.2%    | $2.2641 | 0.36       |
| llm_traversal_filter | 160     | 6,905,171 | 2,078,826  | 100.0%   | $6.2292 | 1          |

------