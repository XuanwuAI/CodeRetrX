# NeonDatabase Analysis by Bug Type

## Introduction

To assess the effectiveness of the **coderetrx_filter** strategy, we conducted experiments on the NeonDatabase repository using nine filter prompts, with **llm_traversal_filter** serving as the ground truth comparator.  

We employed *text-embedding-3-large* for embedding generation. For task resolution, *google/gemini-2.5-flash-lite-preview-06-17* was utilised for simpler tasks, while *openai/gpt-4.1-mini* was used for more complex tasks. The complete configuration details can be found in `.env.example`.

## Experiments

### 1. Dynamic Code Execution

**Filter Prompt**: The code snippet contains a function call that dynamically executes code or system commands. Examples include Python's `eval()`, `exec()`, or functions like `os.system()`, `subprocess.run()` (especially with `shell=True`), `subprocess.call()` (with `shell=True`), or `popen()`. The critical feature is that the string representing the code or command to be executed is not a hardcoded literal; instead, it's derived from a variable, function argument, string concatenation/formatting, or an external source such as user input, network request, or LLM output.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 8       | 117,589   | 16,316     | 80.0%    | $0.0731 | 0.33          |
| llm_traversal_filter | 5       | 287,478   | 68,452     | 100.0%   | $0.2245 | 1             |

------

### 2. Pickle Deserialization

**Filter Prompt**: This code snippet deserializes data using Python's `pickle` or `cloudpickle` libraries (e.g., `pickle.load()`, `pickle.loads()`, `cloudpickle.load()`, `cloudpickle.loads()`). Deserializing data from untrusted sources with these libraries is a significant security risk, potentially leading to arbitrary code execution. This configuration specifically targets instances where the input to these deserialization functions is not a hardcoded literal, indicating that the data might originate from an external or dynamic source.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 0       | 82,818    | 9,110      | 100.0%   | $0.0477 | 0.22          |
| llm_traversal_filter | 0       | 274,064   | 69,596     | 100.0%   | $0.2210 | 1             |

------

### 3. Magic Bytes Validation

**Filter Prompt**: This code snippet implements logic to determine or validate a file's type by reading and analyzing its initial bytes (e.g., magic bytes, file signature, or header). This is often part of a file upload handling mechanism or file processing pipeline where verifying the actual content type based on its leading bytes is critical.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- |-----------|------------| -------- |---------| ------------- |
| coderetrx_filter     | 2       | 89,815    | 9,532      | 33.3%    | $0.0512 | 0.61          |
| llm_traversal_filter | 3       | 281,154   | 73,600     | 100.0%   | $0.2302 | 1             |

------

### 4. Shell Command Execution

**Filter Prompt**: This code snippet executes a shell command, system command, an external program, or evaluates a string as code. This is often done using functions like `os.system`, `subprocess.call`, `subprocess.run` (especially with `shell=True`), `subprocess.Popen` (especially with `shell=True`), `commands.getoutput`, `Runtime.getRuntime().exec`, `ProcessBuilder`, `php.system`, `php.exec`, `php.shell_exec`, `php.passthru`, `php.popen`, PHP backticks (`` ` ``), `Node.child_process.exec`, `Node.child_process.execSync`, `eval`, `exec`, `ScriptEngine.eval()`, `execCommand`, `Perl.system`, `Ruby.system`, `Ruby.exec`, Ruby backticks (`` ` ``), `Go.os/exec.Command`, etc. The command string, arguments to the command, or the string being evaluated as code, are derived from variables, function parameters, or other dynamic sources, rather than being solely hardcoded string literals.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 11      | 97,291    | 9,923      | 100.0%   | $0.0548 | 0.24          |
| llm_traversal_filter | 11      | 296,046   | 70,910     | 100.0%   | $0.2319 | 1             |

------

### 5. CLI Command Injection

**Filter Prompt**: This code snippet executes operating system commands using functions like `os.system`, `subprocess.run`, `subprocess.Popen`, `subprocess.call`, `subprocess.check_output`, `commands.getoutput`, or `pty.spawn`. The command being executed is dynamically constructed using string operations (e.g., concatenation, f-strings, `.format()`) with variables that could hold data from external sources like command-line arguments or file content. Prioritize instances where `subprocess` functions are used with `shell=True` or where command components are assembled from non-literal string variables.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 4       | 79,153    | 7,621      | 100.0%   | $0.0439 | 0.20          |
| llm_traversal_filter | 2       | 286,764   | 65,519     | 100.0%   | $0.2195 | 1             |

------

### 6. Other Deserialization

**Filter Prompt**: This code snippet performs deserialization of data using PyTorch's `torch.load()` (or similar model loading functions in AI/ML frameworks), Python's `shelve` module (e.g., `shelve.open()`, `shelf[key]`), or JDBC connection mechanisms (e.g., constructing connection URLs or using drivers). The deserialization is flagged if the input data (such as a model file path or content, data from a shelve file, or components of a JDBC URL) is not a hardcoded literal and could originate from an untrusted external source.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 5       | 87,717    | 8,169      | 100.0%   | $0.0482 | 0.20          |
| llm_traversal_filter | 4       | 286,968   | 77,481     | 100.0%   | $0.2388 | 1             |

------

### 7. Path Traversal/File Ops

**Filter Prompt**: Locate code snippets that perform file system operations (such as reading, writing, deleting, moving files or directories, extracting archives, or including files) or use file paths or names within system commands. Focus on cases where these file paths or names are derived from, or can be influenced by, external sources (e.g., user input, network data, API parameters, environment variables, or function arguments traceable to such sources) and where there is a potential lack of, or insufficient, sanitization or validation against path traversal techniques (e.g., sequences like `..`, absolute paths, symbolic links, null bytes, or encoding tricks).

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 76      | 132,290   | 23,332     | 89.7%    | $0.0902 | 0.38          |
| llm_traversal_filter | 78      | 287,784   | 74,781     | 100.0%   | $0.2348 | 1             |

------

### 8. Arbitrary File Write

**Filter Prompt**: This code snippet involves a file system write operation (such as creating, writing to, or moving a file). The destination path, filename, or the content of the file appears to be constructed or influenced by data originating from an external source (e.g., user input, API request parameters, network data, configuration files, environment variables) and there is no clear evidence of robust sanitization, validation, or restriction of the path to a predefined safe directory.

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 38      | 116,399   | 20,189     | 89.5%    | $0.0789 | 0.34          |
| llm_traversal_filter | 38      | 284,214   | 74,949     | 100.0%   | $0.2336 | 1             |

------

### 9. File Upload Processing

**Filter Prompt**: This code snippet is involved in processing files uploaded by users. This includes operations such as retrieving the original filename or file extension, determining the destination path or filename for storage, moving or saving the uploaded file to the server's filesystem, and/or implementing validation rules to restrict allowed file types. These validation rules might be based on the file's extension (e.g., checking against a list of permitted or forbidden extensions like `.php`, `.jsp`, `.asp`, `.exe`, `.gif`, `.jpg`), its MIME type, or its initial bytes (magic bytes/file signatures).

| strategy             | Results | In Tokens | Out Tokens | Coverage | Cost    | Relative Cost |
| -------------------- | ------- | --------- | ---------- | -------- | ------- | ------------- |
| coderetrx_filter     | 0       | 96,770    | 16,540     | 100.0%   | $0.0652 | 0.29          |
| llm_traversal_filter | 0       | 286,560   | 68,993     | 100.0%   | $0.2250 | 1             |

------
