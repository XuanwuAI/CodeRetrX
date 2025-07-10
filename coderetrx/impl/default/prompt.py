from textwrap import dedent
import logging
import os
from typing import Any, List, Dict

from pydantic import BaseModel

logger = logging.getLogger(__name__)
if os.getenv("DEBUG"):
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled for LLM tool prompt processing")


llm_filter_prompt_template = dedent(
    """\
<input>
{code_elements}
</input>
<task>
Here are {code_element_number} code element(s) (i.g. file/function/dependency/class/keyword etc.). you need to analysis. 
You need to analysis whether each code element meets this condition: 
<requirement>
{requirement}
</requirement>
You should think step by step. 
</task>
<output_format>
return a JSONArray which elements are JsonObjects that contain index, reason and result.                                            
[{{
  "index":  0, // the index of the code element
  "reason": "Thought for code element", 
  "result": true/false, // a boolean indicating whether the text meets the requirement
}}, 
{{
  "index":  1, // the index of the code element
  "reason": "Thought for code element", 
  "result": true/false, // a boolean indicating whether the text meets the requirement
}},
...
]
</output_format>
            """
).strip()


llm_mapping_prompt_template = dedent(
    """\
<input>
{code_elements}
</input>
<task>
Here are {code_element_number} code element(s) (i.g. file/function/dependency/class/keyword etc.) you need to analysis. 
You need to analysis whether each code element and extract information according to the following requirements
<requirement>
{requirement}
</requirement>
You should think step by step. 
</task>
<output_format>
return a JSONArray which elements are JsonObjects that contain index, reason and result.                                             
[{{
  "index":  0, // the index of the code element
  "reason": "Thought for code element", 
  "result": "The extracted result, must be a pretty print string, If no result can be extracted, just return an empty string"
}},
{{
  "index":  1, // the index of the code element
  "reason": "Thought for code element", 
  "result": "The extracted result, must be a pretty print string, If no result can be extracted, just return an empty string"
}},
...
]
</output_format>
            """
).strip()


topic_extraction_prompt_template = dedent(
    """\
<input>
{input}
</input>
<task>

<requirement>
The objective of this task is to extract the **core topic or concept** from the given input text. 
The extracted topic will be used as a **query vector** for vector recall in a **Retrieval-Augmented Generation (RAG)** system. 
This process helps retrieve relevant documents or information from a vector database based on the topic's semantic meaning.

Key objectives:
1. Identify the **main functionality, feature, or action** described in the input.
2. Summarize the topic into a **concise, meaningful phrase** that can serve as an effective input for vector search.
3. Ensure the extracted topic is specific enough to capture the essence of the input but general enough for semantic matching in vector recall.
4. If there is a `<content_criterias>` tag in the input, the extracted topic should focus **only on the content criteria** described within the tag and ignore the rest of the input.

You should think step by step. 
</requirement>

</task>
<output_format>
return a JSONObject that contains:
- "reason": an explanation of how the result was derived.
- "result": the extracted topic.
{{
    "reason": "The input describes the use of decompression functions to handle compressed files.",
    "result": "decompression functions"
}}
</output_format>

<examples>
<example>
    <input>This code snippet uses decompression functions (such as tarfile.extract, tarfile.extractall) to handle compressed files.</input>
    <output>
    {{
        "reason": "The input describes the use of decompression functions to handle compressed files.",
        "result": "decompression functions"
    }}
    </output>
</example>
<example>
    <input>This code snippet uses dynamic loading to execute a piece of code, for example, using the eval() function in Python to execute Python code.</input>
    <output>
    {{
        "reason": "The input focuses on dynamic loading and execution of code using methods like eval().",
        "result": "dynamic code execution"
    }}
    </output>
</example>
<example>
    <input>
        A file with this path is highly likely to contain content that matches the following criteria:
        <content_criterias>
            This resource flags all code snippets that execute bash commands (For example, in Python, functions like `system()`, `os.popen()`, and `subprocess.Popen()` are used.), 
            and where the bash command is not hardcoded.
        </content_criterias>
        <note>
            The objective of this requirement is to preliminarily identify files based on their paths that are likely to meet specific content criteria.
            Files with matching paths will proceed to a deeper analysis in the content filter (content_criterias) at a later stage (not in this run).  
        </note>
    </input>
    <output>
    {{
        "reason": "The input describes the execution of bash commands, identifying it as a code execution task.",
        "result": "code execution"
    }}
    </output>
</example>
</examples>
    """
).strip()


class CodeMapFilterResult(BaseModel):
    index: int
    reason: str
    result: Any


class KeywordExtractorResult(BaseModel):
    reason: str
    result: str


# Function call system prompts
llm_filter_function_call_system_prompt = dedent(
    """\
You are a code analysis assistant. You need to analyze each code element and determine whether it meets the specified requirement.

CRITICAL INSTRUCTIONS:
1. You MUST call the analyze_code_elements function exactly once
2. For each code element, provide exactly three fields: index, reason, and result
3. The index must be the zero-based integer index of the code element
4. The reason must be a string explaining your analysis
5. The result must be a boolean (true/false) indicating whether the element meets the requirement
6. Do NOT add any extra fields or modify the field names

Think step by step and provide reasoning for each decision.
    """
).strip()

llm_mapping_function_call_system_prompt = dedent(
    """\
You are a code analysis assistant. You need to analyze each code element and extract information according to the specified requirements.

CRITICAL INSTRUCTIONS:
1. You MUST call the analyze_code_elements function exactly once
2. For each code element, provide exactly three fields: index, reason, and result
3. The index must be the zero-based integer index of the code element
4. The reason must be a string explaining your extraction process
5. The result must be a string containing the extracted information, or empty string if nothing extracted
6. Do NOT add any extra fields or modify the field names

Think step by step and provide reasoning for each extraction.
    """
).strip()

topic_extraction_function_call_system_prompt = dedent(
    """\
You are a topic extraction assistant. You need to analyze the input text and extract the core topic or concept that can be used as a query vector for RAG system.

CRITICAL INSTRUCTIONS:
1. You MUST call the extract_topic function exactly once
2. Provide exactly two fields: reason and result
3. The reason must be a string explaining your extraction process
4. The result must be a string containing the extracted topic
5. Do NOT add any extra fields or modify the field names

Think step by step and provide clear reasoning for the extraction.
    """
).strip()

# Function call user prompt template
filter_and_mapping_function_call_user_prompt_template = dedent(
    """\
<input>
{code_elements}
</input>
<task>
Here are {code_element_number} code element(s) (i.e., file/function/dependency/class/keyword etc.) you need to analyze. 
<requirement>
{requirement}
</requirement>

IMPORTANT: 
- Analyze ALL {code_element_number} code elements provided above
- For each element with index 0 to {code_element_number_minus_one}, provide an analysis
- Use the analyze_code_elements function with exactly these fields: index, reason, result
- Do NOT use any other field names like "commands", "command", "description", etc.
- The field names must be exactly: "index", "reason", "result"

You should think step by step.
</task>
    """
).strip()

topic_extraction_function_call_user_prompt_template = dedent(
    """\
<input>
{input}
</input>
<task>
You need to extract the core topic or concept from the input text above, which will be used as a query vector for a Retrieval-Augmented Generation (RAG) system.
<requirement>
- Identify the main functionality, feature, or action described in the input.
- Summarize the topic into a concise, meaningful phrase suitable for vector search.
- If a <content_criterias> tag is present, focus only on the content criteria within that tag.
</requirement>

IMPORTANT:
- You MUST call the extract_topic function exactly once.
- Provide exactly these fields: "reason" (your explanation) and "result" (the extracted topic).
- Do NOT use any other field names or add extra fields.

You should think step by step.
</task>
    """
).strip()

# Function definitions for function calls
def get_filter_function_definition() -> Dict[str, Any]:
    """Get the function definition for filter operations."""
    return {
        "name": "analyze_code_elements",
        "description": """Analyze code elements to determine if they meet specified requirements. 

IMPORTANT: Each analysis object must contain exactly these three fields:
- index: integer (0, 1, 2, etc.)
- reason: string (your explanation)
- result: boolean (true or false only)

Example: {"index": 0, "reason": "This code uses subprocess.run which executes shell commands", "result": true}""",
        "parameters": {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "description": """Array of analysis results for each code element. 
                    
Example format:
[
  {"index": 0, "reason": "Contains os.system call", "result": true},
  {"index": 1, "reason": "No command execution found", "result": false}
]""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "The zero-based index of the code element being analyzed (0, 1, 2, etc.)"
                            },
                            "reason": {
                                "type": "string", 
                                "description": "Detailed reasoning explaining why the code element does or does not meet the requirement"
                            },
                            "result": {
                                "type": "boolean",
                                "description": "Boolean value: true if the code element meets the requirement, false if it does not. Must be exactly true or false, not a string."
                            }
                        },
                        "required": ["index", "reason", "result"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["analyses"]
        }
    }


def get_mapping_function_definition() -> Dict[str, Any]:
    """Get the function definition for mapping operations."""
    return {
        "name": "analyze_code_elements",
        "description": """Analyze code elements and extract information according to specified requirements.

IMPORTANT: Each analysis object must contain exactly these three fields:
- index: integer (0, 1, 2, etc.)
- reason: string (your explanation)
- result: string (extracted info or empty string)

Example: {"index": 0, "reason": "Extracted function name", "result": "execute_command"}""",
        "parameters": {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "description": """Array of analysis results for each code element.
                    
Example format:
[
  {"index": 0, "reason": "Found command execution", "result": "subprocess.run"},
  {"index": 1, "reason": "No relevant information", "result": ""}
]""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "The zero-based index of the code element being analyzed (0, 1, 2, etc.)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Detailed reasoning explaining why information was or was not extracted from the code element"
                            },
                            "result": {
                                "type": "string",
                                "description": "The extracted information as a string, or empty string if nothing can be extracted. Must be a string, not boolean or other type."
                            }
                        },
                        "required": ["index", "reason", "result"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["analyses"]
        }
    }

def get_topic_extraction_function_definition() -> Dict[str, Any]:
    """Get the function definition for topic extraction operations."""
    return {
        "name": "extract_topic",
        "description": """Extract the core topic or concept from input text for use as a query vector in RAG systems.

IMPORTANT: The analysis object must contain exactly these two fields:
- reason: string (your explanation)
- result: string (extracted topic)

Example: {"reason": "The input describes file decompression operations", "result": "decompression functions"}""",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Detailed reasoning explaining how the topic was extracted from the input text"
                },
                "result": {
                    "type": "string",
                    "description": "The extracted topic as a concise, meaningful phrase suitable for vector search"
                }
            },
            "required": ["reason", "result"]
        }
    }
