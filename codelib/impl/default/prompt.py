from textwrap import dedent
import logging
import os
from typing import Any

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
