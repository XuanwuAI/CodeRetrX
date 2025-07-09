# Pydantic based structured logging into a JSONL file.
# The "logging" here should specifically be for dataset creation.
from datetime import datetime
from os import PathLike
from typing import Literal, Optional, List, TYPE_CHECKING

from pydantic import BaseModel, RootModel

from coderetrx.static.codebase.models import IsolatedCodeChunkModel
from coderetrx.static.codebase.languages import IDXSupportedTag


class FilteringLog(BaseModel):
    type: Literal["filtering"] = "filtering"
    query: str
    total_chunks: int
    strategy: Literal["static", "dynamic"]
    limit: int
    filter_tags: Optional[List["IDXSupportedTag"]] = None


class CodeChunkClassificationLog(BaseModel):
    type: Literal["code_chunk_classification"] = "code_chunk_classification"
    code_chunk: IsolatedCodeChunkModel
    classification: str
    rationale: str


class VecSearchLog(BaseModel):
    type: Literal["vec_search"] = "vec_search"
    query: str
    total_retrieved: int
    matched_count: int
    strategy: Literal["static", "dynamic"]
    initial_limit: int
    final_limit: int
    filter_tags: Optional[List["IDXSupportedTag"]] = None
    success_ratio: float
    llm_model: str


class LLMCallLog(BaseModel):
    type: Literal["llm_call"] = "llm_call"
    completion_id: str
    model: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    call_url: str


class ErrLog(BaseModel):
    type: Literal["err"] = "err"
    error_type: str
    error: str


type LogData = (
    FilteringLog | CodeChunkClassificationLog | VecSearchLog | LLMCallLog | ErrLog
)


class LogEntry(BaseModel):
    timestamp: str
    data: LogData


def write_log(entry: LogEntry, file: PathLike | str):
    with open(file, "a") as f:
        f.write(entry.model_dump_json() + "\n")


def read_logs(file: PathLike | str):
    with open(file, "r") as f:
        for line in f:
            yield LogEntry.model_validate_json(line)

class JsonLogger:
    def __init__(self, file: PathLike | str):
        self.file = file
        self.file.touch(exist_ok=True)

    def log(self, data: LogData):
        entry = LogEntry(timestamp=datetime.now().isoformat(), data=data)
        write_log(entry, self.file)

if __name__ == "__main__":
    from rich import print

    print(LogEntry.model_json_schema())
