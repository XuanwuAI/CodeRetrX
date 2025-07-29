# Pydantic based structured logging into a JSONL file.
# The "logging" here should specifically be for dataset creation.
import asyncio
from contextvars import ContextVar, Token
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Literal, Optional, List, TYPE_CHECKING, Callable, Any
from contextlib import asynccontextmanager
from functools import wraps

from pydantic import BaseModel, RootModel

from coderetrx.static.codebase.models import IsolatedCodeChunkModel
from coderetrx.static.codebase.languages import IDXSupportedTag


# Context variable to track the current span name
current_span: ContextVar[Optional[str]] = ContextVar('current_span', default=None)


def set_span(span_name: Optional[str], replace: bool = False) -> Token[Optional[str]]:
    """Set the current span name in the context.
    
    If there's an existing span, appends to it with dot notation unless replace=True.
    If span_name is None, clears the current span.
    
    Args:
        span_name: The span name to set. If None, clears the current span.
        replace: If True, replaces the foremost span instead of appending.
        
    Returns:
        A Token that can be used with current_span.reset() to restore the previous state.
    """
    def wrap_reset(token):
        def inner():
            current_span.reset(token)
        return inner

    if span_name is None:
        return wrap_reset(current_span.set(None))
    
    existing = current_span.get()
    if existing and not replace:
        return wrap_reset(current_span.set(f"{existing}.{span_name}"))
    elif existing and replace:
        # Replace the foremost span (first part before the first dot)
        parts = existing.split('.', 1)
        if len(parts) > 1:
            # Keep everything after the first dot and prepend the new span
            return wrap_reset(current_span.set(f"{span_name}.{parts[1]}"))
        else:
            # Only one span, just replace it entirely
            return wrap_reset(current_span.set(span_name))
    else:
        return wrap_reset(current_span.set(span_name))


def get_current_span() -> Optional[str]:
    """Get the current span name from context."""
    return current_span.get()


@asynccontextmanager
async def use_span(span_name: Optional[str], replace: bool = False):
    """Async context manager for managing spans.
    
    Args:
        span_name: The span name to set. If None, clears the current span.
        replace: If True, replaces the foremost span instead of appending.
        
    Example:
        async with use_span("database_query"):
            # Code here will have span set to "database_query" 
            # (or "parent.database_query" if there was a parent span)
            pass
        # Span is automatically restored to previous value
    """
    reset_func = set_span(span_name, replace)
    try:
        yield
    finally:
        reset_func()


def wrap_span(span_name: Optional[str] = None, replace: bool = False):
    """Decorator for automatically setting spans on function calls.
    
    Args:
        span_name: The span name to set. If None, uses the function name.
        replace: If True, replaces the foremost span instead of appending.
        
    Example:
        @wrap_span("data_processing")
        async def process_data():
            pass
            
        @wrap_span()  # Uses function name as span
        async def calculate_metrics():
            pass
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        actual_span_name = span_name if span_name is not None else func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with use_span(actual_span_name, replace):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                reset_func = set_span(actual_span_name, replace)
                try:
                    return func(*args, **kwargs)
                finally:
                    reset_func()
            return sync_wrapper
    
    return decorator


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
    cached: bool = False


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
    span: Optional[str] = None


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
        Path(file).touch(exist_ok=True)

    def log(self, data: LogData):
        entry = LogEntry(
            timestamp=datetime.now().isoformat(), 
            data=data, 
            span=get_current_span()
        )
        write_log(entry, self.file)

if __name__ == "__main__":
    from rich import print

    print(LogEntry.model_json_schema())
