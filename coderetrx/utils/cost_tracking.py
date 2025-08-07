#  Cost tracking for LLM calls; currently only for OpenRouter.

import httpx
from .logger import JsonLogger, LLMCallLog, ErrLog
from pydantic import BaseModel, Field
from typing import Dict, Optional
from attrs import define
from httpx import AsyncClient
from .logger import read_logs
from os import PathLike
import rich
from rich.table import Table
from rich.console import Console
from collections import defaultdict

a_client = AsyncClient()


@define
class ModelCost:
    prompt: float
    completion: float


type ModelCosts = Dict[str, ModelCost]

_model_costs_cache: Optional[ModelCosts] = None

def get_cost_hook(json_logger: JsonLogger, base_url: str = "https://openrouter.ai/api/v1"):
    async def on_response(response: httpx.Response):
        if not str(response.request.url).startswith(base_url):
            return

        try:
            await response.aread()
            response_json = response.json()
            model = response_json["model"]
            usage_data = response_json["usage"]
            json_logger.log(
                LLMCallLog(
                    model=model,
                    completion_id=response_json["id"],
                    completion_tokens=usage_data["completion_tokens"],
                    prompt_tokens=usage_data["prompt_tokens"],
                    total_tokens=usage_data["total_tokens"],
                    call_url=str(response.request.url),
                )
            )
        except Exception as e:
            json_logger.log(
                ErrLog(
                    error_type="LLM_CALL_ERROR",
                    error=str(e),
                )
            )
    return on_response

async def load_model_costs() -> ModelCosts:
    global _model_costs_cache
    if _model_costs_cache is not None:
        return _model_costs_cache

    all_model_costs_rsp = await a_client.get("https://openrouter.ai/api/v1/models")
    all_model_costs = all_model_costs_rsp.json()
    model_costs_parsed = {}
    for model in all_model_costs["data"]:
        try:
            model_id = model["id"]
            model_slug = model["canonical_slug"]
            model_pricing = model["pricing"]
            model_cost = ModelCost(
                prompt=float(model_pricing["prompt"]),
                completion=float(model_pricing["completion"]),
            )
            model_costs_parsed[model_id] = model_cost
            model_costs_parsed[model_slug] = model_cost
        except Exception as e:
            print(f"Error parsing model {model_slug}: {e}")
    print(f"Loaded {len(model_costs_parsed)} models")
    _model_costs_cache = model_costs_parsed
    return model_costs_parsed

def clear_model_costs_cache():
    """Clear the global model costs cache to force a refresh on next load."""
    global _model_costs_cache
    _model_costs_cache = None

async def calc_llm_costs(log_paths: PathLike | str | list[PathLike | str], model_costs: Optional[ModelCosts] = None):
    from pathlib import Path
    
    # Check if log file exists
    if not Path(log_path).exists():
        return 0.0
        
    model_costs_parsed = model_costs or await load_model_costs()
    
    # Normalize log_paths to always be a list
    if not isinstance(log_paths, list):
        log_paths = [log_paths]
    
    total_cost = 0
    for log_path in log_paths:
        for log_item in read_logs(log_path):
            if log_item.data.type == "llm_call":
                log = log_item.data
                if log.model not in model_costs_parsed:
                    print(f"Model {log.model} not found in model costs")
                    continue
                cost_info = model_costs_parsed[log.model]
                total_cost += (
                    log.prompt_tokens * cost_info.prompt
                    + log.completion_tokens * cost_info.completion
                )
    return total_cost

def calc_input_tokens(log_paths: PathLike | str | list[PathLike | str]):
    # Normalize log_paths to always be a list
    if not isinstance(log_paths, list):
        log_paths = [log_paths]
    
    from pathlib import Path
    
        
    total_input_tokens = 0
    for log_path in log_paths:
        if not Path(log_path).exists():
            continue
        for log_item in read_logs(log_path):
            if log_item.data.type == "llm_call":
                log = log_item.data
                total_input_tokens += log.prompt_tokens
    return total_input_tokens

def calc_output_tokens(log_paths: PathLike | str | list[PathLike | str]):
    # Normalize log_paths to always be a list
    if not isinstance(log_paths, list):
        log_paths = [log_paths]
    
    from pathlib import Path
    
        
    total_output_tokens = 0
    for log_path in log_paths:
        if not Path(log_path).exists():
            continue
        for log_item in read_logs(log_path):
            if log_item.data.type == "llm_call":
                log = log_item.data
                total_output_tokens += log.completion_tokens
    return total_output_tokens

async def cost_breakdown(log_paths: PathLike | str | list[PathLike | str], model_costs: Optional[ModelCosts] = None):
    """Print a table breakdown of LLM costs and tokens by span."""
    model_costs_parsed = model_costs or await load_model_costs()
    console = Console()
    
    # Normalize log_paths to always be a list
    if not isinstance(log_paths, list):
        log_paths = [log_paths]
    
    # Group by span
    span_data = defaultdict(lambda: {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'cost': 0.0,
        'calls': 0
    })
    
    # Group by model
    model_data = defaultdict(lambda: {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'cost': 0.0,
        'calls': 0
    })
    
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0
    
    # Process all log files
    for log_path in log_paths:
        for log_item in read_logs(log_path):
            if log_item.data.type == "llm_call":
                log = log_item.data
                span = log_item.span or "(no span)"
                model = log.model
                
                # Calculate cost for this call
                call_cost = 0.0
                if log.model in model_costs_parsed:
                    cost_info = model_costs_parsed[log.model]
                    call_cost = (
                        log.prompt_tokens * cost_info.prompt
                        + log.completion_tokens * cost_info.completion
                    )
                
                # Update span data
                span_data[span]['prompt_tokens'] += log.prompt_tokens
                span_data[span]['completion_tokens'] += log.completion_tokens
                span_data[span]['total_tokens'] += log.total_tokens
                span_data[span]['cost'] += call_cost
                span_data[span]['calls'] += 1
                
                # Update model data
                model_data[model]['prompt_tokens'] += log.prompt_tokens
                model_data[model]['completion_tokens'] += log.completion_tokens
                model_data[model]['total_tokens'] += log.total_tokens
                model_data[model]['cost'] += call_cost
                model_data[model]['calls'] += 1
                
                # Update totals
                total_cost += call_cost
                total_prompt_tokens += log.prompt_tokens
                total_completion_tokens += log.completion_tokens
                total_calls += 1
    
    # Update table title to reflect if it's aggregated data
    title = "LLM Cost Breakdown by Span"
    if len(log_paths) > 1:
        title += f" (Aggregated from {len(log_paths)} log files)"
    
    # Create the span breakdown table
    table = Table(title=title)
    table.add_column("Span", style="cyan", no_wrap=True)
    table.add_column("Calls", justify="right", style="magenta")
    table.add_column("Prompt Tokens", justify="right", style="green")
    table.add_column("Completion Tokens", justify="right", style="blue")
    table.add_column("Total Tokens", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right", style="red")
    table.add_column("% of Total Cost", justify="right", style="bright_red")
    
    # Sort spans by cost (descending)
    sorted_spans = sorted(span_data.items(), key=lambda x: x[1]['cost'], reverse=True)
    
    for span, data in sorted_spans:
        cost_percentage = (data['cost'] / total_cost * 100) if total_cost > 0 else 0
        table.add_row(
            span,
            str(data['calls']),
            f"{data['prompt_tokens']:,}",
            f"{data['completion_tokens']:,}",
            f"{data['total_tokens']:,}",
            f"${data['cost']:.4f}",
            f"{cost_percentage:.1f}%"
        )
    
    # Add totals row
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_calls}[/bold]",
        f"[bold]{total_prompt_tokens:,}[/bold]",
        f"[bold]{total_completion_tokens:,}[/bold]",
        f"[bold]{total_prompt_tokens + total_completion_tokens:,}[/bold]",
        f"[bold]${total_cost:.4f}[/bold]",
        f"[bold]100.0%[/bold]",
        style="bold"
    )
    
    console.print(table)
    
    # Create the model usage table
    console.print("\n")
    model_title = "LLM Model Usage Breakdown"
    if len(log_paths) > 1:
        model_title += f" (Aggregated from {len(log_paths)} log files)"
    
    model_table = Table(title=model_title)
    model_table.add_column("Model", style="cyan", no_wrap=True)
    model_table.add_column("Calls", justify="right", style="magenta")
    model_table.add_column("Prompt Tokens", justify="right", style="green")
    model_table.add_column("Completion Tokens", justify="right", style="blue")
    model_table.add_column("Total Tokens", justify="right", style="yellow")
    model_table.add_column("Cost ($)", justify="right", style="red")
    model_table.add_column("% of Total Cost", justify="right", style="bright_red")
    
    # Sort models by cost (descending)
    sorted_models = sorted(model_data.items(), key=lambda x: x[1]['cost'], reverse=True)
    
    for model, data in sorted_models:
        cost_percentage = (data['cost'] / total_cost * 100) if total_cost > 0 else 0
        model_table.add_row(
            model,
            str(data['calls']),
            f"{data['prompt_tokens']:,}",
            f"{data['completion_tokens']:,}",
            f"{data['total_tokens']:,}",
            f"${data['cost']:.4f}",
            f"{cost_percentage:.1f}%"
        )
    
    console.print(model_table)
    
    # Print summary
    console.print(f"\n[bold green]Summary:[/bold green]")
    console.print(f"• Total LLM calls: {total_calls}")
    console.print(f"• Total tokens: {total_prompt_tokens + total_completion_tokens:,}")
    console.print(f"• Total cost: ${total_cost:.4f}")
    console.print(f"• Number of spans: {len(span_data)}")
    console.print(f"• Number of unique models: {len(model_data)}")
    
    if model_data:
        most_used_model = max(model_data.items(), key=lambda x: x[1]['calls'])
        most_expensive_model = max(model_data.items(), key=lambda x: x[1]['cost'])
        console.print(f"• Most used model: {most_used_model[0]} ({most_used_model[1]['calls']} calls)")
        console.print(f"• Most expensive model: {most_expensive_model[0]} (${most_expensive_model[1]['cost']:.4f})")
    
    return total_cost
