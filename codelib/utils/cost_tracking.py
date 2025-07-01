#  Cost tracking for LLM calls; currently only for OpenRouter.

import httpx
from .logger import JsonLogger, LLMCallLog, ErrLog
from pydantic import BaseModel, Field
from typing import Dict, Optional
from attrs import define
from httpx import AsyncClient
from .logger import read_logs
from os import PathLike

a_client = AsyncClient()


@define
class ModelCost:
    prompt: float
    completion: float


type ModelCosts = Dict[str, ModelCost]

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
    return model_costs_parsed

async def calc_llm_costs(log_path: PathLike | str, model_costs: Optional[ModelCosts] = None):
    model_costs_parsed = model_costs or await load_model_costs()
    total_cost = 0
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

def calc_input_tokens(log_path: PathLike | str):
    total_input_tokens = 0
    for log_item in read_logs(log_path):
        if log_item.data.type == "llm_call":
            log = log_item.data
            total_input_tokens += log.prompt_tokens
    return total_input_tokens

def calc_output_tokens(log_path: PathLike | str):
    total_output_tokens = 0
    for log_item in read_logs(log_path):
        if log_item.data.type == "llm_call":
            log = log_item.data
            total_output_tokens += log.completion_tokens
    return total_output_tokens
