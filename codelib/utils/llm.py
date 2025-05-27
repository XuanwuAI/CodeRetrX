from ._extras import require_extra

require_extra("langchain", "builtin-impl")

import os
from typing import Union, List, Dict, Optional, Any, cast
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Sequence, Union, overload

from pathlib import Path

from codelib.utils.jsonparser import TolerantJsonParser

def get_langchain_model(model_name) -> BaseChatModel:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
    )


def load_langchain_prompt_template(
    obj: Union[str, Sequence[Dict[str, str]]],
    template_format: str = "f-string",
) -> ChatPromptTemplate:
    if isinstance(obj, str):
        obj = [{"user": obj}]
    messages = []
    for msg in obj:
        if len(msg) != 1:
            raise ValueError("Chat template format error.")
        key, value = next(iter(msg.items()))
        messages.append((key, value))
    return ChatPromptTemplate.from_messages(
        messages,
        template_format=template_format,  # type: ignore
    )


from typing import Any, Dict, List, Type, TypeVar, get_origin, get_args
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import logging

# Assume these imports already exist
# from your_module import get_langchain_model, load_langchain_prompt_template, TolerantJsonParser

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@overload
async def call_llm_with_fallback(
    response_model: Type[T],
    input_data: Dict[str, Any],
    prompt_template: str,
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet",
    ],
    attempt: int = 1,
) -> T: ...


@overload
async def call_llm_with_fallback(
    response_model: Type[List[T]],
    input_data: Dict[str, Any],
    prompt_template: str,
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet",
    ],
    attempt: int = 1,
) -> List[T]: ...


@retry(
    retry=retry_if_exception_type((ValueError, Exception)),
    stop=stop_after_attempt(2),  # Number of models available
    wait=wait_exponential(multiplier=1, min=1, max=4),
)
async def call_llm_with_fallback(
    response_model: Type[T] | Type[List[T]],
    input_data: Dict[str, Any],
    prompt_template: str,
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet",
    ],
    attempt: int = 1,
) -> Union[T, List[T]]:
    """
    Process data with LLM and validate response as a Pydantic model or list of Pydantic models

    Args:
        response_model: Pydantic model class for validating the response
                       If List[Type], validates as list of models
                       If Type, validates as a single model
        input_data: Dictionary containing input data for the prompt
        prompt_template: The prompt template to use
        attempt: Current attempt number

    Returns:
        Either a single validated model instance or a list of validated model instances,
        depending on the response_model type

    Raises:
        ValueError: If LLM output validation fails
    """

    def _validate_list_output(output: Any, item_model: Type[T]) -> List[T]:
        """
        Validate LLM output as a list of Pydantic models

        Args:
            output: The LLM output to validate
            item_model: The Pydantic model class to validate against for each item

        Returns:
            List of validated model instances

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(output, list):
            raise ValueError(f"Expected list output, got {type(output)}")

        if not output:
            raise ValueError("List output cannot be empty")

        validated_items = []
        for i, item in enumerate(output):
            try:
                validated_items.append(item_model.model_validate(item))
            except ValidationError as e:
                raise ValueError(f"Item at index {i} failed validation: {e}")

        return validated_items

    def _validate_single_output(output: Any, model: Type[T]) -> T:
        """
        Validate LLM output as a single Pydantic model

        Args:
            output: The LLM output to validate
            model: The Pydantic model class to validate against

        Returns:
            Validated model instance

        Raises:
            ValueError: If validation fails
        """
        if isinstance(output, list):
            raise ValueError(f"Expected dict output, got list")

        try:
            return model.model_validate(output)
        except ValidationError as e:
            raise ValueError(f"Output failed validation: {e}")

    model_id = model_ids[attempt - 1]
    logger.info(
        f"LLM call attempt {attempt}/{len(model_ids)}: Using model '{model_id}'"
    )

    model = get_langchain_model(model_id)
    langchain_prompt = load_langchain_prompt_template(prompt_template)
    chain = langchain_prompt | model.with_retry() | TolerantJsonParser()

    try:
        llm_result = await chain.ainvoke(input_data)

        # Check if response_model is a List type
        is_list_type = get_origin(response_model) is list

        if is_list_type:
            # Get the type inside the List
            item_model = get_args(response_model)[0]
            validated_result = _validate_list_output(llm_result, item_model)
            logger.info(
                f"Successfully processed LLM request with model '{model_id}', returned {len(validated_result)} validated items"
            )
        else:
            # Validate as a single item
            assert issubclass(response_model, BaseModel)
            validated_result = _validate_single_output(llm_result, response_model)
            logger.info(
                f"Successfully processed LLM request with model '{model_id}', returned a validated item"
            )

        return validated_result

    except Exception as e:
        logger.error(
            f"LLM processing failed with model '{model_id}': {str(e)}", exc_info=True
        )

        # Always use fallback strategy (try next model)
        if attempt < len(model_ids):
            return await call_llm_with_fallback(
                response_model, input_data, prompt_template, model_ids, attempt + 1
            )
        else:
            raise


llm_cache_dir = Path(__file__).parent.parent.parent / ".cache" / "llm"
llm_cache_dir.mkdir(parents=True, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=str(llm_cache_dir / ".langchain.db")))
