from ._extras import require_extra
require_extra("langchain", "builtin-impl")

import os
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Sequence, Union, overload
import json
from openai import AsyncOpenAI
import asyncio
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


async def call_llm_with_function_call(
    system_prompt: str,
    user_prompt: str,
    function_definition: Dict[str, Any],
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet"
    ],
    attempt: int = 1,
) -> Dict[str, Any]:
    """
    Call LLM with function call support using direct OpenAI-style API.
    
    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        function_definition: OpenAI-style function definition
        model_ids: List of model IDs to try in fallback
        attempt: Current attempt number
        
    Returns:
        Dict containing the function call arguments
        
    Raises:
        ValueError: If function call fails or validation fails
    """
    model_id = model_ids[attempt - 1] if attempt <= len(model_ids) else "unknown"
    
    try:
        logger.info(f"Function call attempt {attempt}/{len(model_ids)}: Using model '{model_id}'")
        
        # Get configuration
        base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        logger.debug(f"Using base_url: {base_url}")
        logger.debug(f"API key length: {len(api_key) if api_key else 0}")
        
        # Initialize client with proper resource management and timeout settings
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
            max_retries=0,
        )
        
        try:
            # Make the function call using the new tools format
            tool = {
                "type": "function",
                "function": function_definition
            }
            
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": function_definition["name"]}},
                temperature=0.1,
            )
            
            message = response.choices[0].message
            if message.tool_calls and len(message.tool_calls) > 0:
                function_args = json.loads(message.tool_calls[0].function.arguments)
                logger.info(f"Successfully received function call response with model '{model_id}'")
                return function_args
            else:
                raise ValueError("No function call in response")
        finally:
            try:
                await client.close()
            except Exception as close_error:
                logger.warning(f"Error closing client: {close_error}")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Function call failed with model '{model_id}': {error_msg}", exc_info=True)
        
        if "nodename nor servname provided" in error_msg or "Connection error" in error_msg:
            logger.warning(f"Network connectivity issue detected, adding delay before retry")
            await asyncio.sleep(1)
        
        if attempt < len(model_ids):
            return await call_llm_with_function_call(
                system_prompt, user_prompt, function_definition, model_ids, attempt + 1
            )
        else:
            raise

if not os.environ.get("DISABLE_LLM_CACHE", "").lower() == "true":
    llm_cache_dir = Path(__file__).parent.parent.parent / ".cache" / "llm"
    llm_cache_dir.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(llm_cache_dir / ".langchain.db")))
else:
    logger.info("LLM cache disabled via DISABLE_LLM_CACHE environment variable")
