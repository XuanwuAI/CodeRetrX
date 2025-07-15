from coderetrx.utils.path import get_cache_dir

import asyncio
import json
import logging
import os
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from httpx import AsyncClient
from .llm_cache import get_llm_cache_provider
from nanoid import generate
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationError, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from coderetrx.utils.cost_tracking import get_cost_hook
from coderetrx.utils.logger import JsonLogger
import json_repair


def generate_session_id() -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{date_str}-{generate(size=16)}"


class LLMSettings(BaseSettings):
    """Settings for LLM configuration using environment variables."""
    
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )
    
    # OpenAI Configuration
    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenAI API",
        alias="OPENAI_BASE_URL"
    )
    openai_api_key: str = Field(
        default="",
        description="API key for OpenAI",
        alias="OPENAI_API_KEY"
    )
    
    # Cache Configuration
    disable_llm_cache: bool = Field(
        default=False,
        description="Disable LLM response caching",
        alias="DISABLE_LLM_CACHE"
    )
    
    # Session Configuration
    session_id: str = Field(
        default_factory=generate_session_id,
        description="Session ID for tracking requests"
    )
    
    # Logging Configuration
    enable_json_log: bool = Field(
        default=True,
        description="Enable JSON logging for LLM requests",
        alias="ENABLE_JSON_LOG"
    )
    json_log_base_path: Path = Field(
        default_factory=lambda: Path("logs/"),
        description="Base path for JSON log files",
        alias="JSON_LOG_BASE_PATH"
    )
    
    # Proxy Configuration
    proxy: Optional[str] = Field(
        default=None,
        description="Proxy URL for HTTP requests, support socks5 and http(s) proxies",
        alias="LLM_PROXY"
    )

    def get_json_log_path(self) -> Path:
        path = self.json_log_base_path
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{self.session_id}.jsonl"

    def get_json_logger(self) -> JsonLogger:
        return JsonLogger(self.get_json_log_path()) 
    
    def get_httpx_client(self) -> AsyncClient:
        hook = get_cost_hook(self.get_json_logger(), self.openai_base_url)
        if self.proxy:
            return AsyncClient(proxy=self.proxy, event_hooks={"response": [hook]})
        return AsyncClient(
            event_hooks={"response": [hook]},
        )


# Create a global settings instance
llm_settings = LLMSettings()
def set_llm_settings(settings: LLMSettings) -> None:
    """
    Set global LLM settings.
    
    Args:
        settings: LLMSettings instance to set as global settings
    """
    global llm_settings
    llm_settings = settings

def get_llm_settings() -> LLMSettings:
    """
    Get the global LLM settings.
    
    Returns:
        The current LLMSettings instance
    """
    return llm_settings

def _parse_prompt_template(
    prompt_template: Union[str, Sequence[Dict[str, str]]],
    input_data: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Parse prompt template and format with input data to create OpenAI messages format.
    
    Args:
        prompt_template: Either a string or sequence of role-content dictionaries
        input_data: Data to format the template with
        
    Returns:
        List of message dictionaries in OpenAI format
    """
    if isinstance(prompt_template, str):
        # Simple string template - treat as user message
        formatted_content = prompt_template.format(**input_data)
        return [{"role": "user", "content": formatted_content}]
    
    # Sequence of role-content dictionaries
    messages = []
    for msg in prompt_template:
        if len(msg) != 1:
            raise ValueError("Each message must have exactly one role-content pair")
        
        role, content_template = next(iter(msg.items()))
        formatted_content = content_template.format(**input_data)
        messages.append({"role": role, "content": formatted_content})
    
    return messages


def _parse_json_with_repair(text: str) -> Any:
    """
    Parse JSON with multiple fallback strategies, similar to TolerantJsonParser.
    
    Args:
        text: Raw text potentially containing JSON
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If all parsing strategies fail
    """
    # Remove potential markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try different parsing strategies
    parsers = [
        json.loads,  # Standard JSON parsing
        json_repair.loads,  # JSON repair
    ]
    
    # Try parsing the full text first
    for parser in parsers:
        try:
            return parser(text)
        except (ValueError, json.JSONDecodeError):
            continue
    
    # Try finding JSON substrings starting with { or [
    for i, char in enumerate(text):
        if char in ['{', '[']:
            substring = text[i:]
            for parser in parsers:
                try:
                    return parser(substring)
                except (ValueError, json.JSONDecodeError):
                    continue
    
    raise ValueError(f"Failed to parse JSON from text: {text[:200]}...")


logger = logging.getLogger(__name__)

# Configure httpx logging to warning level to suppress INFO messages
logging.getLogger("httpx").setLevel(logging.WARNING)

T = TypeVar("T", bound=BaseModel)


@overload
async def call_llm_with_fallback(
    response_model: Type[T],
    input_data: Dict[str, Any],
    prompt_template: Union[str, Sequence[Dict[str, str]]],
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet",
    ],
    attempt: int = 1,
    settings: Optional[LLMSettings] = None,
) -> T: ...


@overload
async def call_llm_with_fallback(
    response_model: Type[List[T]],
    input_data: Dict[str, Any],
    prompt_template: Union[str, Sequence[Dict[str, str]]],
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet",
    ],
    attempt: int = 1,
    settings: Optional[LLMSettings] = None,
) -> List[T]: ...


@retry(
    retry=retry_if_exception_type((ValueError, Exception)),
    stop=stop_after_attempt(2),  # Number of models available
    wait=wait_exponential(multiplier=1, min=1, max=4),
)
async def call_llm_with_fallback(
    response_model: Type[T] | Type[List[T]],
    input_data: Dict[str, Any],
    prompt_template: Union[str, Sequence[Dict[str, str]]],
    model_ids: List[str] = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet",
    ],
    attempt: int = 1,
    settings: Optional[LLMSettings] = None,
) -> Union[T, List[T]]:
    """
    Process data with LLM and validate response as a Pydantic model or list of Pydantic models.
    Uses async OpenAI API instead of LangChain.

    Args:
        response_model: Pydantic model class for validating the response
                       If List[Type], validates as list of models
                       If Type, validates as a single model
        input_data: Dictionary containing input data for the prompt
        prompt_template: The prompt template to use (string or sequence of role-content dicts)
        model_ids: List of model IDs to try in fallback
        attempt: Current attempt number
        settings: Optional LLM settings to use, defaults to global settings

    Returns:
        Either a single validated model instance or a list of validated model instances,
        depending on the response_model type

    Raises:
        ValueError: If LLM output validation fails
    """
    
    # Use provided settings or fall back to global settings
    if settings is None:
        settings = llm_settings

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

    model_id = model_ids[attempt - 1] if attempt <= len(model_ids) else "unknown"
    logger.debug(
        f"LLM call attempt {attempt}/{len(model_ids)}: Using model '{model_id}'"
    )

    try:
        # Get configuration
        base_url = settings.openai_base_url
        api_key = settings.openai_api_key
        
        logger.debug(f"Using base_url: {base_url}")
        logger.debug(f"API key length: {len(api_key) if api_key else 0}")
        
        # Initialize client with proper resource management and timeout settings
        httpx_client = settings.get_httpx_client() if settings.enable_json_log else None
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300.0,
            max_retries=0,
            http_client=httpx_client
        )
        cache_provider = get_llm_cache_provider()
        
        try:
            # Parse prompt template and format with input data
            messages = _parse_prompt_template(prompt_template, input_data)
            
            # Create request
            request = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.1,
            }
            
            # Check cache first
            cache_key = cache_provider.hash_params(request)
            cached_response = cache_provider.get(cache_key)
            if cached_response:
                # Cache hit, return the cached response
                logger.info(f"Cache hit for key: {cache_key}")
                response = ChatCompletion.model_validate_json(cached_response)
            else: 
                response: ChatCompletion = await client.chat.completions.create(**request)
                cache_provider.insert(
                    cache_key, 
                    request,
                    response.model_dump(exclude_none=True, exclude_unset=True, by_alias=True)
                )
            
            # Parse the response content as JSON
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
                
            llm_result = _parse_json_with_repair(content)

            # Check if response_model is a List type
            is_list_type = get_origin(response_model) is list

            if is_list_type:
                # Get the type inside the List
                item_model = get_args(response_model)[0]
                validated_result = _validate_list_output(llm_result, item_model)
                logger.debug(
                    f"Successfully processed LLM request with model '{model_id}', returned {len(validated_result)} validated items"
                )
            else:
                # Validate as a single item
                assert issubclass(response_model, BaseModel)
                validated_result = _validate_single_output(llm_result, response_model)
                logger.debug(
                    f"Successfully processed LLM request with model '{model_id}', returned a validated item"
                )

            return validated_result
            
        finally:
            try:
                await client.close()
            except Exception as close_error:
                logger.warning(f"Error closing client: {close_error}")

    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"LLM processing failed with model '{model_id}': {error_msg}", exc_info=True
        )
        
        if "nodename nor servname provided" in error_msg or "Connection error" in error_msg:
            logger.warning(f"Network connectivity issue detected, adding delay before retry")
            await asyncio.sleep(1)

        # Always use fallback strategy (try next model)
        if attempt < len(model_ids):
            return await call_llm_with_fallback(
                response_model, input_data, prompt_template, model_ids, attempt + 1, settings
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
    settings: Optional[LLMSettings] = None,
) -> Dict[str, Any]:
    """
    Call LLM with function call support using direct OpenAI-style API.
    
    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        function_definition: OpenAI-style function definition
        model_ids: List of model IDs to try in fallback
        attempt: Current attempt number
        settings: Optional LLM settings to use, defaults to global settings
        
    Returns:
        Dict containing the function call arguments
        
    Raises:
        ValueError: If function call fails or validation fails
    """
    # Use provided settings or fall back to global settings
    if settings is None:
        settings = llm_settings
    
    model_id = model_ids[attempt - 1] if attempt <= len(model_ids) else "unknown"
    
    try:
        logger.debug(f"Function call attempt {attempt}/{len(model_ids)}: Using model '{model_id}'")
        
        # Get configuration
        base_url = settings.openai_base_url
        api_key = settings.openai_api_key
        
        logger.debug(f"Using base_url: {base_url}")
        logger.debug(f"API key length: {len(api_key) if api_key else 0}")
        
        # Initialize client with proper resource management and timeout settings
        httpx_client = settings.get_httpx_client() if settings.enable_json_log else None
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=180.0,
            max_retries=0,
            http_client=httpx_client
        )
        cache_provider = get_llm_cache_provider()
        
        try:
            # Make the function call using the new tools format
            tool = {
                "type": "function",
                "function": function_definition
            }
            request = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "tools": [tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": function_definition["name"]}
                },
                "temperature": 0.1,
            }
            cache_key = cache_provider.hash_params(request)
            cached_response = cache_provider.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for key: {cache_key}")
                # Cache hit, return the cached response
                response = ChatCompletion.model_validate_json(cached_response)
            else: 
                logger.debug(f"Cache miss for key: {cache_key}")
                response: ChatCompletion = await client.chat.completions.create(**request)
                cache_provider.insert(
                    cache_key, 
                    request,
                    response.model_dump(exclude_none=True, exclude_unset=True, by_alias=True)
                )
                
            message = response.choices[0].message
            if message.tool_calls and len(message.tool_calls) > 0:
                function_args = json.loads(message.tool_calls[0].function.arguments)
                logger.debug(f"Successfully received function call response with model '{model_id}'")
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
                system_prompt, user_prompt, function_definition, model_ids, attempt + 1, settings
            )
        else:
            raise
