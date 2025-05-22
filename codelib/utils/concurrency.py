import logging
from typing import Callable, Any, TypeVar, Coroutine, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def abatch_func_call(
    max_concurrency: int, func: Callable[..., Any], kwargs_list: list[dict]
) -> list[Any]:
    """Execute a batch of function calls with controlled concurrency.

    Handles both async and sync functions automatically:
    - Async functions run directly in the event loop
    - Sync functions run in thread pool executor to avoid blocking

    Features:
    - Semaphore-based concurrency control
    - Automatic error logging with function context
    - Preserves original exception stack traces

    Args:
        max_concurrency: Maximum parallel executions allowed
        func: Callable to execute (async or sync)
        kwargs_list: List of keyword arguments dictionaries for each call

    Returns:
        list: Results in the same order as kwargs_list, exceptions will propagate

    Raises:
        Exception: Re-raises the first encountered exception from any task
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def a_func_call(func: Callable, kwargs: dict) -> Any:
        """Execute a single function call with concurrency control.

        Args:
            func: Target function to execute
            kwargs: Keyword arguments for this call

        Returns:
            Any: Result of the function call

        Raises:
            Exception: Propagates any exceptions from the function call
        """
        async with semaphore:  # Acquire semaphore slot
            try:
                if asyncio.iscoroutinefunction(func):
                    # Directly await async functions
                    result = await func(**kwargs)
                else:
                    # Run sync functions in thread pool to prevent blocking
                    result = await asyncio.to_thread(func, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__} with args {kwargs}: {e}")
                raise  # Re-raise to maintain stack trace

    # Create and schedule all tasks
    tasks = [a_func_call(func, kwargs) for kwargs in kwargs_list]

    # Execute all tasks concurrently with controlled concurrency
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return results


def run_coroutine_sync(
    coroutine: Coroutine[Any, Any, T], timeout: Optional[float] = None
) -> T:
    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
