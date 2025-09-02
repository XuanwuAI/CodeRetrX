import asyncio
import logging
from pathlib import Path
from typing import ClassVar, Optional

from coderetrx.utils.path import get_data_dir, get_repo_path
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from typing import Any, Type
from pydantic import BaseModel
from abc import abstractmethod

logger = logging.getLogger(__name__)


class BaseTool:
    """Base class for tools that work with repositories."""

    name: str
    description: str
    args_schema: ClassVar[Type[BaseModel]]

    def __init__(self, repo_url: str, uuid: Optional[str] = None):
        super().__init__()
        logger.info(f"Init base repo tool {self.name} with uuid: {uuid} ...")

        self.repo_url = repo_url
        self.repo_id = get_repo_id(repo_url)
        self.uuid = uuid
        self.repo_path = get_repo_path(repo_url)

        clone_repo_if_not_exists(repo_url, str(self.repo_path))

    def run_sync(self, *args, **kwargs):
        """Synchronous wrapper for async _run method."""
        try:
            loop = asyncio.get_running_loop()
            # If we're already in a running loop, we need to use a different approach
            import concurrent.futures
            import threading

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.run(*args, **kwargs))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()

        except RuntimeError:
            # No event loop is running, we can create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.run(*args, **kwargs))
            finally:
                loop.close()

    async def _run_repr(self, *args, **kwargs: dict[str, str]):
        result = await self._run(*args, **kwargs)
        if not result:
            return "No result Found."
        if type(result) == str:
            return result
        return type(result[0]).repr(result)

    @abstractmethod
    async def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Async implementation to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _run method")

    async def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        repr_output = kwargs.pop("repr_output", True)
        if repr_output:
            return await self._run_repr(*args, **kwargs)
        else:
            return await self._run(*args, **kwargs)
