from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for CodeRetrX tools."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Tools that are available.
    coderetrx_available_tools: List[str] = []

    @property
    def available_tools(self) -> List[str]:
        return self.coderetrx_available_tools


settings = Settings()
