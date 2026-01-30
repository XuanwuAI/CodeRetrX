from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for CodeRetrX tools."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Tools that are disabled by default
    coderetrx_disabled_tools: List[str] = ["codeql_query"]

    @property
    def disabled_tools(self) -> List[str]:
        return self.coderetrx_disabled_tools


settings = Settings()
