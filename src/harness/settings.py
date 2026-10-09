"""for pydnatic controls"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.getenv("APP_ENV", "dev")  # dev | staging | prod


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=(".env", f".env.{APP_ENV}"),  # later file wins
        extra="ignore",
    )

    model: str = "openai:gpt-5.6-luna"
    routing_model: str = "openai:gpt-4o-mini"
    temperature: float = 0
    reasoning_effort: str = "none"
    prompt_cache_enabled: bool = False
    prompt_cache_key: str = "harness-v1"

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")


settings = Settings()