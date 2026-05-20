from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "files"


class Settings(BaseSettings):
    prompts_dir: Path = DEFAULT_PROMPTS_DIR
    prompt_file_suffix: str = ".txt"

    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_BASE_URL",
            "V2A_LLM_BASE_URL",
        ),
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_API_KEY",
            "V2A_LLM_API_KEY",
            "GEMINI_API_KEY",
            "API_KEY",
        ),
    )
    llm_small_model: str = Field(
        default="gemini-3-flash-preview",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_SMALL_MODEL",
            "V2A_LLM_SMALL_MODEL",
        ),
    )
    llm_medium_model: str = Field(
        default="gemini-3-pro-preview",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_MEDIUM_MODEL",
            "V2A_LLM_MEDIUM_MODEL",
            "V2A_INSPECT_LLM_MODEL",
            "V2A_LLM_MODEL",
        ),
    )
    llm_large_model: str = Field(
        default="gemini-3-pro-preview",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_LARGE_MODEL",
            "V2A_LLM_LARGE_MODEL",
        ),
    )
    llm_temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_TEMPERATURE",
            "V2A_LLM_TEMPERATURE",
        ),
    )
    llm_timeout_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_TIMEOUT_SECONDS",
            "V2A_LLM_TIMEOUT_SECONDS",
        ),
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_MAX_RETRIES",
            "V2A_LLM_MAX_RETRIES",
        ),
    )

    model_config = SettingsConfigDict(env_prefix="V2A_INSPECT_")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
