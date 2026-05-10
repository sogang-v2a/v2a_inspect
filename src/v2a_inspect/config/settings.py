from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "files"


class Settings(BaseSettings):
    prompts_dir: Path = DEFAULT_PROMPTS_DIR
    prompt_file_suffix: str = ".txt"

    model_config = SettingsConfigDict(env_prefix="V2A_INSPECT_")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
