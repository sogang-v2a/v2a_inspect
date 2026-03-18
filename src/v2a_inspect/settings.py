from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):
    # Did not include model name here because it is dynamic
    gemini_api_key: Optional[SecretStr] = None
    openrouter_api_key: Optional[SecretStr] = None

    class Config:
        env_file = [".env", ".env.secure"]
        env_file_encoding = "utf-8"
        secrets_dir = "/run/secrets"  # for Docker secrets

    @classmethod
    def validate_api_keys(cls, values):
        if not values.get("gemini_api_key") and not values.get("openrouter_api_key"):
            raise ValueError(
                "At least one of GEMINI_API_KEY or OPENROUTER_API_KEY must be set."
            )
        return values


settings = Settings()
