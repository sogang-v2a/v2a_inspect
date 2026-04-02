from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SecretsSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    # Did not include model name here because it is dynamic
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "API_KEY"),
    )
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    )
    gpu_provider_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GPU_PROVIDER_API_KEY", "RUNPOD_API_KEY"),
    )
    openrouter_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str | None = None
    langfuse_environment: str = "local"
    langfuse_release: str | None = None
    langfuse_sample_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    prompt_backend: Literal["local", "auto", "langfuse"] = Field(
        default="local",
        validation_alias=AliasChoices("PROMPT_BACKEND", "LANGFUSE_PROMPT_BACKEND"),
    )
    langfuse_prompt_label: str = "production"
    auth_mode: Literal["disabled", "password"] = "password"
    auth_allow_self_signup: bool = True
    auth_cookie_key: SecretStr | None = None
    auth_cookie_name: str = "v2a_inspect_cookie"
    auth_cookie_expiry_days: int = Field(default=1, ge=1)
    auth_credentials_path: Path | None = None
    ui_analysis_concurrency_limit: int = Field(default=2, ge=1)
    ui_analysis_acquire_timeout_seconds: int = Field(default=120, ge=1)
    ui_temp_cleanup_max_age_seconds: int = Field(default=3600, ge=1)
    ui_cleanup_interval_seconds: int = Field(default=1800, ge=1)
    gpu_provider: str = "runpod"
    provider_mode: Literal["sync_endpoint", "async_job"] = "sync_endpoint"
    provider_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GPU_PROVIDER_BASE_URL", "RUNPOD_BASE_URL"),
    )
    sam3_service: str = "sam3"
    embedding_service: str = "embedding"
    label_service: str = "label"
    model_cache_dir: Path = Path(".cache/v2a_inspect_server/models")
    weights_manifest_path: Path = Path("server/model-manifest.json")
    remote_timeout_seconds: int = Field(default=120, ge=1)
    remote_gpu_preference: Literal["A4000", "A4500"] = "A4000"
    remote_gpu_fallback: Literal["A4000", "A4500"] = "A4500"
    remote_gpu_vram_preference_gb: int = Field(default=16, ge=1, le=24)
    remote_gpu_vram_cap_gb: int = Field(default=24, ge=1, le=24)
    visual_pipeline_mode: Literal["legacy_gemini", "tool_first_foundation"] = (
        "legacy_gemini"
    )

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.secure"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]

        secrets_dir = Path("/run/secrets")
        if secrets_dir.is_dir():
            sources.append(SecretsSettingsSource(settings_cls, secrets_dir=secrets_dir))

        return tuple(sources)

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if (self.langfuse_public_key is None) != (self.langfuse_secret_key is None):
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must either both be set or both be omitted."
            )

        if self.remote_gpu_preference == self.remote_gpu_fallback:
            raise ValueError(
                "REMOTE_GPU_PREFERENCE and REMOTE_GPU_FALLBACK must be different SKUs."
            )

        if self.remote_gpu_vram_preference_gb > self.remote_gpu_vram_cap_gb:
            raise ValueError(
                "REMOTE_GPU_VRAM_PREFERENCE_GB cannot exceed REMOTE_GPU_VRAM_CAP_GB."
            )

        if not self.gpu_provider.strip():
            raise ValueError("GPU_PROVIDER must not be empty.")

        return self


settings = Settings()
