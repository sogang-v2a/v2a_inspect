from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings

class ServerSettings(BaseSettings):
    upload_dir: Path = Path(os.getenv("V2A_UPLOAD_DIR", "/tmp/v2a_uploads"))
    host: str = "127.0.0.1"
    port: int = 8080
    sam3_model_id: str = "facebook/sam3"

    class Config:
        env_prefix = "V2A_SERVER_"

settings = ServerSettings()

# Ensure the upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
