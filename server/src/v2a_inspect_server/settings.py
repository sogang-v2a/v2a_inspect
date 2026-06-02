from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    upload_dir: Path = Path(os.getenv("V2A_UPLOAD_DIR", "/tmp/v2a_uploads"))
    host: str = "0.0.0.0"
    port: int = 8080
    sam31_max_num_objects: int = 48
    sam31_use_fa3: bool = False
    sam31_use_rope_real: bool = True
    sam31_compile: bool = False
    sam31_warm_up: bool = False
    opencv_video_backend: str = "ffmpeg"
    opencv_hw_acceleration: bool = True
    opencv_hw_device: int | None = None
    opencv_ffmpeg_capture_options: str | None = "hw_decoders_any;cuda"
    embedding_model_id: str = "facebook/dinov2-base"
    label_model_id: str = "google/siglip2-base-patch16-224"
    sam3_bpe_url: str = (
        "https://raw.githubusercontent.com/openai/CLIP/main/clip/"
        "bpe_simple_vocab_16e6.txt.gz"
    )
    sam3_bpe_path: Path = Path(
        os.getenv(
            "V2A_SERVER_SAM3_BPE_PATH",
            "/data/model-cache/sam3/bpe_simple_vocab_16e6.txt.gz",
        )
    )

    class Config:
        env_prefix = "V2A_SERVER_"


settings = ServerSettings()

# Ensure the upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
