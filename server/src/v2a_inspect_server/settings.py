from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    upload_dir: Path = Path(os.getenv("V2A_UPLOAD_DIR", "/tmp/v2a_uploads"))
    host: str = "0.0.0.0"
    port: int = 8080
    sam3_model_id: str = "facebook/sam3"
    sam3_image_size: int = 1008
    sam3_dtype: str = "bfloat16"
    sam3_attention_implementation: str = "sdpa"
    sam3_score_threshold_detection: float = 0.35
    sam3_new_det_thresh: float = 0.35
    opencv_video_backend: str = "ffmpeg"
    opencv_hw_acceleration: bool = True
    opencv_hw_device: int | None = None
    opencv_ffmpeg_capture_options: str | None = "hw_decoders_any;cuda"
    embedding_model_id: str = "facebook/dinov2-base"
    label_model_id: str = "google/siglip2-base-patch16-224"

    class Config:
        env_prefix = "V2A_SERVER_"


settings = ServerSettings()

# Ensure the upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
