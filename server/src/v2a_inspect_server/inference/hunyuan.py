from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from ..settings import settings
from ..models.hunyuan import HunyuanGenerateV2ARequest

logger = logging.getLogger("uvicorn.error")

try:
    import torch
    import torchaudio
    from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
    from hunyuanvideo_foley.utils.feature_utils import feature_process

    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False


class HunyuanInferenceClient:
    def __init__(self) -> None:
        if not HUNYUAN_AVAILABLE:
            logger.warning(
                "HunyuanVideo-Foley is not installed. V2A generation will fail."
            )
            self.model_dict = None
            self.cfg = None
            return

        logger.info("Initializing HunyuanVideo-Foley model...")
        model_size = settings.hunyuan_model_size

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.pytorch_cuda_alloc_conf

        config_path = f"configs/hunyuanvideo-foley-{model_size}.yaml"

        try:
            self.model_dict, self.cfg = load_model(
                model_path=settings.hunyuan_model_id,
                config_path=config_path,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                enable_offload=settings.hunyuan_enable_offload,
                model_size=model_size,
            )
            logger.info("HunyuanVideo-Foley model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load HunyuanVideo-Foley model: %s", e)
            self.model_dict = None
            self.cfg = None

    def close(self) -> None:
        if self.model_dict is not None:
            # Free memory
            del self.model_dict
            self.model_dict = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_v2a(self, request: HunyuanGenerateV2ARequest) -> str:
        if not HUNYUAN_AVAILABLE or self.model_dict is None:
            raise RuntimeError("HunyuanVideo-Foley model is not loaded.")

        start_time = time.perf_counter()
        video_path = self._find_video_path(request.video_id)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps if total_frames > 0 else 0
        cap.release()

        start_s = request.start_frame_index / fps
        end_s = request.end_frame_index / fps

        # Clamp to avoid ffmpeg seeking past EOF
        if video_duration > 0:
            start_s = max(0.0, min(start_s, video_duration - 0.1))
            end_s = max(0.0, min(end_s, video_duration))

        # Re-encoding을 피하기 위해 tpad 필터를 사용할 수 없으므로,
        # 최소 1.5초가 필요하다면 원본 비디오 내에서 구간(window) 자체를 1.5초로 확장합니다.
        actual_duration = end_s - start_s
        if actual_duration < 1.5:
            end_s = start_s + 1.5
            if video_duration > 0 and end_s > video_duration:
                end_s = video_duration
                start_s = max(0.0, end_s - 1.5)

        final_duration = end_s - start_s
        if final_duration <= 0:
            final_duration = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            # -c copy 시 포맷 호환성을 위해 원본 확장자 유지
            ext = video_path.suffix
            tmp_video_path = str(Path(temp_dir) / f"cropped{ext}")

            if final_duration < 1.5:
                # 원본 비디오 자체가 1.5초보다 짧은 극단적인 경우, 어쩔 수 없이 tpad와 재인코딩 사용
                padded_duration = 1.5
                codec = "h264_nvenc" if settings.enable_nvenc else "libx264"
                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-i",
                        str(video_path),
                        "-ss",
                        str(start_s),
                        "-t",
                        str(padded_duration),
                        "-vf",
                        f"tpad=stop_mode=clone:stop_duration={padded_duration}",
                        "-c:v",
                        codec,
                        "-an",
                        tmp_video_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # 길이가 충분한 경우 재인코딩 없이 빠르게 자르기 위해 -c:v copy 사용
                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-ss",
                        str(start_s),
                        "-t",
                        str(final_duration),
                        "-i",
                        str(video_path),
                        "-c:v",
                        "copy",
                        "-an",
                        tmp_video_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Generate audio using the cropped video
            audio_tensor, sample_rate = self._infer(
                tmp_video_path,
                request.prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                neg_prompt=request.negative_prompt,
            )

            # Save audio to a temp file
            out_audio_path = str(Path(temp_dir) / "output.wav")
            torchaudio.save(out_audio_path, audio_tensor.cpu(), sample_rate)

            logger.info(
                f"Hunyuan generation took {time.perf_counter() - start_time:.2f}s"
            )

            final_audio_path = settings.upload_dir / f"audio_{uuid.uuid4().hex}.wav"

            shutil.copy(out_audio_path, final_audio_path)

        return str(final_audio_path)

    def _infer(
        self,
        video_path: str,
        prompt: str,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 50,
        neg_prompt: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        self._set_manual_seed(42)
        visual_feats, text_feats, audio_len_in_s = feature_process(
            video_path, prompt, self.model_dict, self.cfg, neg_prompt=neg_prompt
        )
        audio, sample_rate = denoise_process(
            visual_feats,
            text_feats,
            audio_len_in_s,
            self.model_dict,
            self.cfg,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        return audio[0], sample_rate

    def _set_manual_seed(self, global_seed: int) -> None:
        random.seed(global_seed)
        np.random.seed(global_seed)
        if torch.cuda.is_available():
            torch.manual_seed(global_seed)

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")
