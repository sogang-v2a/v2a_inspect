import os
import random
import numpy as np
import torch  # type: ignore
import torchaudio  # type: ignore
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager

from loguru import logger  # type: ignore
from hunyuanvideo_foley.utils.model_utils import load_model  # type: ignore
from hunyuanvideo_foley.utils.feature_utils import feature_process  # type: ignore
from hunyuanvideo_foley.utils.model_utils import denoise_process  # type: ignore

app = FastAPI(title="HunyuanVideo-Foley V2A API")

# Global variables to hold the loaded model
model_dict = None
cfg = None


def set_manual_seed(global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)


def startup_event():
    global model_dict, cfg
    logger.info("Initializing HunyuanVideo-Foley model...")
    model_path = os.environ.get("HUNYUAN_MODEL_PATH", "HunyuanVideo-Foley")
    model_size = os.environ.get("HUNYUAN_MODEL_SIZE", "xxl")
    enable_offload = os.environ.get("HUNYUAN_ENABLE_OFFLOAD", "true").lower() == "true"

    config_path = f"configs/hunyuanvideo-foley-{model_size}.yaml"

    try:
        model_dict, cfg = load_model(
            model_path=model_path,
            config_path=config_path,
            device=torch.device("cuda"),
            enable_offload=enable_offload,
            model_size=model_size,
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Not exiting so the server can start and show errors on request


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    startup_event()
    yield


app.router.lifespan_context = lifespan


def infer(
    video_path, prompt, guidance_scale=4.5, num_inference_steps=50, neg_prompt=None
):
    set_manual_seed(42)
    visual_feats, text_feats, audio_len_in_s = feature_process(
        video_path, prompt, model_dict, cfg, neg_prompt=neg_prompt
    )
    audio, sample_rate = denoise_process(
        visual_feats,
        text_feats,
        audio_len_in_s,
        model_dict,
        cfg,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    return audio[0], sample_rate


@app.post("/generate_v2a")
async def generate_v2a_endpoint(
    video: UploadFile = File(...),
    prompt: str = Form(""),
    guidance_scale: float = Form(4.5),
    num_inference_steps: int = Form(50),
):
    if model_dict is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Save uploaded video
        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, f"input_{uuid.uuid4().hex}.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        logger.info(
            f"Processing V2A request for video: {video.filename}, prompt: {prompt}"
        )

        # Run inference
        audio_tensor, sample_rate = infer(
            video_path,
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        # Save output
        out_wav = os.path.join(tmp_dir, f"output_{uuid.uuid4().hex}.wav")
        torchaudio.save(out_wav, audio_tensor, sample_rate)

        return FileResponse(
            out_wav,
            media_type="audio/wav",
            filename="generated.wav",
            background=None,  # Ideally cleanup tmp_dir using BackgroundTasks
        )

    except Exception as e:
        logger.exception("Error during generation")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
