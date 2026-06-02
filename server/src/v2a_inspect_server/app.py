from __future__ import annotations

import logging
import shutil
import traceback
import uuid
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from v2a_inspect_server.models import (
    DinoV2EmbedImagesRequest,
    LabelScoreRequest,
    Sam3SegmentFramesRequest,
    Sam3TrackVideoRequest,
)
from v2a_inspect_server.inference.sam3_image import Sam3ImageInferenceClient
from v2a_inspect_server.inference.sam3 import Sam3InferenceClient
from v2a_inspect_server.inference.embed import DinoV2InferenceClient
from v2a_inspect_server.inference.score import Siglip2InferenceClient
from v2a_inspect_server.settings import settings

sam3_client: Sam3InferenceClient | None = None
sam3_image_client: Sam3ImageInferenceClient | None = None
embed_client: DinoV2InferenceClient | None = None
score_client: Siglip2InferenceClient | None = None
logger = logging.getLogger("uvicorn.error")


def _exception_detail(error: Exception) -> dict[str, str]:
    return {
        "error": str(error),
        "type": type(error).__name__,
        "traceback": traceback.format_exc(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam3_client, sam3_image_client, embed_client, score_client
    # Initialize the clients on startup
    sam3_client = Sam3InferenceClient()
    sam3_image_client = Sam3ImageInferenceClient()
    embed_client = DinoV2InferenceClient()
    score_client = Siglip2InferenceClient()
    yield
    # Cleanup on shutdown
    if sam3_client is not None:
        sam3_client.close()
    if sam3_image_client is not None:
        sam3_image_client.close()
    sam3_client = sam3_image_client = embed_client = score_client = None


app = FastAPI(title="v2a-inspect-server", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Only allow basic video extensions
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    video_id = str(uuid.uuid4())
    save_path = settings.upload_dir / f"{video_id}{ext}"

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("Failed to upload video")
        detail = _exception_detail(e)
        detail["error"] = f"Could not save file: {detail['error']}"
        raise HTTPException(status_code=500, detail=detail)

    return {"video_id": video_id}


@app.post("/infer/sam3/track-video")
async def track_video_sam3(request: Sam3TrackVideoRequest):
    if sam3_client is None:
        raise HTTPException(status_code=503, detail="SAM3 client not initialized")
    try:
        return sam3_client.track_video(request)
    except FileNotFoundError as e:
        logger.exception("SAM3 track-video file not found")
        raise HTTPException(status_code=404, detail=_exception_detail(e))
    except Exception as e:
        logger.exception("SAM3 track-video failed")
        raise HTTPException(status_code=500, detail=_exception_detail(e))


@app.post("/infer/sam3/segment-frames")
async def segment_frames_sam3(request: Sam3SegmentFramesRequest):
    if sam3_image_client is None:
        raise HTTPException(status_code=503, detail="SAM3 image client not initialized")
    try:
        return sam3_image_client.segment_frames(request)
    except FileNotFoundError as e:
        logger.exception("SAM3 segment-frames file not found")
        raise HTTPException(status_code=404, detail=_exception_detail(e))
    except Exception as e:
        logger.exception("SAM3 segment-frames failed")
        raise HTTPException(status_code=500, detail=_exception_detail(e))


@app.post("/infer/dinov2/embed-images")
async def embed_images_dinov2(request: DinoV2EmbedImagesRequest):
    if embed_client is None:
        raise HTTPException(
            status_code=503, detail="DINOv2 embedding client not initialized"
        )
    try:
        return embed_client.embed_images(request)
    except FileNotFoundError as e:
        logger.exception("DINOv2 embed-images file not found")
        raise HTTPException(status_code=404, detail=_exception_detail(e))
    except Exception as e:
        logger.exception("DINOv2 embed-images failed")
        raise HTTPException(status_code=500, detail=_exception_detail(e))


@app.post("/infer/score")
async def score_labels(request: LabelScoreRequest):
    if score_client is None:
        raise HTTPException(
            status_code=503, detail="SigLIP2 scoring client not initialized"
        )
    try:
        return score_client.score(request)
    except FileNotFoundError as e:
        logger.exception("Score labels file not found")
        raise HTTPException(status_code=404, detail=_exception_detail(e))
    except Exception as e:
        logger.exception("Score labels failed")
        raise HTTPException(status_code=500, detail=_exception_detail(e))
