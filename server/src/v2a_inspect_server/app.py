from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from v2a_inspect_server.models import Sam3ExtractRequest
from v2a_inspect_server.sam3_inference import Sam3InferenceClient
from v2a_inspect_server.settings import settings

sam3_client: Sam3InferenceClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam3_client
    # Initialize the SAM3 client on startup
    sam3_client = Sam3InferenceClient()
    yield
    # Cleanup on shutdown
    sam3_client = None

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
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    return {"video_id": video_id}

@app.post("/infer/sam3")
async def infer_sam3(request: Sam3ExtractRequest):
    if sam3_client is None:
        raise HTTPException(status_code=503, detail="SAM3 client not initialized")
    try:
        return sam3_client.process(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
