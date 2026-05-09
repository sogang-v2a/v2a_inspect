from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from v2a_inspect_server.settings import settings

app = FastAPI(title="v2a-inspect-server")

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
