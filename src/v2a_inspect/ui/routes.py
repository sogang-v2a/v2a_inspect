from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .pipeline import PipelineOptions, run_uploaded_video_pipeline
from .rows import current_frame_rows, timeline_rows
from .store import VideoAssetSnapshot, VideoAssetStore


def create_router(store: VideoAssetStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/asset")
    async def get_asset() -> dict[str, object]:
        snapshot = await store.snapshot()
        return _snapshot_payload(snapshot)

    @router.get("/api/video")
    async def get_video() -> FileResponse:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        return FileResponse(snapshot.asset.source_path)

    @router.get("/api/rows/timeline")
    async def get_timeline_rows() -> dict[str, object]:
        snapshot = await store.snapshot()
        rows = [] if snapshot.asset is None else timeline_rows(snapshot.asset)
        return {"version": snapshot.version, "rows": rows}

    @router.get("/api/rows/current-frame")
    async def get_current_frame(frame: int) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            return {"version": snapshot.version, "frame": frame, "rows": None}
        return {
            "version": snapshot.version,
            "frame": frame,
            "rows": current_frame_rows(snapshot.asset, frame),
        }

    @router.post("/api/runs")
    async def start_run(
        background_tasks: BackgroundTasks,
        video: Annotated[UploadFile, File()],
        work_dir: Annotated[str | None, Form()] = None,
        server_url: Annotated[str | None, Form()] = None,
        scene_threshold: Annotated[float, Form()] = 27.0,
        max_keyframes_per_scene: Annotated[int, Form()] = 20,
    ) -> dict[str, object]:
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video filename provided")

        root_dir = Path(work_dir or tempfile.gettempdir()) / "v2a-inspect-ui"
        upload_dir = root_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{uuid.uuid4()}{Path(video.filename).suffix}"
        upload_path.write_bytes(await video.read())

        options = PipelineOptions(
            scene_threshold=scene_threshold,
            max_keyframes_per_scene=max_keyframes_per_scene,
            server_url=server_url or None,
        )
        background_tasks.add_task(
            run_uploaded_video_pipeline,
            upload_path,
            root_dir,
            store,
            options,
        )
        await store.set_running(stage="queued")
        return {"status": "queued"}

    @router.get("/events")
    async def events() -> StreamingResponse:
        async def event_stream():
            snapshot = await store.snapshot()
            yield _sse(snapshot)
            version = snapshot.version
            while True:
                snapshot = await store.wait_for_change(version)
                version = snapshot.version
                yield _sse(snapshot)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router


def _snapshot_payload(snapshot: VideoAssetSnapshot) -> dict[str, object]:
    return {
        "status": snapshot.status,
        "stage": snapshot.current_stage,
        "error": snapshot.error,
        "version": snapshot.version,
        "updated_at": snapshot.updated_at.isoformat(),
        "asset": None
        if snapshot.asset is None
        else snapshot.asset.model_dump(mode="json", exclude_computed_fields=True),
        "timeline_rows": []
        if snapshot.asset is None
        else timeline_rows(snapshot.asset),
    }


def _sse(snapshot: VideoAssetSnapshot) -> str:
    data = {
        "version": snapshot.version,
        "status": snapshot.status,
        "stage": snapshot.current_stage,
        "error": snapshot.error,
    }
    return f"event: asset_update\ndata: {json.dumps(data)}\n\n"
