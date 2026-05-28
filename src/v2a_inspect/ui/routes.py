from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import ValidationError

from v2a_inspect.media_utils import probe_prepared_video
from v2a_inspect.models import VideoAsset
from v2a_inspect.preprocessing import prepare_video

from .overlays import (
    MASK_ATLAS_MAX_FRAMES,
    MASK_ATLAS_TILE_SIZE,
    render_tracking_mask_atlas,
    render_tracking_overlay,
)
from .pipeline import (
    PipelineOptions,
    run_sound_timeline_pipeline,
    run_uploaded_video_pipeline,
)
from .rows import current_frame_rows, timeline_rows, tracking_window_rows
from .store import VideoAssetSnapshot, VideoAssetStore


DEFAULT_WORK_DIR = os.getenv("V2A_INSPECT_UI_WORK_DIR")
DEFAULT_SERVER_URL = os.getenv("V2A_INSPECT_UI_SERVER_URL")
DEFAULT_SCENE_THRESHOLD = float(os.getenv("V2A_INSPECT_UI_SCENE_THRESHOLD", "27.0"))
DEFAULT_MAX_KEYFRAMES_PER_SCENE = int(
    os.getenv("V2A_INSPECT_UI_MAX_KEYFRAMES_PER_SCENE", "20")
)


def create_router(store: VideoAssetStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/asset")
    async def get_asset() -> dict[str, object]:
        snapshot = await store.snapshot()
        return _full_snapshot_payload(snapshot)

    @router.get("/api/asset/export")
    async def export_asset() -> Response:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        return Response(
            snapshot.asset.model_dump_json(
                indent=2,
                exclude_computed_fields=True,
            ),
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="video-asset.json"',
            },
        )

    @router.get("/api/asset-summary")
    async def get_asset_summary() -> dict[str, object]:
        snapshot = await store.snapshot()
        return _summary_payload(snapshot)

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

    @router.get("/api/tracks/window")
    async def get_tracking_window(
        start_frame: int, end_frame: int
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            return {
                "version": snapshot.version,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "tracks": [],
            }
        start = max(0, start_frame)
        end = min(snapshot.asset.frame_count - 1, end_frame)
        if end < start:
            raise HTTPException(status_code=400, detail="Invalid frame window")
        return {
            "version": snapshot.version,
            "start_frame": start,
            "end_frame": end,
            "tracks": tracking_window_rows(snapshot.asset, start, end),
        }

    @router.get("/api/frames/tracking-overlay")
    async def get_tracking_overlay(
        frame: int,
        masks: bool = True,
        boxes: bool = True,
        labels: bool = True,
    ) -> Response:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        if frame < 0 or frame >= snapshot.asset.frame_count:
            raise HTTPException(status_code=400, detail="Frame out of range")
        return Response(
            render_tracking_overlay(
                snapshot.asset,
                frame,
                masks=masks,
                boxes=boxes,
                labels=labels,
            ),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/api/frames/tracking-mask-atlas")
    async def get_tracking_mask_atlas(
        start_frame: int,
        end_frame: int,
    ) -> Response:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        if end_frame < start_frame:
            raise HTTPException(status_code=400, detail="Invalid frame window")
        start = max(0, start_frame)
        end = min(snapshot.asset.frame_count - 1, end_frame)
        end = min(end, start + MASK_ATLAS_MAX_FRAMES - 1)
        if end < start:
            raise HTTPException(status_code=400, detail="Invalid frame window")
        tile_width, tile_height = MASK_ATLAS_TILE_SIZE
        return Response(
            render_tracking_mask_atlas(snapshot.asset, start, end),
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Start-Frame": str(start),
                "X-End-Frame": str(end),
                "X-Tile-Width": str(tile_width),
                "X-Tile-Height": str(tile_height),
            },
        )

    @router.post("/api/runs")
    async def start_run(
        background_tasks: BackgroundTasks,
        video: Annotated[UploadFile, File()],
        work_dir: Annotated[str | None, Form()] = None,
        server_url: Annotated[str | None, Form()] = None,
        scene_threshold: Annotated[float, Form()] = DEFAULT_SCENE_THRESHOLD,
        max_keyframes_per_scene: Annotated[
            int, Form()
        ] = DEFAULT_MAX_KEYFRAMES_PER_SCENE,
    ) -> dict[str, object]:
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video filename provided")

        root_dir = _root_dir(work_dir)
        upload_path = _write_upload(root_dir, video.filename, await video.read())

        options = PipelineOptions(
            scene_threshold=scene_threshold,
            max_keyframes_per_scene=max_keyframes_per_scene,
            server_url=server_url or DEFAULT_SERVER_URL or None,
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

    @router.post("/api/asset/import")
    async def import_asset(
        video: Annotated[UploadFile, File()],
        asset: Annotated[UploadFile, File()],
        work_dir: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.status == "running":
            raise HTTPException(status_code=409, detail="Pipeline is running")
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video filename provided")
        if not asset.filename:
            raise HTTPException(status_code=400, detail="No asset filename provided")

        try:
            imported_asset = VideoAsset.model_validate_json(await asset.read())
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid VideoAsset JSON: {exc}",
            ) from exc

        root_dir = _root_dir(work_dir)
        upload_path = _write_upload(
            root_dir,
            video.filename,
            await video.read(),
        )
        try:
            resolved_video_asset = await _resolve_import_video(upload_path, root_dir)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video: {exc}",
            ) from exc
        if resolved_video_asset.frame_count != imported_asset.frame_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Imported video frame count does not match VideoAsset: "
                    f"{resolved_video_asset.frame_count} != "
                    f"{imported_asset.frame_count}"
                ),
            )

        sam3_tracking_path = resolved_video_asset.sam3_tracking_path
        if sam3_tracking_path is not None and not sam3_tracking_path.exists():
            sam3_tracking_path = None
        imported_asset = imported_asset.model_copy(
            update={
                "source_path": resolved_video_asset.source_path,
                "sam3_tracking_path": sam3_tracking_path,
            }
        )
        await store.set_complete_asset(imported_asset, stage="imported asset")
        return {"status": "imported"}

    @router.post("/api/sound-timeline/reset-run")
    async def reset_sound_timeline(
        background_tasks: BackgroundTasks,
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.status == "running":
            raise HTTPException(status_code=409, detail="Pipeline is running")
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        video_asset = snapshot.asset.model_copy(update={"sound_timeline": None})
        await store.publish_asset_mutation(
            video_asset,
            stage="reset sound timeline",
        )
        background_tasks.add_task(run_sound_timeline_pipeline, video_asset, store)
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


def _root_dir(work_dir: str | None) -> Path:
    root_dir = Path(work_dir or DEFAULT_WORK_DIR or tempfile.gettempdir())
    if root_dir.name != "v2a-inspect-ui":
        root_dir /= "v2a-inspect-ui"
    return root_dir


def _write_upload(root_dir: Path, filename: str, data: bytes) -> Path:
    upload_dir = root_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{uuid.uuid4()}{Path(filename).suffix}"
    upload_path.write_bytes(data)
    return upload_path


async def _resolve_import_video(upload_path: Path, root_dir: Path) -> VideoAsset:
    try:
        probe = probe_prepared_video(upload_path)
    except Exception:
        return await asyncio.to_thread(prepare_video, upload_path, root_dir)
    return VideoAsset(
        source_path=probe.path,
        frame_count=probe.frame_count,
    )


def _summary_payload(snapshot: VideoAssetSnapshot) -> dict[str, object]:
    asset = snapshot.asset
    return {
        "status": snapshot.status,
        "stage": snapshot.current_stage,
        "error": snapshot.error,
        "version": snapshot.version,
        "asset_version": snapshot.asset_version,
        "updated_at": snapshot.updated_at.isoformat(),
        "video": None
        if asset is None
        else {
            "video_id": str(asset.video_id),
            "frame_count": asset.frame_count,
            "fps": asset.fps,
            "duration_sec": asset.duration_sec,
            "width": asset.width,
            "height": asset.height,
        },
        "timeline_rows": [] if asset is None else timeline_rows(asset),
    }


def _full_snapshot_payload(snapshot: VideoAssetSnapshot) -> dict[str, object]:
    payload = _summary_payload(snapshot)
    payload["asset"] = (
        None
        if snapshot.asset is None
        else snapshot.asset.model_dump(mode="json", exclude_computed_fields=True)
    )
    return payload


def _sse(snapshot: VideoAssetSnapshot) -> str:
    data = {
        "version": snapshot.version,
        "asset_version": snapshot.asset_version,
        "status": snapshot.status,
        "stage": snapshot.current_stage,
        "error": snapshot.error,
    }
    return f"event: asset_update\ndata: {json.dumps(data)}\n\n"
