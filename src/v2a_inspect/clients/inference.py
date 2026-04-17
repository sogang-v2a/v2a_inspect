from __future__ import annotations

import json
import mimetypes
import uuid
from io import BytesIO
from pathlib import Path
from urllib import request

from v2a_inspect.remote_inference_payloads import (
    EmbedImagesManifest,
    LabelScoreManifest,
    Sam3ExtractManifest,
    TrackImageBatch,
    UploadedFrameBatch,
    UploadedFrameRef,
    UploadedImageRef,
)
from v2a_inspect.tools.types import (
    EntityEmbedding,
    FrameBatch,
    LabelScore,
    Sam3RegionSeed,
    Sam3TrackSet,
)


CLIENT_USER_AGENT = "v2a-inspect-inference-client/1.0"


def server_runtime_info(server_base_url: str, *, timeout_seconds: float = 30.0) -> dict[str, object]:
    with request.urlopen(
        f"{server_base_url.rstrip('/')}/runtime-info",
        timeout=timeout_seconds,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("runtime-info response must be a JSON object")
    return payload


def warmup_server(server_base_url: str, *, timeout_seconds: float = 600.0) -> dict[str, object]:
    request_obj = request.Request(
        url=f"{server_base_url.rstrip('/')}/warmup",
        headers={"Content-Type": "application/json", "User-Agent": CLIENT_USER_AGENT},
        data=b"{}",
        method="POST",
    )
    with request.urlopen(request_obj, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("warmup response must be a JSON object")
    return payload


class RemoteSam3Client:
    def __init__(self, *, server_base_url: str, timeout_seconds: float = 120.0) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def extract_entities(
        self,
        frame_batches: list[FrameBatch],
        *,
        prompts_by_scene: dict[int, list[str]] | None = None,
        region_seeds_by_scene: dict[int, list[Sam3RegionSeed]] | None = None,
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
    ) -> Sam3TrackSet:
        files: dict[str, Path] = {}
        manifest_batches: list[UploadedFrameBatch] = []
        for batch in frame_batches:
            manifest_frames: list[UploadedFrameRef] = []
            for frame_index, frame in enumerate(batch.frames):
                upload_key = f"frame_{batch.scene_index}_{frame_index}_{uuid.uuid4().hex[:8]}"
                files[upload_key] = Path(frame.image_path)
                manifest_frames.append(
                    UploadedFrameRef(
                        upload_key=upload_key,
                        filename=Path(frame.image_path).name,
                        scene_index=batch.scene_index,
                        timestamp_seconds=frame.timestamp_seconds,
                    )
                )
            manifest_batches.append(
                UploadedFrameBatch(scene_index=batch.scene_index, frames=manifest_frames)
            )
        manifest = Sam3ExtractManifest(
            frame_batches=manifest_batches,
            prompts_by_scene=prompts_by_scene,
            region_seeds_by_scene=region_seeds_by_scene,
            score_threshold=score_threshold,
            min_points=min_points,
            high_confidence_threshold=high_confidence_threshold,
            match_threshold=match_threshold,
        )
        payload = _post_multipart_json(
            url=f"{self.server_base_url}/infer/sam3-extract",
            manifest=manifest.model_dump(mode="json"),
            files=files,
            timeout_seconds=self.timeout_seconds,
        )
        return Sam3TrackSet.model_validate(payload)


class RemoteEmbeddingClient:
    def __init__(self, *, server_base_url: str, timeout_seconds: float = 120.0) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed_images(self, image_paths_by_track: dict[str, list[str]]) -> list[EntityEmbedding]:
        files: dict[str, Path] = {}
        tracks: list[TrackImageBatch] = []
        for track_id, image_paths in image_paths_by_track.items():
            images: list[UploadedImageRef] = []
            for image_index, image_path in enumerate(image_paths):
                upload_key = f"embed_{track_id}_{image_index}_{uuid.uuid4().hex[:8]}"
                files[upload_key] = Path(image_path)
                images.append(UploadedImageRef(upload_key=upload_key, filename=Path(image_path).name))
            tracks.append(TrackImageBatch(track_id=track_id, images=images))
        manifest = EmbedImagesManifest(tracks=tracks)
        payload = _post_multipart_json(
            url=f"{self.server_base_url}/infer/embed-crops",
            manifest=manifest.model_dump(mode="json"),
            files=files,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(payload, list):
            raise TypeError("embed-crops response must be a list")
        return [EntityEmbedding.model_validate(item) for item in payload]


class RemoteLabelClient:
    def __init__(self, *, server_base_url: str, timeout_seconds: float = 120.0) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def score_image_labels(
        self,
        *,
        image_paths: list[str],
        labels: list[str],
    ) -> list[LabelScore]:
        files: dict[str, Path] = {}
        images: list[UploadedImageRef] = []
        for image_index, image_path in enumerate(image_paths):
            upload_key = f"label_{image_index}_{uuid.uuid4().hex[:8]}"
            files[upload_key] = Path(image_path)
            images.append(UploadedImageRef(upload_key=upload_key, filename=Path(image_path).name))
        manifest = LabelScoreManifest(labels=labels, images=images)
        payload = _post_multipart_json(
            url=f"{self.server_base_url}/infer/score-labels",
            manifest=manifest.model_dump(mode="json"),
            files=files,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(payload, list):
            raise TypeError("score-labels response must be a list")
        return [LabelScore.model_validate(item) for item in payload]


def _post_multipart_json(
    *,
    url: str,
    manifest: dict[str, object],
    files: dict[str, Path],
    timeout_seconds: float,
) -> object:
    boundary = f"----v2ainspect{uuid.uuid4().hex}"
    body = BytesIO()
    _write_field(body, boundary, "manifest", json.dumps(manifest).encode("utf-8"), "application/json")
    for field_name, file_path in files.items():
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        _write_file(body, boundary, field_name, file_path.name, file_path.read_bytes(), mime_type)
    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    payload = body.getvalue()
    request_obj = request.Request(
        url=url,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
            "Accept": "application/json",
            "User-Agent": CLIENT_USER_AGENT,
        },
        data=payload,
        method="POST",
    )
    with request.urlopen(request_obj, timeout=timeout_seconds) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    return decoded


def _write_field(buffer: BytesIO, boundary: str, name: str, payload: bytes, content_type: str) -> None:
    buffer.write(f"--{boundary}\r\n".encode("utf-8"))
    buffer.write(f'Content-Disposition: form-data; name="{name}"\r\n'.encode("utf-8"))
    buffer.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    buffer.write(payload)
    buffer.write(b"\r\n")


def _write_file(
    buffer: BytesIO,
    boundary: str,
    field_name: str,
    filename: str,
    payload: bytes,
    content_type: str,
) -> None:
    buffer.write(f"--{boundary}\r\n".encode("utf-8"))
    buffer.write(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    buffer.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    buffer.write(payload)
    buffer.write(b"\r\n")
