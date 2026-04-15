from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat
from pydantic import BaseModel, Field

from v2a_inspect.tools.types import FrameBatch


class RegionProposal(BaseModel):
    scene_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(default_factory=list, min_length=4, max_length=4)
    crop_path: str | None = None


def propose_moving_regions(
    frame_batches: Sequence[FrameBatch],
    *,
    output_root: str,
    threshold: float = 0.08,
) -> dict[int, list[RegionProposal]]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    proposals_by_scene: dict[int, list[RegionProposal]] = {}
    for batch in frame_batches:
        proposals: list[RegionProposal] = []
        frames = list(batch.frames)
        for index, (left, right) in enumerate(zip(frames, frames[1:], strict=False)):
            proposal = _proposal_from_frame_pair(
                scene_index=batch.scene_index,
                frame_index=index + 1,
                left_path=left.image_path,
                right_path=right.image_path,
                output_root=root,
                threshold=threshold,
            )
            if proposal is not None:
                proposals.append(proposal)
        proposals_by_scene[batch.scene_index] = proposals
    return proposals_by_scene


def _proposal_from_frame_pair(
    *,
    scene_index: int,
    frame_index: int,
    left_path: str,
    right_path: str,
    output_root: Path,
    threshold: float,
) -> RegionProposal | None:
    with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
        left_gray = ImageOps.grayscale(left_image.convert("RGB")).resize((128, 128))
        right_gray = ImageOps.grayscale(right_image.convert("RGB")).resize((128, 128))
        diff = ImageChops.difference(left_gray, right_gray)
        score = float(ImageStat.Stat(diff).mean[0] / 255.0)
        if score < threshold:
            return None
        bbox = diff.getbbox()
        if bbox is None:
            return None
        scale_x = max(left_image.width / 128.0, 1.0)
        scale_y = max(left_image.height / 128.0, 1.0)
        bbox_xyxy = [
            round(bbox[0] * scale_x, 3),
            round(bbox[1] * scale_y, 3),
            round(bbox[2] * scale_x, 3),
            round(bbox[3] * scale_y, 3),
        ]
        crop_path = output_root / f"scene_{scene_index:04d}_motion_{frame_index:02d}.jpg"
        cropped = right_image.crop(
            (
                max(int(bbox_xyxy[0]), 0),
                max(int(bbox_xyxy[1]), 0),
                max(int(bbox_xyxy[2]), 1),
                max(int(bbox_xyxy[3]), 1),
            )
        )
        cropped.save(crop_path, format="JPEG")
    return RegionProposal(
        scene_index=scene_index,
        frame_index=frame_index,
        motion_score=round(min(max(score, 0.0), 1.0), 4),
        bbox_xyxy=bbox_xyxy,
        crop_path=str(crop_path),
    )


def _image_block(image_path: str) -> dict[str, object]:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }
