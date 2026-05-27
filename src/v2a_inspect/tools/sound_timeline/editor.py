from __future__ import annotations

from uuid import UUID

from langchain_core.tools import StructuredTool

from v2a_inspect.models import (
    SoundEvent,
    SoundSource,
    SoundTimeline,
    SoundTrack,
    VideoAsset,
)

from .langchain import build_sound_timeline_tools
from .read_tools import SoundTimelineReadTools
from .schemas import (
    AnnotatedFrameOutput,
    DeleteSoundEventOutput,
    DeleteSoundSourceOutput,
    DeleteSoundTrackOutput,
    FrameResolutionMode,
    ListScenesOutput,
    ListTracksOutput,
    SceneSummaryOutput,
    SoundGenerationMode,
    SoundSourceType,
    SoundTimelineViewOutput,
    SoundTrackType,
    VisualEventsOutput,
)
from .write_tools import SoundTimelineWriteTools


class SoundTimelineEditor:
    """Tool-backed editor for one mutable VideoAsset.sound_timeline."""

    def __init__(self, video_asset: VideoAsset) -> None:
        self.video_asset = video_asset
        self.read = SoundTimelineReadTools(self)
        self.write = SoundTimelineWriteTools(self)

    def list_scenes(
        self,
        start_scene_index: int = 0,
        limit: int = 25,
    ) -> ListScenesOutput:
        return self.read.list_scenes(start_scene_index, limit)

    def get_scene_summary(self, scene_index: int) -> SceneSummaryOutput:
        return self.read.get_scene_summary(scene_index)

    def get_annotated_frame(
        self,
        frame_index: int,
        resolution_mode: FrameResolutionMode = "low",
    ) -> AnnotatedFrameOutput:
        return self.read.get_annotated_frame(frame_index, resolution_mode)

    def list_tracks(self, scene_index: int) -> ListTracksOutput:
        return self.read.list_tracks(scene_index)

    def get_visual_events(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 50,
    ) -> VisualEventsOutput:
        return self.read.get_visual_events(
            start_frame_index,
            end_frame_index,
            limit,
        )

    def get_sound_timeline(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 50,
    ) -> SoundTimelineViewOutput:
        return self.read.get_sound_timeline(
            start_frame_index,
            end_frame_index,
            limit,
        )

    def upsert_sound_source(
        self,
        source_type: SoundSourceType,
        label: str,
        sound_source_id: UUID | None = None,
        visual_object_id: UUID | None = None,
        notes: str | None = None,
    ) -> SoundSource:
        return self.write.upsert_sound_source(
            source_type=source_type,
            label=label,
            sound_source_id=sound_source_id,
            visual_object_id=visual_object_id,
            notes=notes,
        )

    def delete_sound_source(self, sound_source_id: UUID) -> DeleteSoundSourceOutput:
        return self.write.delete_sound_source(sound_source_id)

    def upsert_sound_track(
        self,
        track_type: SoundTrackType,
        label: str,
        sound_track_id: UUID | None = None,
        sound_source_id: UUID | None = None,
        generation_mode: SoundGenerationMode = "unknown",
        notes: str | None = None,
    ) -> SoundTrack:
        return self.write.upsert_sound_track(
            track_type=track_type,
            label=label,
            sound_track_id=sound_track_id,
            sound_source_id=sound_source_id,
            generation_mode=generation_mode,
            notes=notes,
        )

    def delete_sound_track(self, sound_track_id: UUID) -> DeleteSoundTrackOutput:
        return self.write.delete_sound_track(sound_track_id)

    def upsert_sound_event(
        self,
        start_frame_index: int,
        end_frame_index: int,
        description: str,
        sound_track_id: UUID,
        sound_event_id: UUID | None = None,
        notes: str | None = None,
    ) -> SoundEvent:
        return self.write.upsert_sound_event(
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            description=description,
            sound_track_id=sound_track_id,
            sound_event_id=sound_event_id,
            notes=notes,
        )

    def delete_sound_event(self, sound_event_id: UUID) -> DeleteSoundEventOutput:
        return self.write.delete_sound_event(sound_event_id)

    def tools(self) -> list[StructuredTool]:
        return build_sound_timeline_tools(self)

    def ensure_sound_timeline(self) -> SoundTimeline:
        if self.video_asset.sound_timeline is None:
            self.video_asset.sound_timeline = SoundTimeline()
        return self.video_asset.sound_timeline

    def check_frame_index(self, frame_index: int, *, allow_end: bool) -> None:
        upper = (
            self.video_asset.frame_count
            if allow_end
            else self.video_asset.frame_count - 1
        )
        if frame_index < 0 or frame_index > upper:
            raise ValueError(f"frame_index out of range: {frame_index}")

    def check_frame_range(self, start_frame_index: int, end_frame_index: int) -> None:
        self.check_frame_index(start_frame_index, allow_end=False)
        self.check_frame_index(end_frame_index, allow_end=True)
        if end_frame_index <= start_frame_index:
            raise ValueError("end_frame_index must be greater than start_frame_index")
