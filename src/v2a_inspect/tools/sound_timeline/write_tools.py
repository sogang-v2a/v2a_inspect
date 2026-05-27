from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from v2a_inspect.models import SoundEvent, SoundSource, SoundTrack

from .schemas import (
    DeleteSoundEventOutput,
    DeleteSoundSourceOutput,
    DeleteSoundTrackOutput,
    SoundGenerationMode,
    SoundSourceType,
    SoundTrackType,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor


class SoundTimelineWriteTools:
    def __init__(self, editor: SoundTimelineEditor) -> None:
        self.editor = editor

    def upsert_sound_source(
        self,
        source_type: SoundSourceType,
        label: str,
        sound_source_id: UUID | None = None,
        visual_object_id: UUID | None = None,
        notes: str | None = None,
    ) -> SoundSource:
        timeline = self.editor.ensure_sound_timeline()
        source = SoundSource(
            sound_source_id=sound_source_id or uuid4(),
            source_type=source_type,
            label=label,
            visual_object_id=visual_object_id,
            notes=notes,
        )
        sources = list(timeline.sound_sources)
        if sound_source_id is None:
            sources.append(source)
        else:
            for index, existing in enumerate(sources):
                if existing.sound_source_id == sound_source_id:
                    sources[index] = source
                    break
            else:
                raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_sources": sources}
        )
        return source

    def delete_sound_source(self, sound_source_id: UUID) -> DeleteSoundSourceOutput:
        timeline = self.editor.ensure_sound_timeline()
        sources = [
            source
            for source in timeline.sound_sources
            if source.sound_source_id != sound_source_id
        ]
        if len(sources) == len(timeline.sound_sources):
            raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
        tracks = [
            track.model_copy(update={"sound_source_id": None})
            if track.sound_source_id == sound_source_id
            else track
            for track in timeline.sound_tracks
        ]
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_sources": sources, "sound_tracks": tracks}
        )
        return DeleteSoundSourceOutput(deleted_sound_source_id=sound_source_id)

    def upsert_sound_track(
        self,
        track_type: SoundTrackType,
        label: str,
        sound_track_id: UUID | None = None,
        sound_source_id: UUID | None = None,
        generation_mode: SoundGenerationMode = "unknown",
        notes: str | None = None,
    ) -> SoundTrack:
        timeline = self.editor.ensure_sound_timeline()
        if sound_source_id is not None and not any(
            source.sound_source_id == sound_source_id
            for source in timeline.sound_sources
        ):
            raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
        track = SoundTrack(
            sound_track_id=sound_track_id or uuid4(),
            track_type=track_type,
            label=label,
            sound_source_id=sound_source_id,
            generation_mode=generation_mode,
            notes=notes,
        )
        tracks = list(timeline.sound_tracks)
        if sound_track_id is None:
            tracks.append(track)
        else:
            for index, existing in enumerate(tracks):
                if existing.sound_track_id == sound_track_id:
                    tracks[index] = track
                    break
            else:
                raise ValueError(f"Unknown sound_track_id: {sound_track_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_tracks": tracks}
        )
        return track

    def delete_sound_track(self, sound_track_id: UUID) -> DeleteSoundTrackOutput:
        timeline = self.editor.ensure_sound_timeline()
        if any(event.sound_track_id == sound_track_id for event in timeline.sound_events):
            raise ValueError(
                f"Cannot delete sound_track_id with existing events: {sound_track_id}"
            )
        tracks = [
            track
            for track in timeline.sound_tracks
            if track.sound_track_id != sound_track_id
        ]
        if len(tracks) == len(timeline.sound_tracks):
            raise ValueError(f"Unknown sound_track_id: {sound_track_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_tracks": tracks}
        )
        return DeleteSoundTrackOutput(deleted_sound_track_id=sound_track_id)

    def upsert_sound_event(
        self,
        start_frame_index: int,
        end_frame_index: int,
        description: str,
        sound_track_id: UUID,
        sound_event_id: UUID | None = None,
        notes: str | None = None,
    ) -> SoundEvent:
        self.editor.check_frame_range(start_frame_index, end_frame_index)
        timeline = self.editor.ensure_sound_timeline()
        if not any(
            track.sound_track_id == sound_track_id for track in timeline.sound_tracks
        ):
            raise ValueError(f"Unknown sound_track_id: {sound_track_id}")
        event = SoundEvent(
            sound_event_id=sound_event_id or uuid4(),
            sound_track_id=sound_track_id,
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            description=description,
            notes=notes,
        )
        events = list(timeline.sound_events)
        if sound_event_id is None:
            events.append(event)
        else:
            for index, existing in enumerate(events):
                if existing.sound_event_id == sound_event_id:
                    events[index] = event
                    break
            else:
                raise ValueError(f"Unknown sound_event_id: {sound_event_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_events": events}
        )
        return event

    def delete_sound_event(self, sound_event_id: UUID) -> DeleteSoundEventOutput:
        timeline = self.editor.ensure_sound_timeline()
        events = [
            event
            for event in timeline.sound_events
            if event.sound_event_id != sound_event_id
        ]
        if len(events) == len(timeline.sound_events):
            raise ValueError(f"Unknown sound_event_id: {sound_event_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_events": events}
        )
        return DeleteSoundEventOutput(deleted_sound_event_id=sound_event_id)
