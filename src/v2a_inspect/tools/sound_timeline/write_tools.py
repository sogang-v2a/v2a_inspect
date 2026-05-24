from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

from v2a_inspect.models import SoundEvent, SoundSource

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor


class SoundTimelineWriteTools:
    def __init__(self, editor: SoundTimelineEditor) -> None:
        self.editor = editor

    def upsert_sound_source(
        self,
        source_type: str,
        label: str,
        sound_source_id: str | None = None,
        visual_object_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        timeline = self.editor.ensure_sound_timeline()
        source_uuid = UUID(sound_source_id) if sound_source_id is not None else None
        visual_uuid = UUID(visual_object_id) if visual_object_id is not None else None
        source = SoundSource(
            sound_source_id=source_uuid or uuid4(),
            source_type=cast(Any, source_type),
            label=label,
            visual_object_id=visual_uuid,
            notes=notes,
        )
        sources = list(timeline.sound_sources)
        if source_uuid is None:
            sources.append(source)
        else:
            for index, existing in enumerate(sources):
                if existing.sound_source_id == source_uuid:
                    sources[index] = source
                    break
            else:
                raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_sources": sources}
        )
        return source.model_dump(mode="json")

    def delete_sound_source(self, sound_source_id: str) -> dict[str, Any]:
        timeline = self.editor.ensure_sound_timeline()
        source_uuid = UUID(sound_source_id)
        sources = [
            source
            for source in timeline.sound_sources
            if source.sound_source_id != source_uuid
        ]
        if len(sources) == len(timeline.sound_sources):
            raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
        events = [
            event.model_copy(update={"sound_source_id": None})
            if event.sound_source_id == source_uuid
            else event
            for event in timeline.sound_events
        ]
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_sources": sources, "sound_events": events}
        )
        return {"deleted_sound_source_id": str(source_uuid)}

    def upsert_sound_event(
        self,
        start_frame_index: int,
        end_frame_index: int,
        description: str,
        track_type: str,
        sound_event_id: str | None = None,
        sound_source_id: str | None = None,
        generation_mode: str = "unknown",
        notes: str | None = None,
    ) -> dict[str, Any]:
        self.editor.check_frame_range(start_frame_index, end_frame_index)
        timeline = self.editor.ensure_sound_timeline()
        event_uuid = UUID(sound_event_id) if sound_event_id is not None else None
        source_uuid = UUID(sound_source_id) if sound_source_id is not None else None
        if source_uuid is not None and not any(
            source.sound_source_id == source_uuid for source in timeline.sound_sources
        ):
            raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
        event = SoundEvent(
            sound_event_id=event_uuid or uuid4(),
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            description=description,
            track_type=cast(Any, track_type),
            sound_source_id=source_uuid,
            generation_mode=cast(Any, generation_mode),
            notes=notes,
        )
        events = list(timeline.sound_events)
        if event_uuid is None:
            events.append(event)
        else:
            for index, existing in enumerate(events):
                if existing.sound_event_id == event_uuid:
                    events[index] = event
                    break
            else:
                raise ValueError(f"Unknown sound_event_id: {sound_event_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_events": events}
        )
        return event.model_dump(mode="json")

    def delete_sound_event(self, sound_event_id: str) -> dict[str, Any]:
        timeline = self.editor.ensure_sound_timeline()
        event_uuid = UUID(sound_event_id)
        events = [
            event
            for event in timeline.sound_events
            if event.sound_event_id != event_uuid
        ]
        if len(events) == len(timeline.sound_events):
            raise ValueError(f"Unknown sound_event_id: {sound_event_id}")
        self.editor.video_asset.sound_timeline = timeline.model_copy(
            update={"sound_events": events}
        )
        return {"deleted_sound_event_id": str(event_uuid)}
