import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type PointerEvent,
} from "react";
import type { SoundTrack, TimelineRow } from "../types";

interface TimelineProps {
  rows: TimelineRow[];
  soundTracks: SoundTrack[];
  frame: number;
  frameCount: number;
  onSelectFrame: (frame: number) => void;
  onEditSoundEvent?: (
    soundEventId: string,
    startFrame: number,
    endFrame: number,
  ) => void;
  onCreateSoundTrack?: (input: CreateSoundTrackInput) => void;
  onDeleteSoundTrack?: (soundTrackId: string) => void;
  onCreateSoundEvent?: (soundTrackId: string, startFrame: number) => void;
  onDeleteSoundEvent?: (soundEventId: string) => void;
  onEditSoundEventDetails?: (soundEventId: string) => void;
}

type LaneKind = "scene" | "tracking" | "visual" | "sound";
type TrackType = "sfx" | "ambience" | "dialogue" | "music";
type GenerationMode = "unknown" | "tta" | "vta" | "hybrid";

export interface CreateSoundTrackInput {
  trackType: TrackType;
  label: string;
  generationMode: GenerationMode;
  canonicalKey: string | null;
}

interface PackedTimelineRow extends TimelineRow {
  depth: number;
}

interface LaneGroup {
  lane: string;
  soundTrackId?: string;
  rows: PackedTimelineRow[];
  depthCount: number;
}

interface ActiveDrag {
  soundEventId: string;
  mode: "move" | "resize-left" | "resize-right";
  startX: number;
  laneWidth: number;
  startFrame: number;
  endFrame: number;
}

const laneToggles: { kind: LaneKind; label: string }[] = [
  { kind: "scene", label: "Scenes" },
  { kind: "tracking", label: "Tracking" },
  { kind: "visual", label: "Visual" },
  { kind: "sound", label: "Sound" },
];

export default function Timeline({
  rows,
  soundTracks,
  frame,
  frameCount,
  onSelectFrame,
  onEditSoundEvent,
  onCreateSoundTrack,
  onDeleteSoundTrack,
  onCreateSoundEvent,
  onDeleteSoundEvent,
  onEditSoundEventDetails,
}: TimelineProps) {
  const [enabledKinds, setEnabledKinds] = useState<Record<LaneKind, boolean>>({
    scene: true,
    tracking: true,
    visual: true,
    sound: true,
  });
  const dragRef = useRef<ActiveDrag | null>(null);
  const [showCreateTrack, setShowCreateTrack] = useState(false);
  const [trackType, setTrackType] = useState<TrackType>("sfx");
  const [generationMode, setGenerationMode] = useState<GenerationMode>("vta");
  const [trackLabel, setTrackLabel] = useState("");
  const [canonicalKey, setCanonicalKey] = useState("");
  const visibleRows = rows.filter((row) => enabledKinds[rowKind(row)]);
  const lanes = groupRows(visibleRows, soundTracks);
  const timelineEndFrame = Math.max(frameCount, maxRowEnd(rows), 1);
  const maxPlaybackFrame = Math.max(0, timelineEndFrame - 1);
  const clampedFrame = clamp(frame, 0, maxPlaybackFrame);
  const playheadRatio = clampedFrame / Math.max(maxPlaybackFrame, 1);
  const playheadLeft = `${playheadRatio * 100}%`;

  useEffect(() => {
    function handlePointerMove(event: globalThis.PointerEvent) {
      const drag = dragRef.current;
      if (!drag || !onEditSoundEvent) {
        return;
      }
      const deltaFrames = Math.round(
        ((event.clientX - drag.startX) / Math.max(drag.laneWidth, 1)) *
          timelineEndFrame,
      );
      const length = Math.max(1, drag.endFrame - drag.startFrame);
      let nextStart = drag.startFrame;
      let nextEnd = drag.endFrame;

      if (drag.mode === "resize-left") {
        nextStart = clamp(drag.startFrame + deltaFrames, 0, drag.endFrame - 1);
      } else if (drag.mode === "resize-right") {
        nextEnd = clamp(drag.endFrame + deltaFrames, drag.startFrame + 1, timelineEndFrame);
      } else {
        nextStart = clamp(
          drag.startFrame + deltaFrames,
          0,
          Math.max(0, timelineEndFrame - length),
        );
        nextEnd = nextStart + length;
      }

      onEditSoundEvent(drag.soundEventId, nextStart, nextEnd);
    }

    function handlePointerUp() {
      dragRef.current = null;
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [onEditSoundEvent, timelineEndFrame]);

  function selectFrame(event: PointerEvent<HTMLElement>) {
    if (maxPlaybackFrame <= 0) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    onSelectFrame(Math.round(ratio * maxPlaybackFrame));
  }

  function startSoundEventEdit(
    event: PointerEvent<HTMLDivElement>,
    row: PackedTimelineRow,
  ) {
    if (!row.sound_event_id || !onEditSoundEvent) {
      return;
    }
    const lane = event.currentTarget.closest(".lane-track");
    if (!(lane instanceof HTMLElement)) {
      return;
    }
    const target = event.target instanceof HTMLElement ? event.target : null;
    const handle = target?.dataset.handle;
    dragRef.current = {
      soundEventId: row.sound_event_id,
      mode:
        handle === "left"
          ? "resize-left"
          : handle === "right"
            ? "resize-right"
            : "move",
      startX: event.clientX,
      laneWidth: lane.getBoundingClientRect().width,
      startFrame: row.start_frame,
      endFrame: row.end_frame,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
  }

  function submitCreateTrack(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const label = trackLabel.trim();
    if (!label || !onCreateSoundTrack) {
      return;
    }
    onCreateSoundTrack({
      trackType,
      label,
      generationMode,
      canonicalKey: canonicalKey.trim() || null,
    });
    setTrackLabel("");
    setCanonicalKey("");
    setShowCreateTrack(false);
  }

  function createEventInLane(event: PointerEvent<HTMLElement>, soundTrackId: string) {
    if (!onCreateSoundEvent) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    onCreateSoundEvent(soundTrackId, Math.round(ratio * timelineEndFrame));
    event.preventDefault();
    event.stopPropagation();
  }

  return (
    <section className="timeline-panel">
      <div className="timeline-header">
        <h2>Timeline</h2>
        <div className="timeline-header-actions">
          <span>
            {visibleRows.length}/{rows.length} items
          </span>
          <div className="timeline-toggles" aria-label="Timeline lane filters">
            {laneToggles.map((toggle) => (
              <button
                className={enabledKinds[toggle.kind] ? "active" : ""}
                key={toggle.kind}
                onClick={() =>
                  setEnabledKinds((current) => ({
                    ...current,
                    [toggle.kind]: !current[toggle.kind],
                  }))
                }
                type="button"
              >
                {toggle.label}
              </button>
            ))}
          </div>
          <button
            className={showCreateTrack ? "timeline-add-track active" : "timeline-add-track"}
            disabled={!onCreateSoundTrack}
            onClick={() => setShowCreateTrack((value) => !value)}
            type="button"
          >
            + Sound Track
          </button>
        </div>
      </div>
      {showCreateTrack ? (
        <form className="timeline-create-track" onSubmit={submitCreateTrack}>
          <select
            value={trackType}
            onChange={(event) => setTrackType(event.target.value as TrackType)}
          >
            <option value="sfx">sfx</option>
            <option value="ambience">ambience</option>
            <option value="dialogue">dialogue</option>
            <option value="music">music</option>
          </select>
          <input
            value={trackLabel}
            onChange={(event) => setTrackLabel(event.target.value)}
            placeholder="Track label"
          />
          <select
            value={generationMode}
            onChange={(event) => setGenerationMode(event.target.value as GenerationMode)}
            title="Generation Mode"
          >
            <option value="vta">VTA (Video-to-Audio)</option>
            <option value="tta">TTA (Text-to-Audio)</option>
            <option value="hybrid">Hybrid</option>
            <option value="unknown">Unknown</option>
          </select>
          <input
            value={canonicalKey}
            onChange={(event) => setCanonicalKey(event.target.value)}
            placeholder="canonical_key"
          />
          <button type="submit">Create</button>
        </form>
      ) : null}
      <div className="timeline">
        <div className="timeline-content">
          <div className="timeline-playhead-area">
            <div className="playhead-line" style={{ left: playheadLeft }} />
          </div>
          <div className="timeline-ruler-row">
            <div className="timeline-label-spacer" />
            <div
              className="timeline-ruler"
              onPointerDown={selectFrame}
              onPointerMove={(event) => {
                if (event.buttons === 1) {
                  selectFrame(event);
                }
              }}
            />
          </div>
          {lanes.length === 0 ? (
            <div className="empty-lane">Pipeline lanes will appear here.</div>
          ) : (
            lanes.map(({ lane, soundTrackId, rows: laneRows, depthCount }) => (
              <div className="lane" key={lane}>
                <div className="lane-label">
                  <span>{lane}</span>
                  {soundTrackId ? (
                    <div className="lane-actions">
                      <button
                        title="Add sound event"
                        onClick={() => onCreateSoundEvent?.(soundTrackId, clampedFrame)}
                        type="button"
                      >
                        +
                      </button>
                      <button
                        title="Delete sound track"
                        onClick={() => onDeleteSoundTrack?.(soundTrackId)}
                        type="button"
                      >
                        x
                      </button>
                    </div>
                  ) : null}
                </div>
                <div
                  className="lane-track"
                  onPointerDown={(event) => {
                    if (event.detail === 2 && soundTrackId) {
                      createEventInLane(event, soundTrackId);
                      return;
                    }
                    selectFrame(event);
                  }}
                  onPointerMove={(event) => {
                    if (event.buttons === 1 && !dragRef.current) {
                      selectFrame(event);
                    }
                  }}
                  style={{ minHeight: `${depthCount * 24 + 8}px` }}
                >
                  {laneRows.map((row, index) => {
                    const bar = barStyle(row, timelineEndFrame);
                    const kind = rowKind(row);
                    const editable = isEditableSoundRow(row) && !!onEditSoundEvent;
                    const visualStyle =
                      kind === "visual"
                        ? ({ "--bar-color": visualEventColor(row.kind) } as CSSProperties)
                        : {};
                    const genPrefix = row.generation_mode && row.generation_mode !== 'unknown' ? `[${row.generation_mode.toUpperCase()}] ` : "";
                    return (
                      <div
                        className={`bar bar-${kind} bar-${safeClass(row.kind)}${
                          editable ? " bar-editable" : ""
                        }`}
                        key={
                          row.sound_event_id ??
                          `${lane}-${row.start_frame}-${row.end_frame}-${index}`
                        }
                        onPointerDown={
                          editable
                            ? (event) => startSoundEventEdit(event, row)
                            : undefined
                        }
                        style={{
                          left: `${bar.left}%`,
                          width: `${bar.width}%`,
                          top: `${row.depth * 24 + 4}px`,
                          ...visualStyle,
                        }}
                        title={`${genPrefix}${row.label}: ${row.start_frame}-${row.end_frame}`}
                        onDoubleClick={
                          editable && onEditSoundEventDetails
                            ? (event) => {
                                event.stopPropagation();
                                onEditSoundEventDetails(row.sound_event_id!);
                              }
                            : undefined
                        }
                      >
                        {editable && onDeleteSoundEvent ? (
                          <button
                            className="bar-delete"
                            onClick={(event) => {
                              event.stopPropagation();
                              onDeleteSoundEvent(row.sound_event_id ?? "");
                            }}
                            onPointerDown={(event) => event.stopPropagation()}
                            type="button"
                          >
                            x
                          </button>
                        ) : null}
                        {editable ? (
                          <span
                            aria-hidden="true"
                            className="bar-resize-handle bar-resize-left"
                            data-handle="left"
                          />
                        ) : null}
                        {bar.width < 2 ? "" : bar.width < 7 ? row.kind : row.label}
                        {editable ? (
                          <span
                            aria-hidden="true"
                            className="bar-resize-handle bar-resize-right"
                            data-handle="right"
                          />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}

function groupRows(rows: TimelineRow[], soundTracks: SoundTrack[]): LaneGroup[] {
  const lanes = new Map<string, TimelineRow[]>();
  for (const row of rows) {
    if (!lanes.has(row.lane)) {
      lanes.set(row.lane, []);
    }
    lanes.get(row.lane)?.push(row);
  }
  for (const track of soundTracks) {
    const lane = soundLane(track);
    if (!lanes.has(lane)) {
      lanes.set(lane, []);
    }
  }
  return [...lanes.entries()].map(([lane, laneRows]) => {
    const packedRows = packLaneRows(laneRows);
    const depthCount =
      packedRows.reduce((maxDepth, row) => Math.max(maxDepth, row.depth), 0) + 1;
    return {
      lane,
      soundTrackId: soundTracks.find((track) => soundLane(track) === lane)
        ?.sound_track_id,
      rows: packedRows,
      depthCount,
    };
  });
}

function soundLane(track: SoundTrack): string {
  return `[${track.track_type}] ${track.label}`;
}

function packLaneRows(rows: TimelineRow[]): PackedTimelineRow[] {
  const depthEndFrames: number[] = [];
  return [...rows]
    .sort(
      (left, right) =>
        left.start_frame - right.start_frame || left.end_frame - right.end_frame,
    )
    .map((row) => {
      const depth = depthEndFrames.findIndex((endFrame) => endFrame <= row.start_frame);
      const nextDepth = depth === -1 ? depthEndFrames.length : depth;
      depthEndFrames[nextDepth] = row.end_frame;
      return { ...row, depth: nextDepth };
    });
}

function rowKind(row: TimelineRow): LaneKind {
  if (row.kind === "scene") {
    return "scene";
  }
  if (row.kind === "track") {
    return "tracking";
  }
  if (row.lane.startsWith("visual: ")) {
    return "visual";
  }
  return "sound";
}

function isEditableSoundRow(row: TimelineRow): boolean {
  return rowKind(row) === "sound" && !!row.sound_event_id;
}

function maxRowEnd(rows: TimelineRow[]): number {
  return rows.reduce((maxEnd, row) => Math.max(maxEnd, row.end_frame), 0);
}

function safeClass(value: string): string {
  return value.replace(/[^a-z0-9_-]/gi, "-").toLowerCase();
}

function barStyle(
  row: TimelineRow,
  timelineEndFrame: number,
): { left: number; width: number } {
  const start = clamp(row.start_frame, 0, timelineEndFrame);
  const end = clamp(row.end_frame, start, timelineEndFrame);
  const left = (start / timelineEndFrame) * 100;
  const rawWidth = ((end - start) / timelineEndFrame) * 100;
  const width = Math.min(100 - left, Math.max(0.4, rawWidth));
  return { left, width };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function visualEventColor(kind: string): string {
  const colors: Record<string, string> = {
    contact: "#cf4e4e",
    collision: "#d55f38",
    occlusion: "#8d6fd1",
    motion: "#2f8f72",
    interaction: "#c2872f",
    gesture: "#3f7bd4",
    state_change: "#b64f9f",
  };
  return colors[kind] ?? `hsl(${hashKind(kind)} 68% 42%)`;
}

function hashKind(value: string): number {
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) % 360;
  }
  return hash;
}
