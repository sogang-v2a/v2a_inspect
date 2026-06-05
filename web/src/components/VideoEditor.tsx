import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Inspector from "./Inspector";
import Timeline, { type CreateSoundTrackInput } from "./Timeline";
import TrackingOverlay from "./TrackingOverlay";
import { fetchAssetForExport, fetchTrackingWindow } from "../api";
import type {
  AssetResponse,
  SoundEvent,
  SoundTrack,
  TimelineRow,
  TrackWindowResponse,
  VideoAsset,
} from "../types";

interface VideoEditorProps {
  state: AssetResponse;
  submitError: string | null;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onImport: (event: FormEvent<HTMLFormElement>) => void;
  onResetSoundTimeline: () => void;
}

export default function VideoEditor({
  state,
  submitError,
  onSubmit,
  onImport,
  onResetSoundTimeline,
}: VideoEditorProps) {
  const [frame, setFrame] = useState(0);
  const [showTrackingOverlay, setShowTrackingOverlay] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [trackWindow, setTrackWindow] = useState<TrackWindowResponse | null>(null);
  const [timelineRows, setTimelineRows] = useState<TimelineRow[]>(state.timeline_rows);
  const [baseAsset, setBaseAsset] = useState<VideoAsset | null>(null);
  const [draftAsset, setDraftAsset] = useState<VideoAsset | null>(null);
  const [hasTimelineEdits, setHasTimelineEdits] = useState(false);
  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const trackWindowRequestRef = useRef<string | null>(null);
  const video = state.video;
  const fps = video?.fps || 30;
  const timelineFrameCount = Math.max(video?.frame_count ?? 0, inferFrameCount(timelineRows));
  const maxFrame = Math.max(0, timelineFrameCount - 1);
  const selectedFrame = clamp(frame, 0, maxFrame);
  const timeSec = selectedFrame / fps;
  const soundTracks = useMemo(
    () => draftAsset?.sound_timeline?.sound_tracks ?? [],
    [draftAsset],
  );
  const counts = useMemo(() => {
    const scenes = countRows(timelineRows, "scene");
    const tracks = countRows(timelineRows, "track");
    const visualEvents = timelineRows.filter((row) =>
      row.lane.startsWith("visual: "),
    ).length;
    const soundEvents = timelineRows.filter(isSoundRow).length;
    const soundTracks = new Set(
      timelineRows.filter(isSoundRow).map((row) => row.lane),
    ).size;
    return { scenes, tracks, visualEvents, soundTracks, soundEvents };
  }, [timelineRows]);
  const selectFrame = useCallback(
    (nextFrame: number) => {
      const next = clamp(nextFrame, 0, maxFrame);
      setFrame(next);
      const videoElement = videoRef.current;
      if (videoElement) {
        videoElement.currentTime = next / fps;
      }
    },
    [fps, maxFrame],
  );

  useEffect(() => {
    if (hasTimelineEdits) {
      return;
    }
    setTimelineRows(state.timeline_rows);
    setExportStatus(null);
    let alive = true;
    void fetchAssetForExport()
      .then((asset) => {
        if (!alive) {
          return;
        }
        setBaseAsset(asset);
        setDraftAsset(asset ? cloneAsset(asset) : null);
      })
      .catch(() => {
        if (alive) {
          setBaseAsset(null);
          setDraftAsset(null);
        }
      });
    return () => {
      alive = false;
    };
  }, [state.timeline_rows, state.version]);

  useEffect(() => {
    if (frame !== selectedFrame) {
      setFrame(selectedFrame);
    }
  }, [frame, selectedFrame]);

  useEffect(() => {
    if (!showTrackingOverlay || !video) {
      return;
    }
    if (
      trackWindow &&
      trackWindow.version === state.version &&
      selectedFrame >= trackWindow.start_frame + 60 &&
      selectedFrame <= trackWindow.end_frame - 60
    ) {
      return;
    }
    const startFrame = Math.max(0, selectedFrame - 300);
    const endFrame = Math.min(maxFrame, selectedFrame + 300);
    const requestKey = `${state.version}:${startFrame}:${endFrame}`;
    if (trackWindowRequestRef.current === requestKey) {
      return;
    }
    trackWindowRequestRef.current = requestKey;
    void fetchTrackingWindow(startFrame, endFrame)
      .then(setTrackWindow)
      .finally(() => {
        trackWindowRequestRef.current = null;
      });
  }, [maxFrame, selectedFrame, showTrackingOverlay, state.version, trackWindow, video]);

  useEffect(() => {
    if (!isPlaying) {
      if (animationRef.current !== null) {
        window.cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }
    const sync = () => {
      const videoElement = videoRef.current;
      if (videoElement) {
        setFrame(clamp(Math.round(videoElement.currentTime * fps), 0, maxFrame));
      }
      animationRef.current = window.requestAnimationFrame(sync);
    };
    animationRef.current = window.requestAnimationFrame(sync);
    return () => {
      if (animationRef.current !== null) {
        window.cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [fps, isPlaying, maxFrame]);

  function syncFrameFromVideo() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    setFrame(clamp(Math.round(video.currentTime * fps), 0, maxFrame));
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    if (video.paused) {
      void video.play();
      return;
    }
    video.pause();
  }

  const editSoundEventTimestamp = useCallback(
    (soundEventId: string, startFrame: number, endFrame: number) => {
      const nextStart = clamp(startFrame, 0, Math.max(0, maxFrame));
      const nextEnd = clamp(endFrame, nextStart + 1, timelineFrameCount || 1);
      setTimelineRows((currentRows) =>
        currentRows.map((row) => {
          if (row.sound_event_id !== soundEventId) {
            return row;
          }
          return {
            ...row,
            start_frame: nextStart,
            end_frame: nextEnd,
          };
        }),
      );
      setDraftAsset((currentAsset) =>
        currentAsset
          ? updateSoundEventInAsset(currentAsset, soundEventId, (event) => ({
              ...event,
              start_frame_index: nextStart,
              end_frame_index: nextEnd,
            }))
          : currentAsset,
      );
      setHasTimelineEdits(true);
      setExportStatus("Timestamp edits are ready to export.");
    },
    [maxFrame, timelineFrameCount],
  );

  function resetTimestampEdits() {
    setTimelineRows(state.timeline_rows);
    setDraftAsset(baseAsset ? cloneAsset(baseAsset) : null);
    setHasTimelineEdits(false);
    setExportStatus("Local timeline edits reset.");
  }

  function createSoundTrack(input: CreateSoundTrackInput) {
    const track: SoundTrack = {
      sound_track_id: makeId(),
      track_type: input.trackType,
      label: input.label,
      canonical_key: input.canonicalKey ?? normalizeCanonicalKey(input.label),
      sound_source_id: null,
      generation_mode: input.generationMode,
      notes: null,
    };
    setDraftAsset((currentAsset) => {
      if (!currentAsset) {
        return currentAsset;
      }
      const nextAsset = ensureTimeline(cloneAsset(currentAsset));
      nextAsset.sound_timeline?.sound_tracks.push(track);
      return nextAsset;
    });
    setHasTimelineEdits(true);
    setExportStatus(`Created sound track "${track.label}".`);
  }

  function deleteSoundTrack(soundTrackId: string) {
    const track = soundTracks.find((item) => item.sound_track_id === soundTrackId);
    const eventCount =
      draftAsset?.sound_timeline?.sound_events.filter(
        (event) => event.sound_track_id === soundTrackId,
      ).length ?? 0;
    if (
      !window.confirm(
        `Delete "${track?.label ?? "sound track"}" and ${eventCount} events?`,
      )
    ) {
      return;
    }
    setDraftAsset((currentAsset) => {
      if (!currentAsset?.sound_timeline) {
        return currentAsset;
      }
      const nextAsset = cloneAsset(currentAsset);
      const timeline = nextAsset.sound_timeline;
      if (!timeline) {
        return nextAsset;
      }
      timeline.sound_tracks = timeline.sound_tracks.filter(
        (item) => item.sound_track_id !== soundTrackId,
      );
      timeline.sound_events = timeline.sound_events.filter(
        (event) => event.sound_track_id !== soundTrackId,
      );
      return nextAsset;
    });
    setTimelineRows((currentRows) =>
      currentRows.filter((row) => row.sound_track_id !== soundTrackId),
    );
    setHasTimelineEdits(true);
    setExportStatus(`Deleted sound track "${track?.label ?? soundTrackId}".`);
  }

  function createSoundEvent(soundTrackId: string, startFrame: number) {
    const track = soundTracks.find((item) => item.sound_track_id === soundTrackId);
    if (!track) {
      return;
    }
    const nextStart = clamp(startFrame, 0, Math.max(0, maxFrame));
    const nextEnd = clamp(nextStart + 10, nextStart + 1, timelineFrameCount || 1);
    const event: SoundEvent = {
      sound_event_id: makeId(),
      sound_track_id: soundTrackId,
      start_frame_index: nextStart,
      end_frame_index: nextEnd,
      description: `New ${track.label} event`,
      notes: null,
    };
    setDraftAsset((currentAsset) => {
      if (!currentAsset) {
        return currentAsset;
      }
      const nextAsset = ensureTimeline(cloneAsset(currentAsset));
      nextAsset.sound_timeline?.sound_events.push(event);
      return nextAsset;
    });
    setTimelineRows((currentRows) => [
      ...currentRows,
      timelineRowFromSoundEvent(event, track),
    ]);
    setHasTimelineEdits(true);
    setExportStatus(`Created sound event on "${track.label}".`);
  }

  function deleteSoundEvent(soundEventId: string) {
    setDraftAsset((currentAsset) => {
      if (!currentAsset?.sound_timeline) {
        return currentAsset;
      }
      const nextAsset = cloneAsset(currentAsset);
      const timeline = nextAsset.sound_timeline;
      if (!timeline) {
        return nextAsset;
      }
      timeline.sound_events = timeline.sound_events.filter(
        (event) => event.sound_event_id !== soundEventId,
      );
      return nextAsset;
    });
    setTimelineRows((currentRows) =>
      currentRows.filter((row) => row.sound_event_id !== soundEventId),
    );
    setHasTimelineEdits(true);
    setExportStatus("Deleted sound event.");
  }

  async function exportEditedAsset() {
    try {
      const asset = draftAsset ?? (await fetchAssetForExport());
      if (!asset) {
        setExportStatus("No VideoAsset JSON is available to export.");
        return;
      }
      const editedAsset = applyTimelineEdits(asset, timelineRows);
      downloadJson(editedAsset, "video-asset-edited.json");
      setHasTimelineEdits(false);
      setExportStatus("Exported video-asset-edited.json.");
    } catch (error) {
      setExportStatus(
        error instanceof Error
          ? `Export failed: ${error.message}`
          : `Export failed: ${String(error)}`,
      );
    }
  }

  return (
    <main className="editor-shell">
      <header className="topbar">
        <div>
          <h1>V2A Inspect</h1>
          <p>Live VideoAsset editor</p>
        </div>
        <div className={`status status-${state.status}`}>
          <strong>{state.status}</strong>
          <span>{state.stage ?? "no stage"}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="controls">
          <form onSubmit={onSubmit}>
            <label>
              Video
              <input name="video" type="file" accept="video/*" required />
            </label>
            <label>
              Work directory
              <input
                name="work_dir"
                type="text"
                placeholder="/tmp/v2a-inspect-ui"
              />
            </label>
            <label>
              Inference server URL
              <input name="server_url" type="url" placeholder="http://..." />
            </label>
            <div className="split-inputs">
              <label>
                Scene threshold
                <input
                  name="scene_threshold"
                  type="number"
                  min="1"
                  max="100"
                  step="1"
                  defaultValue="27"
                />
              </label>
              <label>
                Keyframes
                <input
                  name="max_keyframes_per_scene"
                  type="number"
                  min="1"
                  max="100"
                  step="1"
                  defaultValue="20"
                />
              </label>
            </div>
            <button type="submit" disabled={state.status === "running"}>
              Run pipeline
            </button>
          </form>
          <form className="import-form" onSubmit={onImport}>
            <label>
              Video
              <input name="video" type="file" accept="video/*" required />
            </label>
            <label>
              VideoAsset JSON
              <input
                name="asset"
                type="file"
                accept="application/json,.json"
                required
              />
            </label>
            <label>
              Work directory
              <input
                name="work_dir"
                type="text"
                placeholder="/tmp/v2a-inspect-ui"
              />
            </label>
            <button type="submit" disabled={state.status === "running"}>
              Import asset
            </button>
          </form>
          {submitError ? <p className="error">{submitError}</p> : null}
          {state.error ? <p className="error">{state.error}</p> : null}
          <section className="export-panel">
            <button
              className="toggle"
              disabled={!video || state.status === "running"}
              onClick={onResetSoundTimeline}
              type="button"
            >
              Reset + rerun soundtrack
            </button>
            <button
              className={showExport ? "toggle active" : "toggle"}
              onClick={() => setShowExport((value) => !value)}
              type="button"
            >
              Export
            </button>
            {showExport ? (
              <>
                <button
                  className="download-link"
                  disabled={!draftAsset}
                  onClick={exportEditedAsset}
                  type="button"
                >
                  Download edited VideoAsset JSON
                </button>
                <button
                  className="download-link secondary"
                  disabled={!hasTimelineEdits}
                  onClick={resetTimestampEdits}
                  type="button"
                >
                  Reset timestamp edits
                </button>
                {exportStatus ? <p className="muted">{exportStatus}</p> : null}
              </>
            ) : null}
          </section>
          <dl className="asset-stats">
            <div>
              <dt>Version</dt>
              <dd>{state.version}</dd>
            </div>
            <div>
              <dt>Frames</dt>
              <dd>{video?.frame_count ?? "-"}</dd>
            </div>
            <div>
              <dt>Scenes</dt>
              <dd>{counts.scenes}</dd>
            </div>
            <div>
              <dt>Tracks</dt>
              <dd>{counts.tracks}</dd>
            </div>
            <div>
              <dt>Visual events</dt>
              <dd>{counts.visualEvents}</dd>
            </div>
            <div>
              <dt>Sound</dt>
              <dd>
                {counts.soundTracks}/{counts.soundEvents}
              </dd>
            </div>
          </dl>
        </aside>

        <section className="editor">
          <div className="preview-grid">
            <div className="video-panel">
              <div className="preview-toolbar">
                <button
                  className="toggle"
                  disabled={!video}
                  onClick={togglePlayback}
                  type="button"
                >
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <button
                  className={showTrackingOverlay ? "toggle active" : "toggle"}
                  disabled={!video}
                  onClick={() => setShowTrackingOverlay((value) => !value)}
                  type="button"
                >
                  Tracking
                </button>
              </div>
              {video ? (
                <div className="video-frame">
                  <video
                    ref={videoRef}
                    src={`/api/video?asset_version=${state.asset_version}`}
                    onPause={() => setIsPlaying(false)}
                    onPlay={() => setIsPlaying(true)}
                    onSeeked={syncFrameFromVideo}
                    onTimeUpdate={syncFrameFromVideo}
                  />
                  <TrackingOverlay
                    video={video}
                    tracks={trackWindow?.tracks ?? []}
                    enabled={showTrackingOverlay}
                    frame={selectedFrame}
                  />
                </div>
              ) : (
                <div className="empty-video">Upload a video to start</div>
              )}
              <label className="playhead">
                <span>
                  Frame {selectedFrame} / {maxFrame} ({timeSec.toFixed(2)}s)
                </span>
                <input
                  type="range"
                  min="0"
                  max={maxFrame}
                  step="1"
                  value={selectedFrame}
                  onChange={(event) => selectFrame(Number(event.target.value))}
                  disabled={!video}
                />
              </label>
            </div>
            <Inspector
              video={video}
              frame={selectedFrame}
              timelineRows={timelineRows}
              version={state.version}
            />
          </div>
          <Timeline
            rows={timelineRows}
            soundTracks={soundTracks}
            frame={selectedFrame}
            frameCount={timelineFrameCount}
            onCreateSoundTrack={createSoundTrack}
            onDeleteSoundTrack={deleteSoundTrack}
            onCreateSoundEvent={createSoundEvent}
            onDeleteSoundEvent={deleteSoundEvent}
            onEditSoundEvent={editSoundEventTimestamp}
            onSelectFrame={selectFrame}
          />
        </section>
      </section>
    </main>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function countRows(rows: AssetResponse["timeline_rows"], kind: string): number {
  return rows.filter((row) => row.kind === kind).length;
}

function isSoundRow(row: AssetResponse["timeline_rows"][number]): boolean {
  return (
    row.kind !== "scene" &&
    row.kind !== "track" &&
    !row.lane.startsWith("visual: ")
  );
}

function inferFrameCount(rows: TimelineRow[]): number {
  return rows.reduce((maxEnd, row) => Math.max(maxEnd, row.end_frame), 0);
}

function applyTimelineEdits(asset: VideoAsset, rows: TimelineRow[]): VideoAsset {
  const editedAsset = cloneAsset(asset);
  const editedRows = new Map(
    rows
      .filter((row) => row.sound_event_id)
      .map((row) => [row.sound_event_id, row]),
  );
  const timeline = editedAsset.sound_timeline;
  if (!timeline) {
    return editedAsset;
  }
  timeline.sound_events = timeline.sound_events.map((event) => {
    const row = editedRows.get(event.sound_event_id);
    if (!row) {
      return event;
    }
    return {
      ...event,
      start_frame_index: row.start_frame,
      end_frame_index: row.end_frame,
    };
  });
  return editedAsset;
}

function ensureTimeline(asset: VideoAsset): VideoAsset {
  if (asset.sound_timeline) {
    return asset;
  }
  asset.sound_timeline = {
    sound_sources: [],
    sound_tracks: [],
    sound_events: [],
    notes: null,
  };
  return asset;
}

function updateSoundEventInAsset(
  asset: VideoAsset,
  soundEventId: string,
  update: (event: SoundEvent) => SoundEvent,
): VideoAsset {
  const nextAsset = cloneAsset(asset);
  const timeline = nextAsset.sound_timeline;
  if (!timeline) {
    return nextAsset;
  }
  timeline.sound_events = timeline.sound_events.map((event) =>
    event.sound_event_id === soundEventId ? update(event) : event,
  );
  return nextAsset;
}

function timelineRowFromSoundEvent(
  event: SoundEvent,
  track: SoundTrack,
): TimelineRow {
  return {
    lane: soundLane(track),
    label: event.description,
    start_frame: event.start_frame_index,
    end_frame: event.end_frame_index,
    kind: track.track_type,
    sound_event_id: event.sound_event_id,
    sound_track_id: event.sound_track_id,
  };
}

function soundLane(track: SoundTrack): string {
  return `[${track.track_type}] ${track.label}`;
}

function normalizeCanonicalKey(value: string): string | null {
  const normalized = value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return normalized || null;
}

function makeId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function cloneAsset(asset: VideoAsset): VideoAsset {
  return typeof structuredClone === "function"
    ? structuredClone(asset)
    : JSON.parse(JSON.stringify(asset));
}

function downloadJson(value: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
