import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Inspector from "./Inspector";
import MaskOverlay from "./MaskOverlay";
import Timeline from "./Timeline";
import TrackingOverlay from "./TrackingOverlay";
import { fetchTrackingWindow } from "../api";
import type { AssetResponse, TrackWindowResponse } from "../types";

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
  const [showSegmentMasks, setShowSegmentMasks] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [trackWindow, setTrackWindow] = useState<TrackWindowResponse | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const trackWindowRequestRef = useRef<string | null>(null);
  const video = state.video;
  const fps = video?.fps || 30;
  const maxFrame = Math.max(0, (video?.frame_count ?? 1) - 1);
  const selectedFrame = clamp(frame, 0, maxFrame);
  const timeSec = selectedFrame / fps;
  const counts = useMemo(() => {
    const scenes = countRows(state.timeline_rows, "scene");
    const tracks = countRows(state.timeline_rows, "track");
    const visualEvents = state.timeline_rows.filter((row) =>
      row.lane.startsWith("visual: "),
    ).length;
    const soundEvents = state.timeline_rows.filter(isSoundRow).length;
    const soundTracks = new Set(
      state.timeline_rows.filter(isSoundRow).map((row) => row.lane),
    ).size;
    return { scenes, tracks, visualEvents, soundTracks, soundEvents };
  }, [state.timeline_rows]);
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
              video ? (
                <a
                  className="download-link"
                  href={`/api/asset/export?asset_version=${state.asset_version}`}
                  download="video-asset.json"
                >
                  Download VideoAsset JSON
                </a>
              ) : (
                <button className="download-link" disabled type="button">
                  Download VideoAsset JSON
                </button>
              )
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
                <button
                  className={showSegmentMasks ? "toggle active" : "toggle"}
                  disabled={!video}
                  onClick={() => setShowSegmentMasks((value) => !value)}
                  type="button"
                >
                  Masks
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
                  <MaskOverlay
                    assetVersion={state.asset_version}
                    enabled={showSegmentMasks}
                    frame={selectedFrame}
                    video={video}
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
              timelineRows={state.timeline_rows}
              version={state.version}
            />
          </div>
          <Timeline
            rows={state.timeline_rows}
            frame={selectedFrame}
            frameCount={video?.frame_count ?? 0}
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
