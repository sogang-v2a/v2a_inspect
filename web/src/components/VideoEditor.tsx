import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Inspector from "./Inspector";
import Timeline from "./Timeline";
import TrackingOverlay from "./TrackingOverlay";
import type { AssetResponse } from "../types";

interface VideoEditorProps {
  state: AssetResponse;
  submitError: string | null;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

export default function VideoEditor({ state, submitError, onSubmit }: VideoEditorProps) {
  const [frame, setFrame] = useState(0);
  const [showTrackingOverlay, setShowTrackingOverlay] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const asset = state.asset;
  const fps = 30;
  const maxFrame = Math.max(0, (asset?.frame_count ?? 1) - 1);
  const selectedFrame = clamp(frame, 0, maxFrame);
  const timeSec = selectedFrame / fps;
  const counts = useMemo(() => {
    const scenes = asset?.initial_scenes.length ?? 0;
    const tracks =
      asset?.initial_scenes.reduce(
        (total, scene) => total + scene.scene_tracks.length,
        0,
      ) ?? 0;
    const visualEvents = asset?.visual_identity_layer?.visual_events.length ?? 0;
    const soundTracks = asset?.sound_timeline?.sound_tracks.length ?? 0;
    const soundEvents = asset?.sound_timeline?.sound_events.length ?? 0;
    return { scenes, tracks, visualEvents, soundTracks, soundEvents };
  }, [asset]);
  const selectFrame = useCallback(
    (nextFrame: number) => {
      setFrame(clamp(nextFrame, 0, maxFrame));
    },
    [maxFrame],
  );

  useEffect(() => {
    if (frame !== selectedFrame) {
      setFrame(selectedFrame);
    }
  }, [frame, selectedFrame]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !asset) {
      return;
    }
    if (Math.abs(video.currentTime - timeSec) > 0.04) {
      video.currentTime = timeSec;
    }
  }, [asset, timeSec]);

  function syncFrameFromVideo() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    selectFrame(Math.round(video.currentTime * fps));
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
          {submitError ? <p className="error">{submitError}</p> : null}
          {state.error ? <p className="error">{state.error}</p> : null}
          <dl className="asset-stats">
            <div>
              <dt>Version</dt>
              <dd>{state.version}</dd>
            </div>
            <div>
              <dt>Frames</dt>
              <dd>{asset?.frame_count ?? "-"}</dd>
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
                  disabled={!asset}
                  onClick={togglePlayback}
                  type="button"
                >
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <button
                  className={showTrackingOverlay ? "toggle active" : "toggle"}
                  disabled={!asset}
                  onClick={() => setShowTrackingOverlay((value) => !value)}
                  type="button"
                >
                  Tracking
                </button>
              </div>
              {asset ? (
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
                    asset={asset}
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
                  disabled={!asset}
                />
              </label>
            </div>
            <Inspector
              asset={asset}
              frame={selectedFrame}
              assetVersion={state.asset_version}
            />
          </div>
          <Timeline
            rows={state.timeline_rows}
            frame={selectedFrame}
            frameCount={asset?.frame_count ?? 0}
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
