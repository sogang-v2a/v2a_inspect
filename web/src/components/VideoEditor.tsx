import { FormEvent, useMemo, useState } from "react";
import Inspector from "./Inspector";
import Timeline from "./Timeline";
import type { AssetResponse } from "../types";

interface VideoEditorProps {
  state: AssetResponse;
  submitError: string | null;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

export default function VideoEditor({ state, submitError, onSubmit }: VideoEditorProps) {
  const [frame, setFrame] = useState(0);
  const asset = state.asset;
  const maxFrame = Math.max(0, (asset?.frame_count ?? 1) - 1);
  const selectedFrame = Math.min(frame, maxFrame);
  const timeSec = selectedFrame / 30;
  const counts = useMemo(() => {
    const scenes = asset?.initial_scenes.length ?? 0;
    const tracks =
      asset?.initial_scenes.reduce(
        (total, scene) => total + scene.scene_tracks.length,
        0,
      ) ?? 0;
    const visualEvents = asset?.visual_identity_layer?.visual_events.length ?? 0;
    const soundEvents = asset?.sound_timeline?.sound_events.length ?? 0;
    return { scenes, tracks, visualEvents, soundEvents };
  }, [asset]);

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
              <dt>Sound events</dt>
              <dd>{counts.soundEvents}</dd>
            </div>
          </dl>
        </aside>

        <section className="editor">
          <div className="preview-grid">
            <div className="video-panel">
              {asset ? (
                <video src="/api/video" controls />
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
                  onChange={(event) => setFrame(Number(event.target.value))}
                  disabled={!asset}
                />
              </label>
            </div>
            <Inspector asset={asset} frame={selectedFrame} version={state.version} />
          </div>
          <Timeline
            rows={state.timeline_rows}
            frame={selectedFrame}
            frameCount={asset?.frame_count ?? 0}
          />
        </section>
      </section>
    </main>
  );
}
