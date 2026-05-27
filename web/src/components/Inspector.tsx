import { useEffect, useState } from "react";
import { fetchCurrentFrame } from "../api";
import type {
  CurrentFrameRows,
  SceneFrameRow,
  SoundEventFrameRow,
  TrackFrameRow,
  VideoAsset,
  VisualEventFrameRow,
} from "../types";

type InspectorTab = "scene" | "tracks" | "visual" | "sound";

interface InspectorProps {
  asset: VideoAsset | null;
  frame: number;
  assetVersion: number;
}

const tabs: { id: InspectorTab; label: string }[] = [
  { id: "scene", label: "Scene" },
  { id: "tracks", label: "Tracks" },
  { id: "visual", label: "Visual" },
  { id: "sound", label: "Sound" },
];

export default function Inspector({ asset, frame, assetVersion }: InspectorProps) {
  const [rows, setRows] = useState<CurrentFrameRows | null>(null);
  const [activeTab, setActiveTab] = useState<InspectorTab>("scene");

  useEffect(() => {
    if (!asset) {
      setRows(null);
      return;
    }
    void fetchCurrentFrame(frame).then(setRows);
  }, [asset, frame, assetVersion]);

  return (
    <aside className="inspector">
      <div className="inspector-header">
        <h2>At Frame</h2>
        <span>{frame}</span>
      </div>
      {asset ? (
        <>
          <div className="inspector-tabs" role="tablist">
            {tabs.map((tab) => (
              <button
                className={activeTab === tab.id ? "active" : ""}
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                role="tab"
                type="button"
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="inspector-body">
            {activeTab === "scene" ? <SceneDetails row={rows?.scene ?? null} /> : null}
            {activeTab === "tracks" ? (
              <TrackDetails rows={rows?.tracks ?? []} />
            ) : null}
            {activeTab === "visual" ? (
              <VisualEventDetails rows={rows?.visual_events ?? []} />
            ) : null}
            {activeTab === "sound" ? (
              <SoundEventDetails rows={rows?.sound_events ?? []} />
            ) : null}
          </div>
        </>
      ) : (
        <p className="muted">No asset loaded.</p>
      )}
    </aside>
  );
}

function SceneDetails({ row }: { row: SceneFrameRow | null }) {
  if (!row) {
    return <p className="empty-detail">No active scene.</p>;
  }
  return (
    <dl className="detail-grid">
      <Field label="Scene" value={row.scene} />
      <Field label="Frames" value={`${row.start_frame}-${row.end_frame}`} />
      <Field label="Duration" value={`${row.duration_sec}s`} />
    </dl>
  );
}

function TrackDetails({ rows }: { rows: TrackFrameRow[] }) {
  if (rows.length === 0) {
    return <p className="empty-detail">No active tracks.</p>;
  }
  return (
    <div className="detail-list">
      {rows.map((row) => (
        <article className="detail-card" key={`${row.scene}-${row.track}`}>
          <div className="detail-card-title">
            <strong>{row.label}</strong>
            <span>{row.confidence.toFixed(3)}</span>
          </div>
          <dl className="detail-grid">
            <Field label="Scene" value={row.scene} />
            <Field label="Track" value={row.track} />
            <Field label="Bbox" value={formatBbox(row.bbox)} wide />
            <Field label="Mask" value={row.has_mask ? "yes" : "no"} />
          </dl>
        </article>
      ))}
    </div>
  );
}

function VisualEventDetails({ rows }: { rows: VisualEventFrameRow[] }) {
  if (rows.length === 0) {
    return <p className="empty-detail">No visual events at this frame.</p>;
  }
  return (
    <div className="detail-list">
      {rows.map((row, index) => (
        <article
          className="detail-card"
          key={`${row.event_type}-${row.start_frame}-${index}`}
        >
          <div className="detail-card-title">
            <strong>{row.event_type}</strong>
            <span>{row.confidence.toFixed(3)}</span>
          </div>
          <dl className="detail-grid">
            <Field label="Object" value={row.object} />
            <Field label="Related" value={row.related || "-"} />
            <Field label="Frames" value={`${row.start_frame}-${row.end_frame}`} />
            <Field label="Duration" value={`${row.duration_sec}s`} />
            <Field label="Description" value={row.description} wide />
            <Field label="Notes" value={row.notes || "-"} wide />
          </dl>
        </article>
      ))}
    </div>
  );
}

function SoundEventDetails({ rows }: { rows: SoundEventFrameRow[] }) {
  if (rows.length === 0) {
    return <p className="empty-detail">No sound events at this frame.</p>;
  }
  return (
    <div className="detail-list">
      {rows.map((row, index) => (
        <article
          className="detail-card"
          key={`${row.track_type}-${row.start_frame}-${index}`}
        >
          <div className="detail-card-title">
            <strong>{row.track_type}</strong>
            <span>{row.generation_mode}</span>
          </div>
          <dl className="detail-grid">
            <Field label="Source" value={row.source || "-"} />
            <Field label="Frames" value={`${row.start_frame}-${row.end_frame}`} />
            <Field label="Duration" value={`${row.duration_sec}s`} />
            <Field label="Description" value={row.description} wide />
            <Field label="Notes" value={row.notes || "-"} wide />
          </dl>
        </article>
      ))}
    </div>
  );
}

function Field({
  label,
  value,
  wide = false,
}: {
  label: string;
  value: string | number;
  wide?: boolean;
}) {
  return (
    <div className={wide ? "detail-field wide" : "detail-field"}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatBbox(bbox: number[] | null): string {
  if (!bbox) {
    return "-";
  }
  return bbox.map((value) => value.toFixed(1)).join(", ");
}
