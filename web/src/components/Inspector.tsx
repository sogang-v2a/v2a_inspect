import { useEffect, useState } from "react";
import { fetchCurrentFrame } from "../api";
import type { CurrentFrameRows, VideoAsset } from "../types";

interface InspectorProps {
  asset: VideoAsset | null;
  frame: number;
  version: number;
}

export default function Inspector({ asset, frame, version }: InspectorProps) {
  const [rows, setRows] = useState<CurrentFrameRows | null>(null);

  useEffect(() => {
    if (!asset) {
      setRows(null);
      return;
    }
    void fetchCurrentFrame(frame).then(setRows);
  }, [asset, frame, version]);

  return (
    <aside className="inspector">
      <h2>At Frame</h2>
      {asset ? (
        <>
          <Section title="Scene" value={rows?.scene ?? null} />
          <Section title="Tracks" value={rows?.tracks ?? []} />
          <Section title="Visual / Contact Events" value={rows?.visual_events ?? []} />
          <Section title="Sound Events" value={rows?.sound_events ?? []} />
        </>
      ) : (
        <p className="muted">No asset loaded.</p>
      )}
    </aside>
  );
}

function Section({ title, value }: { title: string; value: unknown }) {
  return (
    <section className="inspector-section">
      <h3>{title}</h3>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}
