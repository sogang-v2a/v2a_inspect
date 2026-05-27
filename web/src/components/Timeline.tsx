import type { TimelineRow } from "../types";

interface TimelineProps {
  rows: TimelineRow[];
  frame: number;
  frameCount: number;
}

export default function Timeline({ rows, frame, frameCount }: TimelineProps) {
  const lanes = groupRows(rows);
  const maxFrame = Math.max(frameCount, ...rows.map((row) => row.end_frame), 1);
  const playheadLeft = `${(frame / maxFrame) * 100}%`;

  return (
    <section className="timeline-panel">
      <div className="timeline-header">
        <h2>Timeline</h2>
        <span>{rows.length} items</span>
      </div>
      <div className="timeline">
        <div className="playhead-line" style={{ left: playheadLeft }} />
        {lanes.length === 0 ? (
          <div className="empty-lane">Pipeline lanes will appear here.</div>
        ) : (
          lanes.map(([lane, laneRows]) => (
            <div className="lane" key={lane}>
              <div className="lane-label">{lane}</div>
              <div className="lane-track">
                {laneRows.map((row, index) => (
                  <div
                    className={`bar bar-${safeClass(row.kind)}`}
                    key={`${lane}-${row.start_frame}-${row.end_frame}-${index}`}
                    style={{
                      left: `${(row.start_frame / maxFrame) * 100}%`,
                      width: `${Math.max(
                        0.5,
                        ((row.end_frame - row.start_frame) / maxFrame) * 100,
                      )}%`,
                    }}
                    title={`${row.label}: ${row.start_frame}-${row.end_frame}`}
                  >
                    {row.label}
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function groupRows(rows: TimelineRow[]): [string, TimelineRow[]][] {
  const lanes = new Map<string, TimelineRow[]>();
  for (const row of rows) {
    if (!lanes.has(row.lane)) {
      lanes.set(row.lane, []);
    }
    lanes.get(row.lane)?.push(row);
  }
  return [...lanes.entries()];
}

function safeClass(value: string): string {
  return value.replace(/[^a-z0-9_-]/gi, "-").toLowerCase();
}
