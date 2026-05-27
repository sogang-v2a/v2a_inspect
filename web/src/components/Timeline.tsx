import type { PointerEvent } from "react";
import type { TimelineRow } from "../types";

interface TimelineProps {
  rows: TimelineRow[];
  frame: number;
  frameCount: number;
  onSelectFrame: (frame: number) => void;
}

export default function Timeline({
  rows,
  frame,
  frameCount,
  onSelectFrame,
}: TimelineProps) {
  const lanes = groupRows(rows);
  const maxPlaybackFrame = Math.max(0, frameCount - 1);
  const maxFrame = Math.max(maxPlaybackFrame, ...rows.map((row) => row.end_frame), 1);
  const clampedFrame = clamp(frame, 0, maxFrame);
  const playheadRatio = clampedFrame / maxFrame;
  const playheadLeft = `calc(${playheadRatio * 100}% + ${
    150 * (1 - playheadRatio) - 2 * playheadRatio
  }px)`;

  function selectFrame(event: PointerEvent<HTMLElement>) {
    if (maxPlaybackFrame <= 0) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    onSelectFrame(Math.round(ratio * maxPlaybackFrame));
  }

  return (
    <section className="timeline-panel">
      <div className="timeline-header">
        <h2>Timeline</h2>
        <span>{rows.length} items</span>
      </div>
      <div className="timeline">
        <div className="playhead-line" style={{ left: playheadLeft }} />
        <div
          className="timeline-ruler"
          onPointerDown={selectFrame}
          onPointerMove={(event) => {
            if (event.buttons === 1) {
              selectFrame(event);
            }
          }}
        />
        {lanes.length === 0 ? (
          <div className="empty-lane">Pipeline lanes will appear here.</div>
        ) : (
          lanes.map(([lane, laneRows]) => (
            <div className="lane" key={lane}>
              <div className="lane-label">{lane}</div>
              <div
                className="lane-track"
                onPointerDown={selectFrame}
                onPointerMove={(event) => {
                  if (event.buttons === 1) {
                    selectFrame(event);
                  }
                }}
              >
                {laneRows.map((row, index) => {
                  const bar = barStyle(row, maxFrame);
                  return (
                    <div
                      className={`bar bar-${safeClass(row.kind)}`}
                      key={`${lane}-${row.start_frame}-${row.end_frame}-${index}`}
                      style={{
                        left: `${bar.left}%`,
                        width: `${bar.width}%`,
                      }}
                      title={`${row.label}: ${row.start_frame}-${row.end_frame}`}
                    >
                      {row.label}
                    </div>
                  );
                })}
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

function barStyle(row: TimelineRow, maxFrame: number): { left: number; width: number } {
  const start = clamp(row.start_frame, 0, maxFrame);
  const end = clamp(row.end_frame, start, maxFrame);
  const left = (start / maxFrame) * 100;
  const rawWidth = ((end - start) / maxFrame) * 100;
  const width = Math.min(100 - left, Math.max(0.4, rawWidth));
  return { left, width };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
