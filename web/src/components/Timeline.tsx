import { useState, type PointerEvent } from "react";
import type { TimelineRow } from "../types";

interface TimelineProps {
  rows: TimelineRow[];
  frame: number;
  frameCount: number;
  onSelectFrame: (frame: number) => void;
}

type LaneKind = "scene" | "tracking" | "visual" | "sound";

interface PackedTimelineRow extends TimelineRow {
  depth: number;
}

interface LaneGroup {
  lane: string;
  rows: PackedTimelineRow[];
  depthCount: number;
}

const laneToggles: { kind: LaneKind; label: string }[] = [
  { kind: "scene", label: "Scenes" },
  { kind: "tracking", label: "Tracking" },
  { kind: "visual", label: "Visual" },
  { kind: "sound", label: "Sound" },
];

export default function Timeline({
  rows,
  frame,
  frameCount,
  onSelectFrame,
}: TimelineProps) {
  const [enabledKinds, setEnabledKinds] = useState<Record<LaneKind, boolean>>({
    scene: true,
    tracking: true,
    visual: true,
    sound: true,
  });
  const visibleRows = rows.filter((row) => enabledKinds[rowKind(row)]);
  const lanes = groupRows(visibleRows);
  const maxPlaybackFrame = Math.max(0, frameCount - 1);
  const maxFrame = Math.max(maxPlaybackFrame, 1);
  const clampedFrame = clamp(frame, 0, maxFrame);
  const playheadRatio = clampedFrame / maxFrame;
  const playheadLeft = `${playheadRatio * 100}%`;

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
        </div>
      </div>
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
            lanes.map(({ lane, rows: laneRows, depthCount }) => (
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
                  style={{ minHeight: `${depthCount * 24 + 8}px` }}
                >
                  {laneRows.map((row, index) => {
                    const bar = barStyle(row, maxFrame);
                    const kind = rowKind(row);
                    return (
                      <div
                        className={`bar bar-${kind} bar-${safeClass(row.kind)}`}
                        key={`${lane}-${row.start_frame}-${row.end_frame}-${index}`}
                        style={{
                          left: `${bar.left}%`,
                          width: `${bar.width}%`,
                          top: `${row.depth * 24 + 4}px`,
                          ...(kind === "visual"
                            ? { "--bar-color": visualEventColor(row.kind) }
                            : {}),
                        }}
                        title={`${row.label}: ${row.start_frame}-${row.end_frame}`}
                      >
                        {bar.width < 2 ? "" : bar.width < 7 ? row.kind : row.label}
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

function groupRows(rows: TimelineRow[]): LaneGroup[] {
  const lanes = new Map<string, TimelineRow[]>();
  for (const row of rows) {
    if (!lanes.has(row.lane)) {
      lanes.set(row.lane, []);
    }
    lanes.get(row.lane)?.push(row);
  }
  return [...lanes.entries()].map(([lane, laneRows]) => {
    const packedRows = packLaneRows(laneRows);
    const depthCount =
      packedRows.reduce((maxDepth, row) => Math.max(maxDepth, row.depth), 0) + 1;
    return { lane, rows: packedRows, depthCount };
  });
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
