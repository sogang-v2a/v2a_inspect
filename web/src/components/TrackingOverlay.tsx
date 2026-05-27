import { useEffect, useRef } from "react";
import type { SceneTrack, SceneTrackPoint, VideoAsset } from "../types";

interface TrackingOverlayProps {
  asset: VideoAsset;
  frame: number;
  enabled: boolean;
}

const SOURCE_WIDTH = 1280;
const SOURCE_HEIGHT = 720;

export default function TrackingOverlay({
  asset,
  frame,
  enabled,
}: TrackingOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);

    if (!enabled) {
      return;
    }

    const sourceWidth = asset.width ?? SOURCE_WIDTH;
    const sourceHeight = asset.height ?? SOURCE_HEIGHT;
    const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
    const offsetX = (rect.width - sourceWidth * scale) / 2;
    const offsetY = (rect.height - sourceHeight * scale) / 2;
    const tracks = activeTrackPoints(asset, frame);

    context.lineWidth = 2;
    context.font = "12px Inter, system-ui, sans-serif";
    context.textBaseline = "top";

    tracks.forEach(({ track, point }, index) => {
      if (!point.bbox_xyxy) {
        return;
      }
      const color = colorForIndex(index);
      const [x1, y1, x2, y2] = point.bbox_xyxy;
      const left = offsetX + x1 * scale;
      const top = offsetY + y1 * scale;
      const width = Math.max(1, (x2 - x1) * scale);
      const height = Math.max(1, (y2 - y1) * scale);

      context.strokeStyle = color;
      context.strokeRect(left, top, width, height);
      drawLabel(
        context,
        trackLabel(track, index),
        point.confidence,
        left,
        Math.max(0, top - 18),
        color,
      );
    });
  }, [asset, enabled, frame]);

  return <canvas aria-hidden="true" className="tracking-overlay" ref={canvasRef} />;
}

function activeTrackPoints(
  asset: VideoAsset,
  frame: number,
): { track: SceneTrack; point: SceneTrackPoint }[] {
  const points: { track: SceneTrack; point: SceneTrackPoint }[] = [];
  for (const scene of asset.initial_scenes) {
    if (frame < scene.start_frame_index || frame >= scene.end_frame_index) {
      continue;
    }
    for (const track of scene.scene_tracks) {
      const point = track.points.find((item) => item.frame_index === frame);
      if (point) {
        points.push({ track, point });
      }
    }
  }
  return points;
}

function trackLabel(track: SceneTrack, index: number): string {
  return track.source_object_seed?.label || track.tracking_prompt || `track ${index + 1}`;
}

function drawLabel(
  context: CanvasRenderingContext2D,
  label: string,
  confidence: number,
  x: number,
  y: number,
  color: string,
) {
  const text = `${label} ${confidence.toFixed(2)}`;
  const metrics = context.measureText(text);
  const width = metrics.width + 8;
  const height = 16;

  context.fillStyle = "rgba(5, 6, 9, 0.86)";
  context.fillRect(x, y, width, height);
  context.fillStyle = color;
  context.fillText(text, x + 4, y + 2);
}

function colorForIndex(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue} 72% 62%)`;
}
