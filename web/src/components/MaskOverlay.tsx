import { useEffect, useMemo, useRef } from "react";
import type { MaskWindowResponse, MaskWindowTrack, VideoSummary } from "../types";

interface MaskOverlayProps {
  enabled: boolean;
  frame: number;
  masks: MaskWindowResponse | null;
  video: VideoSummary;
}

const SOURCE_WIDTH = 1280;
const SOURCE_HEIGHT = 720;

export default function MaskOverlay({
  enabled,
  frame,
  masks,
  video,
}: MaskOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointIndex = useMemo(() => indexMaskPoints(masks), [masks]);

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

    if (!enabled || !masks) {
      return;
    }

    const activePoints = pointIndex.get(frame) ?? [];
    if (!activePoints.length) {
      return;
    }

    const sourceWidth = video.width ?? SOURCE_WIDTH;
    const sourceHeight = video.height ?? SOURCE_HEIGHT;
    const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
    const offsetX = (rect.width - sourceWidth * scale) / 2;
    const offsetY = (rect.height - sourceHeight * scale) / 2;

    context.globalAlpha = 0.38;
    activePoints.forEach(({ track, point }, index) => {
      const maskWidth = track.width || masks.width;
      const maskHeight = track.height || masks.height;
      const scaleX = (sourceWidth * scale) / maskWidth;
      const scaleY = (sourceHeight * scale) / maskHeight;
      context.fillStyle = colorForIndex(index);
      for (const [y, x1, x2] of point.spans) {
        context.fillRect(
          offsetX + x1 * scaleX,
          offsetY + y * scaleY,
          Math.max(1, (x2 - x1) * scaleX),
          Math.max(1, scaleY),
        );
      }
    });
    context.globalAlpha = 1;
  }, [enabled, frame, masks, pointIndex, video.height, video.width]);

  return (
    <canvas
      aria-hidden="true"
      className="segmentation-mask-overlay"
      ref={canvasRef}
    />
  );
}

function indexMaskPoints(
  masks: MaskWindowResponse | null,
): Map<
  number,
  { track: MaskWindowTrack; point: MaskWindowTrack["points"][number] }[]
> {
  const index = new Map<
    number,
    { track: MaskWindowTrack; point: MaskWindowTrack["points"][number] }[]
  >();
  if (!masks) {
    return index;
  }
  for (const track of masks.tracks) {
    for (const point of track.points) {
      const framePoints = index.get(point.frame_index) ?? [];
      framePoints.push({ track, point });
      index.set(point.frame_index, framePoints);
    }
  }
  return index;
}

function colorForIndex(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue} 72% 62%)`;
}
