import { useEffect, useRef, useState } from "react";
import type { VideoSummary } from "../types";

interface MaskOverlayProps {
  assetVersion: number;
  enabled: boolean;
  frame: number;
  video: VideoSummary;
}

interface MaskAtlas {
  endFrame: number;
  image: HTMLImageElement;
  startFrame: number;
  tileHeight: number;
  tileWidth: number;
  url: string;
}

const ATLAS_FRAMES = 30;
const SOURCE_WIDTH = 1280;
const SOURCE_HEIGHT = 720;

export default function MaskOverlay({
  assetVersion,
  enabled,
  frame,
  video,
}: MaskOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const atlasRef = useRef<MaskAtlas | null>(null);
  const nextAtlasRef = useRef<MaskAtlas | null>(null);
  const requestKeyRef = useRef<string | null>(null);
  const [atlas, setAtlas] = useState<MaskAtlas | null>(null);
  const [nextAtlas, setNextAtlas] = useState<MaskAtlas | null>(null);

  useEffect(() => {
    atlasRef.current = atlas;
  }, [atlas]);

  useEffect(() => {
    nextAtlasRef.current = nextAtlas;
  }, [nextAtlas]);

  useEffect(() => {
    if (!enabled) {
      abortRef.current?.abort();
      abortRef.current = null;
      requestKeyRef.current = null;
      setAtlas(disposeAtlas);
      setNextAtlas(disposeAtlas);
    }
  }, [enabled]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      disposeAtlas(atlasRef.current);
      disposeAtlas(nextAtlasRef.current);
    };
  }, []);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    if (atlas && containsFrame(atlas, frame)) {
      const midpoint =
        atlas.startFrame + Math.floor((atlas.endFrame - atlas.startFrame) / 2);
      if (
        frame >= midpoint &&
        (!nextAtlas || nextAtlas.startFrame !== atlas.endFrame + 1)
      ) {
        void loadAtlas(atlas.endFrame + 1, video.frame_count - 1);
      }
      return;
    }
    if (nextAtlas && containsFrame(nextAtlas, frame)) {
      setAtlas((previous) => replaceAtlas(previous, nextAtlas));
      setNextAtlas(null);
      return;
    }
    void loadAtlas(frame, video.frame_count - 1);
  }, [atlas, enabled, frame, nextAtlas, video.frame_count]);

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

    if (!enabled || !atlas || !containsFrame(atlas, frame)) {
      return;
    }

    const sourceWidth = video.width ?? SOURCE_WIDTH;
    const sourceHeight = video.height ?? SOURCE_HEIGHT;
    const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
    const offsetX = (rect.width - sourceWidth * scale) / 2;
    const offsetY = (rect.height - sourceHeight * scale) / 2;
    const tileIndex = frame - atlas.startFrame;
    context.drawImage(
      atlas.image,
      0,
      tileIndex * atlas.tileHeight,
      atlas.tileWidth,
      atlas.tileHeight,
      offsetX,
      offsetY,
      sourceWidth * scale,
      sourceHeight * scale,
    );
  }, [atlas, enabled, frame, video.height, video.width]);

  async function loadAtlas(startFrame: number, maxFrame: number): Promise<void> {
    if (startFrame > maxFrame) {
      return;
    }
    const start = Math.max(0, Math.min(startFrame, maxFrame));
    const end = Math.min(maxFrame, start + ATLAS_FRAMES - 1);
    const requestKey = `${assetVersion}:${start}:${end}`;
    if (requestKeyRef.current === requestKey || start > end) {
      return;
    }

    requestKeyRef.current = requestKey;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const params = new URLSearchParams({
      start_frame: String(start),
      end_frame: String(end),
      asset_version: String(assetVersion),
    });

    try {
      const response = await fetch(
        `/api/frames/tracking-mask-atlas?${params.toString()}`,
        { signal: controller.signal },
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch mask atlas: ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const image = new Image();
      image.src = url;
      await image.decode();
      const loadedAtlas = {
        endFrame: Number(response.headers.get("X-End-Frame") ?? end),
        image,
        startFrame: Number(response.headers.get("X-Start-Frame") ?? start),
        tileHeight: Number(response.headers.get("X-Tile-Height") ?? 360),
        tileWidth: Number(response.headers.get("X-Tile-Width") ?? 640),
        url,
      };
      if (loadedAtlas.startFrame > frame) {
        setNextAtlas((previous) => replaceAtlas(previous, loadedAtlas));
      } else {
        setAtlas((previous) => replaceAtlas(previous, loadedAtlas));
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      console.error(error);
    } finally {
      if (requestKeyRef.current === requestKey) {
        requestKeyRef.current = null;
      }
    }
  }

  return (
    <canvas
      aria-hidden="true"
      className="segmentation-mask-overlay"
      ref={canvasRef}
    />
  );
}

function containsFrame(atlas: MaskAtlas, frame: number): boolean {
  return frame >= atlas.startFrame && frame <= atlas.endFrame;
}

function replaceAtlas(previous: MaskAtlas | null, next: MaskAtlas): MaskAtlas {
  if (previous && previous.url !== next.url) {
    URL.revokeObjectURL(previous.url);
  }
  return next;
}

function disposeAtlas(atlas: MaskAtlas | null): null {
  if (atlas) {
    URL.revokeObjectURL(atlas.url);
  }
  return null;
}
