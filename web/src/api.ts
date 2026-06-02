import { mockAssetSummary, mockCurrentFrameRows, mockFullAsset } from "./mockAsset";
import type {
  AssetResponse,
  CurrentFrameRows,
  FullAssetResponse,
  TrackWindowResponse,
  VideoAsset,
} from "./types";

export async function fetchAssetSummary(): Promise<AssetResponse> {
  try {
    const response = await fetch("/api/asset-summary");
    if (!response.ok) {
      throw new Error(`Failed to fetch asset summary: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    if (import.meta.env.DEV) {
      return mockAssetSummary;
    }
    throw error;
  }
}

export async function fetchAssetForExport(): Promise<VideoAsset | null> {
  try {
    const response = await fetch("/api/asset");
    if (!response.ok) {
      throw new Error(`Failed to fetch asset: ${response.status}`);
    }
    const payload = (await response.json()) as FullAssetResponse;
    return payload.asset;
  } catch (error) {
    if (import.meta.env.DEV) {
      return cloneAsset(mockFullAsset.asset);
    }
    throw error;
  }
}

export async function fetchCurrentFrame(
  frame: number,
  signal?: AbortSignal,
): Promise<CurrentFrameRows | null> {
  try {
    const response = await fetch(`/api/rows/current-frame?frame=${frame}`, { signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch frame rows: ${response.status}`);
    }
    const payload = await response.json();
    return payload.rows;
  } catch (error) {
    if (signal?.aborted) {
      throw error;
    }
    if (import.meta.env.DEV) {
      return mockCurrentFrameRows(frame);
    }
    throw error;
  }
}

export async function fetchTrackingWindow(
  startFrame: number,
  endFrame: number,
): Promise<TrackWindowResponse> {
  const params = new URLSearchParams({
    start_frame: String(startFrame),
    end_frame: String(endFrame),
  });
  try {
    const response = await fetch(`/api/tracks/window?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch tracking window: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    if (import.meta.env.DEV) {
      return {
        version: mockAssetSummary.version,
        start_frame: startFrame,
        end_frame: endFrame,
        tracks: [],
      };
    }
    throw error;
  }
}

export async function startRun(form: HTMLFormElement): Promise<void> {
  const formData = new FormData(form);
  const response = await fetch("/api/runs", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to start run: ${response.status}`);
  }
}

export async function importAsset(form: HTMLFormElement): Promise<void> {
  const formData = new FormData(form);
  const response = await fetch("/api/asset/import", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to import asset: ${response.status}`);
  }
}

export async function resetSoundTimeline(): Promise<void> {
  const response = await fetch("/api/sound-timeline/reset-run", {
    method: "POST",
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to reset soundtrack: ${response.status}`);
  }
}

function cloneAsset(asset: VideoAsset | null): VideoAsset | null {
  if (asset === null) {
    return null;
  }
  return typeof structuredClone === "function"
    ? structuredClone(asset)
    : JSON.parse(JSON.stringify(asset));
}
