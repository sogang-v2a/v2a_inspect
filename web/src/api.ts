import type { AssetResponse, CurrentFrameRows } from "./types";

export async function fetchAsset(): Promise<AssetResponse> {
  const response = await fetch("/api/asset");
  if (!response.ok) {
    throw new Error(`Failed to fetch asset: ${response.status}`);
  }
  return response.json();
}

export async function fetchCurrentFrame(frame: number): Promise<CurrentFrameRows | null> {
  const response = await fetch(`/api/rows/current-frame?frame=${frame}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch frame rows: ${response.status}`);
  }
  const payload = await response.json();
  return payload.rows;
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
