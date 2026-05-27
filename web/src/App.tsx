import { FormEvent, useEffect, useState } from "react";
import { fetchAsset, startRun } from "./api";
import VideoEditor from "./components/VideoEditor";
import type { AssetResponse } from "./types";

const emptyAsset: AssetResponse = {
  status: "idle",
  stage: null,
  error: null,
  version: 0,
  asset_version: 0,
  updated_at: "",
  asset: null,
  timeline_rows: [],
};

export default function App() {
  const [state, setState] = useState<AssetResponse>(emptyAsset);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    void refreshAsset();
    const events = new EventSource("/events");
    events.addEventListener("asset_update", () => {
      void refreshAsset();
    });
    return () => events.close();
  }, []);

  async function refreshAsset() {
    const nextState = await fetchAsset();
    setState(nextState);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitError(null);
    try {
      await startRun(event.currentTarget);
      await refreshAsset();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <VideoEditor
      state={state}
      submitError={submitError}
      onSubmit={handleSubmit}
    />
  );
}
