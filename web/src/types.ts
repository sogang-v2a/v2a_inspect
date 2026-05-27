export type RunStatus = "idle" | "running" | "complete" | "failed";

export interface AssetResponse {
  status: RunStatus;
  stage: string | null;
  error: string | null;
  version: number;
  asset_version: number;
  updated_at: string;
  asset: VideoAsset | null;
  timeline_rows: TimelineRow[];
}

export interface AssetUpdateEvent {
  status: RunStatus;
  stage: string | null;
  error: string | null;
  version: number;
  asset_version: number;
}

export interface VideoAsset {
  video_id: string;
  source_path: string;
  sam3_tracking_path?: string | null;
  frame_count: number;
  initial_scenes: InitialScene[];
  visual_identity_layer?: VisualIdentityLayer | null;
  sound_timeline?: SoundTimeline | null;
}

export interface InitialScene {
  initial_scene_id: string;
  start_frame_index: number;
  end_frame_index: number;
  keyframes: unknown[];
  initial_analysis?: unknown | null;
  scene_tracks: SceneTrack[];
}

export interface SceneTrack {
  start_frame_index: number;
  end_frame_index: number;
  confidence: number;
  points: unknown[];
  source_object_seed?: { label?: string | null } | null;
  tracking_prompt?: string | null;
}

export interface VisualIdentityLayer {
  visual_objects: unknown[];
  visual_events: VisualEvent[];
}

export interface VisualEvent {
  event_type: string;
  start_frame_index: number;
  end_frame_index: number;
  description: string;
}

export interface SoundTimeline {
  sound_sources: unknown[];
  sound_events: SoundEvent[];
}

export interface SoundEvent {
  track_type: string;
  start_frame_index: number;
  end_frame_index: number;
  description: string;
}

export interface TimelineRow {
  lane: string;
  label: string;
  start_frame: number;
  end_frame: number;
  kind: string;
}

export interface CurrentFrameRows {
  scene: SceneFrameRow | null;
  tracks: TrackFrameRow[];
  visual_events: VisualEventFrameRow[];
  sound_events: SoundEventFrameRow[];
}

export interface SceneFrameRow {
  scene: number;
  start_frame: number;
  end_frame: number;
  duration_sec: number;
}

export interface TrackFrameRow {
  scene: number;
  track: number;
  label: string;
  bbox: number[] | null;
  confidence: number;
  has_mask: boolean;
}

export interface VisualEventFrameRow {
  object: string;
  related: string;
  event_type: string;
  start_frame: number;
  end_frame: number;
  duration_sec: number;
  confidence: number;
  description: string;
  notes: string | null;
}

export interface SoundEventFrameRow {
  track_type: string;
  source: string | null;
  start_frame: number;
  end_frame: number;
  duration_sec: number;
  generation_mode: string;
  description: string;
  notes: string | null;
}
