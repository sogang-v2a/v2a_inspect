# Agent Tool Contract

## Coarse tools

- `probe_video(video_path) -> VideoProbe`
- `detect_scenes(video_path, probe) -> SceneBoundary[]`
- `sample_frames(video_path, scenes, output_dir) -> FrameBatch[]`
- `extract_entities_with_sam3(frame_batches) -> Sam3TrackSet`
- `embed_entities(image_paths_by_track) -> EntityEmbedding[]`
- `group_entity_embeddings(embeddings, tracks_by_id) -> CandidateGroupSet`
- `route_track(track) -> TrackRoutingDecision`
- `aggregate_group_routes(group_id, member_track_ids, decisions_by_track_id) -> GroupRoutingDecision`

## Mid-level tools

- runtime environment validation and bootstrap helpers
- `Sam3Client.recover_with_text_prompt(...)` for recovery-only extraction
- remote upload helper: store a video on the server before analyze

## Direct Gemini Tool List (target architecture)

In the final architecture, Gemini should have direct control over the
server-side visual tools. These are not just summaries or hints; they are
the callable tool surface Gemini can use during reasoning.

### Video / scene tools

- `probe_video(video_path) -> VideoProbe`
  - returns duration, fps, resolution, codec/container metadata
- `detect_scenes(video_path, probe, target_scene_seconds=5.0) -> SceneBoundary[]`
  - returns scene windows used for downstream sampling/tracking
- `sample_frames(video_path, scenes, output_dir, frames_per_scene=3) -> FrameBatch[]`
  - extracts representative frames per scene

### Prompt / detection tools

- `suggest_scene_prompts(frame_batches) -> dict[scene_index, list[str]]`
  - uses label scoring to suggest likely foreground concepts for each scene
  - examples: `cat`, `person`, `car`, `dog`, `boat`
- `extract_entities(frame_batches, prompts_by_scene) -> Sam3TrackSet`
  - runs SAM3 to produce candidate boxes/masks/tracks
  - prompt-aware foreground extraction
- `recover_with_text_prompt(frame_batches, text_prompt) -> Sam3TrackSet`
  - recovery-only path when default extraction misses a target object

### Track / crop tools

- `crop_tracks(frame_batches, sam3_track_set) -> dict[track_id, list[str]]`
  - creates per-track image crops from sampled frames
  - crops are the canonical visual input for embeddings and labels

### Embedding / label tools

- `embed_images(track_crop_paths) -> EntityEmbedding[]`
  - runs DINOv2 over track crops
  - returns one embedding per track
- `score_track_labels(track_crop_paths, labels) -> CanonicalLabel[]`
  - runs SigLIP2 over track crops
  - returns label scores and best label per track/group

### Grouping / routing tools

- `group_entity_embeddings(embeddings, tracks_by_id) -> CandidateGroupSet`
  - code-side similarity grouping over track embeddings
- `route_track(track) -> TrackRoutingDecision`
  - code-side routing hint for TTA/VTA tendencies
- `aggregate_group_routes(group_id, member_track_ids, decisions_by_track_id) -> GroupRoutingDecision`
  - code-side group-level routing hint

## Intended control flow

Gemini should be able to use these tools iteratively, not just consume
their precomputed summaries.

Target pattern:

1. Gemini probes, samples, and inspects a scene.
2. Gemini requests foreground extraction.
3. If extraction is weak, Gemini retries with a narrower prompt.
4. Gemini requests crop embeddings and label scores.
5. Code proposes groups/routing hints.
6. Gemini adjudicates merge/split, naming, verification, and final decisions.

## Important distinction from the current implementation

Current implementation:
- server runs tools first
- Gemini receives summaries/hints:
  - `tool_scene_summary`
  - `tool_grouping_hints`
  - `tool_verify_hints`
  - `tool_routing_hints`

Final intended implementation:
- Gemini has direct access to the above callable tool surface
- Gemini can request targeted re-runs and object-specific recovery

## Package split

- Client-safe tooling and shared types stay in `v2a_inspect`.
- Remote runtime helpers and future heavy dependencies stay in `v2a_inspect_server`.
- The active deployment target is a single remote GPU-hosted server runtime, with `sogang_gpu` as the default target.
- Hugging Face is a weights/artifact source only.

## Non-goals for this layer

- No audio tools
- No local CUDA/GPU execution
- No Gemini upload or Gemini prompt orchestration in the tool contract
- No Hugging Face inference endpoint interoperability
- No requirement to optimize for multiple GPU providers in the current slice
