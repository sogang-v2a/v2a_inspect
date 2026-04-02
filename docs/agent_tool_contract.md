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

## Package split

- Client-safe tooling and shared types stay in `v2a_inspect`.
- Remote runtime helpers and future heavy dependencies stay in `v2a_inspect_server`.
- The active deployment target is a single Runpod-hosted NVIDIA Docker runtime.
- Hugging Face is a weights/artifact source only.

## Non-goals for this layer

- No audio tools
- No local CUDA/GPU execution
- No Gemini upload or Gemini prompt orchestration in the tool contract
- No Hugging Face inference endpoint interoperability
- No requirement to optimize for multiple GPU providers in the current slice
