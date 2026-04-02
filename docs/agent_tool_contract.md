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

- `post_json(...)` for remote endpoint invocation
- `Sam3RunpodClient.recover_with_text_prompt(...)` for recovery-only extraction

## Non-goals for this layer

- No audio tools
- No local CUDA/GPU execution
- No Gemini upload or Gemini prompt orchestration in the tool contract
