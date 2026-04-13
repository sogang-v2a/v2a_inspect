# Agent Tool Contract

## Active coarse tools

- `probe_video(video_path) -> VideoProbe`
- `detect_scenes(video_path, probe) -> SceneBoundary[]`
- `sample_frames(video_path, scenes, output_dir) -> FrameBatch[]`
- `create_silent_analysis_video(video_path, output_path) -> str`
- `propose_source_hypotheses(frame_batches, storyboard_path, output_root) -> prompts + hypotheses + provenance`
- `extract_entities(frame_batches, prompts_by_scene) -> Sam3TrackSet`
- `embed_entities(image_paths_by_track) -> EntityEmbedding[]`
- `group_entity_embeddings(embeddings, tracks_by_id) -> CandidateGroupSet`
- `route_track(track) -> TrackRoutingDecision`
- `aggregate_group_routes(group_id, member_track_ids, decisions_by_track_id) -> GroupRoutingDecision`

## New proposal-stage expectations

The active silent-video source proposal stage should combine:
- large-ontology SigLIP2 scoring
- Gemini frame/storyboard hypotheses
- motion-region proposals from frame differencing

The output of that stage should split into:
- extraction prompts for SAM3
- semantic hints for downstream grouping/routing
- provenance explaining where each proposal came from

## Gemini usage contract

Gemini may only see:
- sampled still frames
- storyboard panels
- optional **audio-stripped** short clips in targeted recovery paths

Gemini must not receive:
- uploaded source videos
- any media with audio attached

## Agentic control flow

`agentic_tool_first` should behave as:
- foundation pipeline first
- then selective ambiguity repair only when the issue is high-value

High-value agentic classes are:
- weak or missing foreground discovery
- conflicting scene/source hypotheses
- generation-group merge/split ambiguity
- route ambiguity with downstream impact
- stale descriptions after structural edits

## Non-goals for this layer

- No audio tools
- No local CUDA/GPU execution
- No Gemini video upload path
- No legacy compatibility mode
