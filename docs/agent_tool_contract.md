# Agent Tool Contract

## Active coarse tools

- `probe_video(video_path) -> VideoProbe`
- `detect_scenes(video_path, probe) -> SceneBoundary[]`
- `sample_frames(video_path, scenes, output_dir) -> FrameBatch[]`
- `create_silent_analysis_video(video_path, output_path) -> str`
- `propose_source_hypotheses(frame_batches, storyboard_path, output_root) -> open-world source hypotheses + provenance`
- `verify_scene_hypotheses(frame_batches, scene_hypotheses_by_window, moving_regions_by_window, storyboard_path) -> grounded prompts + region seeds + semantic hints`
- `extract_entities(frame_batches, prompts_by_scene, region_seeds_by_scene) -> Sam3TrackSet` via remote GPU inference RPC
- `embed_entities(image_paths_by_track) -> EntityEmbedding[]` via remote GPU inference RPC
- `score_track_labels(image_paths, labels) -> LabelScore[]` via remote GPU inference RPC
- `group_entity_embeddings(embeddings, tracks_by_id) -> CandidateGroupSet`
- `build_source_semantics(...) -> sources + events + ambience + groups + routes`
- `rerun_description_writer(...) -> GenerationGroup[]`
- `validate_bundle(bundle) -> ValidationIssue[]`

## Proposal-stage expectations

The active silent-video proposal stage should combine:
- Gemini open-world source proposal from sampled frames + storyboard + motion crops
- SigLIP2 grounding over Gemini-proposed phrases only
- motion-region proposals from frame differencing

The output of that stage should split into:
- extraction prompts for SAM3
- region-grounded SAM seeds
- semantic hints for downstream interpretation
- rejected / unresolved phrases
- provenance explaining what Gemini proposed and what grounding confirmed

## Gemini usage contract

Gemini may only see:
- sampled still frames
- storyboard panels
- motion-region crops
- track crops
- optional **audio-stripped** short clips in targeted repair paths

Gemini must not receive:
- uploaded source videos
- any media with audio attached

## Agentic control flow

`agentic_tool_first` should behave as:
- foundation pipeline first in interim mode
- then selective ambiguity repair only when the issue is high-value

High-value agentic classes are:
- weak or missing foreground discovery
- conflicting scene/source hypotheses
- generation-group merge/split ambiguity
- unresolved route decisions
- stale or unresolved descriptions

## Non-goals for this layer

- No audio tools
- No local CUDA/GPU execution
- No server-side semantic orchestration on the GPU host
- No Gemini video upload path
- No legacy compatibility mode
- No grouped-analysis export shape
