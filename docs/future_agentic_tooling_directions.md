# Future Agentic Tooling Directions

This note captures desired future evolution for the current bounded agentic loop.

## Current state

The system currently has a **bounded agentic repair loop** layered on top of a
mostly tool-first pipeline.

- The loop is repair-oriented, not open-ended
- It uses a limited tool palette
- Most pipeline stages are still pre-wired and deterministic

## Desired future direction

We would like the loop to become more **open-ended** over time.

The main goal is to expose **more diverse tools that abstract model
functionality**, rather than limiting the loop to large pipeline-stage reruns.

## Preferred tool design direction

Expose **capability-shaped tools**, not just stage-shaped tools.

Examples of future agent-callable tools:

- `track_visible_region(...)`
- `label_crop_open_world(...)`
- `embed_crop_set(...)`
- `find_similar_tracks(...)`
- `merge_identity_candidates(...)`
- `score_phrase_against_frames(...)`
- `reinspect_window_with_dense_sampling(...)`
- `compare_two_source_hypotheses(...)`
- `route_generation_candidate(...)`
- `rewrite_group_description(...)`

## Architectural preference

The future loop should be able to compose lower-level model capabilities such
as:

- tracking
- embeddings
- similarity search
- label scoring
- grounding
- hypothesis comparison
- routing judgment
- description rewriting

This is preferred over relying only on a small set of coarse composite tools.

## Boundary preference

Keep the current responsibility split:

- **GPU server**: inference-only worker
- **Local machine**: orchestration, semantics, and agent loop

Future tool growth should preserve that split rather than moving semantic
control back onto the GPU host.
