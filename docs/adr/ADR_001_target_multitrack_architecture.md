# ADR 001: Target multitrack architecture and semantic invariants

## Status
Accepted

## Context

The existing repository is a useful migration scaffold, but it still carries
scene-first and group-overloaded semantics from the transitional Gemini-heavy
pipeline. The roadmap in `docs/v2a_agent_plan/*` requires a frozen vocabulary
before more structural work proceeds.

## Decision

The project source of truth is the staged roadmap plus the final blueprint under
`docs/v2a_agent_plan/`.

The forward architecture uses these distinct semantic layers:

1. `PhysicalSourceTrack` — visible sound-producing source identity.
2. `SoundEventSegment` — time-localized acoustic behavior from a source.
3. `GenerationGroup` — acoustically equivalent recipe-sharing grouping.

The final export source of truth is `MultitrackDescriptionBundle`, not the
legacy `GroupedAnalysis` shape.

## Invariants

- No input audio is ever consumed.
- Heavy inference remains remote-only.
- Single-server runtime is the primary execution target.
- Gemini remains in the system as adjudicator/description synthesizer, not the
  universal first-pass extractor.
- Crop-level evidence is mandatory once crop support exists.
- Identity merges and generation-group merges must remain inspectable.
- Temporary adapters are allowed, but they must be labeled temporary and must
  point toward the new ontology.

## Consequences

- Legacy objects such as `RawTrack`, `TrackGroup`, and `GroupedAnalysis`
  continue only as temporary compatibility adapters.
- New stage work should build against import-light contracts under
  `v2a_inspect.contracts`.
- Gold-set fixtures and review notes become part of the architecture, not an
  afterthought.
