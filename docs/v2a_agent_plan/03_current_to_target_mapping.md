# Current-to-Target Mapping

This short note exists to keep the migration concrete.

## Current repository concepts

- `VideoSceneAnalysis`
- `SceneBoundary` as fixed windows
- `RawTrack`
- `TrackGroup`
- `tool_*_hints` text blocks
- `tool_context` as a bundled precomputation step

## Target concepts

- `CandidateCut`
- `EvidenceWindow`
- `PhysicalSourceTrack`
- `SoundEventSegment`
- `AmbienceBed`
- `GenerationGroup`
- `RoutingDecision`
- `ValidationIssue`
- `MultitrackDescriptionBundle`

## Mapping rules

| Current | Target | Notes |
|---|---|---|
| `SceneBoundary` fixed window | `CandidateCut` + `EvidenceWindow` | the current windowing logic becomes fallback structure only |
| `Sam3TrackSet` | intermediate extraction artifact | not a final domain object |
| `RawTrack` | adapter to source/event output | eventually removed |
| `TrackGroup` | adapter to generation-group output | eventually removed |
| `tool_grouping_hints` etc. | direct tool calls | hints are compatibility only |
| `GroupedAnalysis` | temporary export adapter | final source of truth should be the bundle |

## Important semantic warning

The current code often invites this wrong equivalence:

`same track group` = `same source` = `same sound recipe`

That equivalence must be broken in the final architecture.
