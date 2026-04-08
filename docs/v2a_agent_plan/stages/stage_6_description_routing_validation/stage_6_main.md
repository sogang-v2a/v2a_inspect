# Stage 6: Canonical Description Synthesis, TTA/VTA Routing, Validation, and Human Review

**Suggested duration:** 1-2 focused weeks

## Goal

Take the new structural output and turn it into the final multitrack description bundle: canonical sound descriptions, TTA/VTA routing decisions, validators, repair triggers, and UI-backed review/edit flows.

## Why this stage exists

The end goal is not only to extract sources and events; it is to produce usable multitrack descriptions for downstream generation and dataset construction. This stage turns the internal graph into a final deliverable the team can inspect, edit, and export.

## Inputs

- Agentic tool loop from Stage 5.
- Generation groups from Stage 4.
- The project requirement to support downstream TTA/VTA selection and validation loops.

## Expected outputs

- Canonical descriptions per generation group.
- Final TTA/VTA routing decisions with rationale and confidence.
- A validation report with actionable issues.
- A review UI that can write edits back into pipeline state.

## Primary code areas

- New description synthesis utilities.
- A routing module that combines heuristics and adjudication.
- Validator modules.
- UI rendering and edit-application logic.

## Explicitly out of scope

- No final paper packaging yet.
- No large-scale dataset export yet.

## Main workstreams

### 1. Synthesize canonical descriptions from structured attributes

Do not rely only on raw free-form text copied from one segment. Use the structured evidence built so far: action type, materials, rhythm, intensity, spatial context, ambience, and secondary sounds. Then ask Gemini to produce compact but generation-ready canonical descriptions grounded in those fields.

### 2. Replace the old routing logic

Routing should no longer be derived from the old 3-score threshold formula alone. Use deterministic priors first—background ambience trends toward TTA; short, tightly synchronized discrete events trend toward VTA; crowded mixed scenes trend toward TTA unless a clear single event dominates—then let the adjudicator resolve ambiguous cases. Keep confidence and rationale on the final route.

### 3. Build validators that understand the new ontology

Validators should check coverage, overlapping contradictory assignments, unresolved low-confidence identities, suspicious generation-group merges, route inconsistency, and description quality. Each issue should be typed and tied to one rerun path or a human-review action.

### 4. Make the UI a real correction surface

The review UI should no longer be read-only or metadata-only. A reviewer should be able to split or merge generation groups, rename a source, override a route, mark an extraction as missing, or approve a low-confidence item. Those edits should become part of the saved bundle and optionally trigger local repairs.


## Exit criteria

- The system produces a final bundle with canonical descriptions and routing.
- Validation issues can trigger targeted reruns instead of full restarts.
- Human reviewers can inspect evidence and apply corrections that persist.
- Legacy score-threshold model selection is no longer the primary routing logic.

## How to use this stage folder

1. Read [the detailed stage plan](stage_6_detailed.md).
2. Use [the stage checklist](stage_6_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means the system outputs a reviewable, exportable multitrack description bundle that can actually drive downstream generation experiments or dataset building.
