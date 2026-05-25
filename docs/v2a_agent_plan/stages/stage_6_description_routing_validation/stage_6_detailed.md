# Stage 6 Detailed Plan: Canonical Description Synthesis, TTA/VTA Routing, Validation, and Human Review

[Back to stage main](stage_6_main.md) · [Go to checklist](stage_6_checklist.md)

## Intended result

By the end of Stage 6, the project should have crossed a clear boundary: Done means the system outputs a reviewable, exportable multitrack description bundle that can actually drive downstream generation experiments or dataset building.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- Agentic tool loop from Stage 5.
- Generation groups from Stage 4.
- The project requirement to support downstream TTA/VTA selection and validation loops.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- Canonical descriptions per generation group.
- Final TTA/VTA routing decisions with rationale and confidence.
- A validation report with actionable issues.
- A review UI that can write edits back into pipeline state.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Synthesize canonical descriptions from structured attributes

Do not rely only on raw free-form text copied from one segment. Use the structured evidence built so far: action type, materials, rhythm, intensity, spatial context, ambience, and secondary sounds. Then ask Gemini to produce compact but generation-ready canonical descriptions grounded in those fields.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Replace the old routing logic

Routing should no longer be derived from the old 3-score threshold formula alone. Use deterministic priors first—background ambience trends toward TTA; short, tightly synchronized discrete events trend toward VTA; crowded mixed scenes trend toward TTA unless a clear single event dominates—then let the adjudicator resolve ambiguous cases. Keep confidence and rationale on the final route.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Build validators that understand the new ontology

Validators should check coverage, overlapping contradictory assignments, unresolved low-confidence identities, suspicious generation-group merges, route inconsistency, and description quality. Each issue should be typed and tied to one rerun path or a human-review action.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Make the UI a real correction surface

The review UI should no longer be read-only or metadata-only. A reviewer should be able to split or merge generation groups, rename a source, override a route, mark an extraction as missing, or approve a low-confidence item. Those edits should become part of the saved bundle and optionally trigger local repairs.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Replace or heavily refactor the current model-selection logic.
- Add new validation modules and issue types.
- Update Streamlit result rendering to show source tracks, event segments, generation groups, routes, and issues.
- Add review-edit application utilities.

## Implementation notes for the current repository

- Replace the old threshold-based model selector rather than layering more heuristics on top of it.
- Canonical descriptions should be generated from structured attributes and evidence, not copied blindly from one arbitrary member segment.
- Validators should be typed and tied to repair actions or review actions.
- UI edits must persist into the final bundle or they are not real corrections.
- Route explanations should remain compact but concrete enough for later audit.

## What must not happen in this stage

- No final paper packaging yet.
- No large-scale dataset export yet.

In addition to the items above, do not silently introduce new semantics that belong to later stages.

## Suggested milestone breakdown inside the stage

A useful internal cadence for this stage is:

1. add or modify contracts and tests first
2. implement the minimal code path
3. run fixtures and inspect artifacts manually
4. only then refine thresholds or prompts
5. update docs and adapters before declaring the stage complete

## Tests and measurement for this stage

A stage is only meaningful if there is a way to tell whether it worked.

- Routing tests on fixed heuristic cases.
- Validator tests for each issue type.
- UI state/edit tests where possible.
- Bundle export tests.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- Descriptions become verbose but not generation-useful.
- Routing logic becomes too model-specific too early.
- Human edits are not persisted, making the UI misleading.

## Questions the agent should ask before merging changes

- Are canonical descriptions grounded in structured evidence?
- Does the routing logic still secretly depend on the old threshold formula?
- Do human review edits actually change saved output?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means the system outputs a reviewable, exportable multitrack description bundle that can actually drive downstream generation experiments or dataset building.
