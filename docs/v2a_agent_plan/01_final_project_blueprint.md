# Final Project Blueprint

## 1. Project in one sentence

Build a **tool-first, agentic silent-video analysis system** that converts a video **without using its audio** into a **reviewable multitrack audio description bundle** for downstream audio generation, dataset construction, and research evaluation.

---

## 2. Why the final architecture should change

The current repository is a valuable transition scaffold, but it is still fundamentally **Gemini-first** in its core reasoning path. The final project should not stay there.

The end-state should change because the current hybrid shape has several structural limits:

- it treats Gemini scene analysis as the main unit of structure
- it lets tool outputs exist mostly as hint text rather than as direct callable evidence
- it still mixes source identity and generation grouping into one ambiguous object
- it lacks a trustworthy “where should we cut?” layer
- it is not yet built around crop-level evidence

That architecture can produce outputs, but it will be difficult to scale, debug, evaluate, and trust for dataset production.

The final system should therefore be:

- **tool-first for structure**
- **agentic for ambiguity resolution**
- **explicitly layered for semantics**
- **reviewable and exportable**

---

## 3. Final success definition

The project is successful when a silent video can be processed into a final bundle that contains:

1. a structural breakdown of the video into evidence windows
2. stable visible sound-producing source tracks
3. event segments describing what those sources are doing over time
4. ambience beds separated from discrete foreground events
5. generation groups that reflect acoustic equivalence rather than raw identity
6. canonical sound descriptions for each generation group
7. TTA/VTA routing decisions with rationale and confidence
8. a validation report with typed issues
9. enough evidence artifacts for a human to inspect and correct the result
10. a stable export format suitable for dataset construction

If the system cannot produce those ten things, it is not yet in the final intended form.

---

## 4. Core architectural principles

### 4.1 Deterministic structure first, adjudication second

The system should not begin by asking Gemini to invent the entire structure of the video. Instead:

- deterministic or semi-deterministic tools should propose cuts, windows, sources, and provisional groupings
- the agent should use Gemini to decide hard or ambiguous cases
- Gemini should synthesize final descriptions and final route decisions only after the structural evidence exists

### 4.2 Separate identity from generation grouping

These are different questions:

- **Identity question:** Is this the same visible sound-producing source?
- **Generation question:** Can these segments share one canonical sound description and one downstream generation recipe?

The final project must keep those questions separate.

### 4.3 Crop-level visual evidence is mandatory

Once the crop tool exists, DINOv2 embeddings and SigLIP2 label scores must come from crops, not whole-scene frames. This is a major quality boundary.

### 4.4 Ambience is not a trash bin

Background ambience, room tone, environment beds, and diffuse scene texture should be represented explicitly rather than used as a catch-all for all extra sources.

### 4.5 Human review must remain possible

The system is allowed to be highly automated, but it must still expose its evidence and rationale so that a human can inspect, correct, and export.

---

## 5. Final conceptual data model

The final project should use **three semantic layers** plus validation and routing metadata.

| Layer | Meaning | Example |
|---|---|---|
| `PhysicalSourceTrack` | A persistent visible sound-producing entity or source region | one person, one car, one machine, one waterfall region |
| `SoundEventSegment` | A time-localized acoustic behavior or event from one source | footsteps while running, hand impact, door close, engine rev burst |
| `GenerationGroup` | A set of event segments or ambience beds that can share one canonical sound recipe | all “running footsteps on pavement” segments across multiple people/scenes |

This resolves the main ambiguity in the current system.

### 5.1 `PhysicalSourceTrack`

A `PhysicalSourceTrack` should contain at least:

- `source_id`
- `kind`: `foreground`, `background_region`, or another controlled enum
- `label_candidates`: scored labels, not just a single label
- `spans`: one or more time ranges where the source is present
- `evidence_refs`: frame IDs, crop IDs, optional clip IDs
- `identity_confidence`
- `reid_neighbors` or equivalent identity-edge evidence

A `PhysicalSourceTrack` is not yet a sound recipe.

### 5.2 `SoundEventSegment`

A `SoundEventSegment` should contain at least:

- `event_id`
- `source_id`
- `start_time`, `end_time`
- `event_type` or `action_class`
- `sync_strength`
- `motion_profile`
- `interaction_flags`
- `material_or_surface` when inferable
- `intensity` / `texture` / `pattern` fields
- `confidence`

This layer is where the project becomes event-aware rather than only object-aware.

### 5.3 `AmbienceBed`

An `AmbienceBed` should contain:

- `ambience_id`
- `start_time`, `end_time`
- `environment_type`
- `acoustic_profile`
- `foreground_exclusion_notes`
- `confidence`

Use ambience only when visually justified. Do not invent invisible soundtrack layers.

### 5.4 `GenerationGroup`

A `GenerationGroup` should contain:

- `group_id`
- `member_event_ids`
- `member_ambience_ids` if applicable
- `canonical_label`
- `canonical_description`
- `group_confidence`
- `route_decision`
- `reasoning_summary`

A `GenerationGroup` answers: “Which segments should share the same generation recipe?”

### 5.5 `RoutingDecision`

A route decision should contain:

- `model_type`: `TTA` or `VTA`
- `confidence`
- `factors`: structured factors used to decide
- `reasoning`
- `rule_based` flag if a deterministic rule was used

### 5.6 `ValidationReport`

A validation report should contain typed issues such as:

- low-confidence identity merge
- suspicious cross-scene generation merge
- missing dominant source
- overlapping contradictory assignments
- route inconsistency
- overly vague description
- unreviewed low-confidence output

---

## 6. Final output bundle

The final export should be a `MultitrackDescriptionBundle` or equivalent.

Example shape:

```json
{
  "video_id": "example_001",
  "video_meta": {
    "duration_seconds": 18.4,
    "fps": 29.97,
    "width": 1920,
    "height": 1080
  },
  "candidate_cuts": [],
  "evidence_windows": [],
  "physical_sources": [],
  "sound_events": [],
  "ambience_beds": [],
  "generation_groups": [],
  "validation": {
    "status": "pass_with_warnings",
    "issues": []
  },
  "artifacts": {
    "storyboard_dir": "...",
    "crop_dir": "...",
    "clip_dir": "..."
  },
  "pipeline_metadata": {
    "pipeline_version": "vNext",
    "tool_versions": {
      "sam3": "...",
      "dinov2": "...",
      "siglip2": "..."
    }
  }
}
```

### Important consequence

The final system should export both:

- a **rich internal bundle** for research, validation, and review
- a **lean downstream generation view** that downstream TTA/VTA steps can consume without needing all internal evidence

---

## 7. Final end-to-end pipeline

The final pipeline should look like this:

```text
Silent video
-> probe + metadata
-> hard cut proposals + evidence windows
-> sampled frames + storyboard
-> source extraction + crop generation
-> DINOv2 re-id + SigLIP2 labels
-> provisional physical source tracks
-> event segmentation + ambience-bed construction
-> provisional generation groups
-> agentic adjudication (merge/split/recover)
-> canonical description synthesis
-> TTA/VTA routing
-> validation + targeted repair
-> human review (optional but supported)
-> final multitrack description bundle
```

### 7.1 Structural pass

The structural pass should answer:

- where are the obvious hard cuts?
- what windows should later tools inspect?
- what is the minimal evidence package for each window?

This pass is high-recall, not final truth.

### 7.2 Source pass

The source pass should answer:

- what visible sound-producing entities or source regions are present?
- which crops and labels support those sources?
- where is identity ambiguous?

### 7.3 Event pass

The event pass should answer:

- what acoustically distinct behaviors occur within each source track?
- which events are sync-critical?
- which are diffuse ambience rather than discrete events?

### 7.4 Grouping pass

The grouping pass should answer:

- which event segments share one canonical sound recipe?
- which apparently similar segments must stay separate because the acoustic context differs?

### 7.5 Routing pass

The routing pass should answer:

- should this generation group go to TTA or VTA?
- how confident is that decision?
- what evidence supports it?

### 7.6 Validation and repair pass

The validation pass should answer:

- what still looks wrong or weak?
- what can be repaired automatically?
- what should be surfaced for human review?

---

## 8. Final tool catalog

The agent should have a small, high-value tool surface. The tools below are the intended final surface.

### 8.1 Structural tools

| Tool | Purpose | Support |
|---|---|---|
| `probe_video(video_path)` | Read duration, fps, resolution, stream metadata | ffprobe / deterministic |
| `detect_shot_boundaries(video_path)` | Propose hard shot cuts | ffmpeg / PySceneDetect-style logic |
| `propose_candidate_cuts(video_path, shot_boundaries, structural_signals)` | Build cut proposals with reasons and confidence | deterministic + heuristics |
| `build_evidence_windows(video_path, candidate_cuts)` | Turn cut proposals into windows | deterministic |
| `sample_frames(video_path, evidence_windows)` | Create representative frames | ffmpeg |
| `make_storyboard(evidence_windows)` | Build a compact global visual summary | deterministic |
| `extract_short_clip(video_path, start, end)` | Create local clip evidence for ambiguous windows | ffmpeg |

### 8.2 Source extraction tools

| Tool | Purpose | Primary model |
|---|---|---|
| `extract_entities_prompt_free(evidence_windows)` | Default extraction path | SAM3 under an external prompt-free contract |
| `recover_with_text_prompt(evidence_windows, text_prompt)` | Recovery-only extraction when needed | SAM3 recovery path |
| `crop_tracks(frame_batches, track_set)` | Produce per-track crop evidence | deterministic geometry |
| `embed_track_crops(track_crop_paths)` | Re-id / similarity features | DINOv2 |
| `score_track_labels(track_crop_paths, labels)` | Label scoring and prompt suggestions | SigLIP2 |
| `build_reid_graph(tracks, embeddings, labels)` | Stable source-track proposals | deterministic + similarity rules |

### 8.3 Event and grouping tools

| Tool | Purpose | Support |
|---|---|---|
| `split_source_into_events(source_track, structural_signals)` | Produce event segments from one source | heuristics + optional adjudication |
| `build_ambience_beds(evidence_windows, source_tracks)` | Create ambience layers | deterministic + adjudication |
| `propose_generation_groups(event_segments, ambience_beds)` | Cluster by acoustic equivalence | deterministic + similarity rules |
| `score_group_labels(group_evidence)` | Canonical naming support | SigLIP2 + Gemini naming |
| `route_generation_group(group)` | Produce routing prior | deterministic routing priors |
| `aggregate_routes(groups)` | Group/clip-level routing summaries | deterministic |

### 8.4 Validation tools

| Tool | Purpose | Support |
|---|---|---|
| `validate_bundle(bundle)` | Run global checks | deterministic |
| `find_low_confidence_merges(bundle)` | Identify risky identity/group merges | deterministic |
| `find_missing_sources(bundle)` | Coverage checks | deterministic + heuristics |
| `export_review_packet(bundle)` | Create human-review assets | deterministic |

---

## 9. Model usage in the final system

The final system should use the existing model stack in a disciplined way.

### 9.1 Gemini

Gemini should be used for:

- adjudicating ambiguous cuts
- deciding difficult merge/split cases
- synthesizing canonical descriptions from structured evidence
- resolving uncertain route decisions
- writing compact rationales and summaries

Gemini should **not** be the universal first-pass extractor.

### 9.2 SAM3

SAM3 should be used for:

- entity or source-region extraction
- mask/box evidence generation
- recovery when a target source was missed

If the runtime needs prompts internally, hide that behind a prompt-free external contract and keep manual text prompting as recovery-only.

### 9.3 DINOv2

DINOv2 should be used for:

- crop-level embeddings
- same-source vs different-source evidence
- cross-window and cross-cut re-identification support

### 9.4 SigLIP2

SigLIP2 should be used for:

- crop-level label scoring
- scene/window label suggestions
- canonical naming support
- sanity checks on grouping proposals

### 9.5 Additional models

Additional models are optional, not default.

If the current stack cannot satisfy a stage’s exit criteria, a new model may be introduced, but only with a written reason. For example, a separate detector such as Grounding DINO or OWLv2 may be justified later for recovery, but it should not become the default plan prematurely.

---

## 10. Final agent loop

The planner should operate as a bounded state machine, not as free-form wandering.

Recommended phases:

1. **Overview**
   - read probe, cuts, storyboard
2. **Source stabilization**
   - run extraction, crops, labels, re-id
3. **Eventization**
   - split sources into event segments, create ambience beds
4. **Grouping**
   - propose generation groups
5. **Adjudication**
   - resolve ambiguous merges/splits
6. **Routing**
   - assign TTA/VTA
7. **Validation**
   - run validators, repair targeted weak spots
8. **Finalize**
   - synthesize final descriptions and export bundle

### Bounded budget policy

Example recommended budgets for a <=60s clip:

- max 1 global overview pass
- max 2 extraction retries per ambiguous window
- max 1 recovery prompt per unresolved source
- max 2 regroup attempts per ambiguity cluster
- max 3 validation-repair rounds per video

These are examples, but some bounded-budget policy must exist.

---

## 11. Final validation philosophy

Validation should not only ask “did the JSON parse?” It should ask whether the output is usable.

Recommended validator categories:

- **coverage** — did we miss a dominant visible sound source?
- **identity** — are low-confidence merges flagged?
- **grouping** — are acoustically incompatible segments grouped together?
- **routing** — does the chosen model type fit the segment/group characteristics?
- **description quality** — is the canonical description specific enough for generation?
- **review state** — are unresolved low-confidence items marked for review?

The validators should drive targeted reruns whenever possible.

---

## 12. Final human-review UI

The final UI should show:

- storyboard overview
- candidate cuts and evidence windows
- source tracks with crop galleries
- event segments on a timeline
- generation groups and their members
- TTA/VTA route decisions
- validation issues
- edit actions

Supported edit actions should include at least:

- split or merge generation groups
- rename a canonical label
- override a route decision
- mark a source as missing
- rerun recovery for one window or one source
- approve or reject a low-confidence item

---

## 13. Final repo/package structure

A good final shape for the repository is:

```text
v2a_inspect/
├── docs/
├── src/v2a_inspect/
│   ├── agent/
│   │   ├── state.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── policies.py
│   │   └── repair.py
│   ├── contracts/
│   │   ├── video.py
│   │   ├── cuts.py
│   │   ├── evidence.py
│   │   ├── sources.py
│   │   ├── events.py
│   │   ├── groups.py
│   │   ├── routing.py
│   │   ├── validation.py
│   │   └── bundle.py
│   ├── pipeline/
│   │   ├── orchestrate.py
│   │   ├── adapters.py
│   │   └── export.py
│   ├── evaluation/
│   ├── dataset/
│   ├── review/
│   ├── ui/
│   └── settings.py
└── server/src/v2a_inspect_server/
    ├── api/
    │   ├── app.py
    │   └── tool_routes.py
    ├── runtime/
    │   ├── bootstrap.py
    │   ├── registry.py
    │   └── health.py
    ├── tools/
    │   ├── probe.py
    │   ├── cuts.py
    │   ├── frames.py
    │   ├── sam3_extract.py
    │   ├── crop_tracks.py
    │   ├── reid.py
    │   ├── labels.py
    │   ├── events.py
    │   ├── grouping.py
    │   ├── routing.py
    │   └── validation.py
    └── models/
        ├── sam3_runtime.py
        ├── dinov2_runtime.py
        └── siglip2_runtime.py
```

### Migration note

The existing files do not need to match these names immediately, but the final responsibilities should end up close to this separation.

---

## 14. Current-to-target migration summary

The following migration should happen conceptually:

| Current artifact | Final role |
|---|---|
| `VideoSceneAnalysis` | temporary compatibility artifact or coarse review view |
| `RawTrack` | deprecated; replaced by source tracks and event segments |
| `TrackGroup` | deprecated as a catch-all; replaced by generation groups |
| `SceneBoundary` fixed windows | replaced by candidate cuts + evidence windows |
| `tool_*_hints` text blocks | replaced by direct tool calls and typed outputs |
| server `tool_context` | temporary adapter; later decomposed into explicit tools |

This lets the existing code continue to help while the new architecture is built.

---

## 15. Non-goals

The first complete end-state should **not** aim for the following:

- general support for many cloud GPU providers
- local GPU or CPU parity for heavy inference
- audio generation as part of the core analysis pipeline
- broad multi-modal chat UX beyond what is needed for review
- speculative model zoo expansion without evidence of need

---

## 16. End-state acceptance criteria

The project should only be considered complete when all of the following are true:

1. the main path is tool-first, not Gemini-first
2. the agent can directly call tools
3. the output schema separates source identity from generation grouping
4. crop-level evidence is used for embeddings and labels
5. the system can explain and log its merge/split decisions
6. a human can review and correct the result
7. the pipeline exports dataset-ready artifacts
8. evaluation baselines exist for legacy vs tool-first vs full-agentic paths

If these conditions are not met, the project may still be valuable, but it is not yet in its intended final form.
