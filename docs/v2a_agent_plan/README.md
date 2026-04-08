# V2A Agent Project Plan

This directory contains the requested end-state blueprint and the staged roadmap for finishing the project.

## Reading order for a human or agent

1. [AGENT_READ_FIRST.md](AGENT_READ_FIRST.md)
2. [01_final_project_blueprint.md](01_final_project_blueprint.md)
3. [02_roadmap_overview.md](02_roadmap_overview.md)
4. Read the current stage's `main`, then `detailed`, then `checklist` file.

## Included documents

- `AGENT_READ_FIRST.md` — execution rules and guardrails.
- `01_final_project_blueprint.md` — the intended final form of the entire project.
- `02_roadmap_overview.md` — summary of all stages and the critical path.
- `stages/` — one directory per stage, each containing:
  - `stage_X_main.md`
  - `stage_X_detailed.md`
  - `stage_X_checklist.md`

## Important reminder

This plan intentionally changes the target architecture from the current hybrid, Gemini-first scaffold to a **tool-first, agentic adjudication system**. The reason is simple: the main goal is not just to get any JSON out of a video, but to build a stable, reviewable, dataset-producing **multitrack audio description pipeline** from silent video.

## Suggested use

- Treat the blueprint as the source of truth.
- Treat the roadmap as the implementation order.
- Treat the checklists as exit gates.
- Do not skip a stage merely because another stage looks more exciting.

- `03_current_to_target_mapping.md` — migration reference from current repo concepts to the final ontology.
