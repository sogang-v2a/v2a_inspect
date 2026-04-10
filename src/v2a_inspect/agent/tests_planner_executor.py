from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

from v2a_inspect.agent import (
    AgentIssue,
    PlannerState,
    PlannedAction,
    RetryBudget,
    ToolExecutor,
    mark_issue_attempted,
    plan_next_action,
    resolve_issue,
)


class PlannerExecutorTests(unittest.TestCase):
    def test_planner_picks_highest_priority_open_issue(self) -> None:
        state = PlannerState(
            video_id="vid-001",
            issues=[
                AgentIssue(
                    issue_id="issue-1",
                    issue_type="validation_issue",
                    description="late",
                    priority=50,
                ),
                AgentIssue(
                    issue_id="issue-0",
                    issue_type="structural_gap",
                    description="early",
                    priority=10,
                ),
            ],
        )
        action = plan_next_action(state)
        self.assertIsNotNone(action)
        action = cast(PlannedAction, action)
        self.assertEqual(action.issue_id, "issue-0")
        self.assertEqual(action.tool_name, "structural_overview")

    def test_budget_blocks_exhausted_issue(self) -> None:
        state = PlannerState(
            video_id="vid-001",
            retry_budget=RetryBudget(extraction_retry_limit=0),
            issues=[
                AgentIssue(
                    issue_id="issue-0",
                    issue_type="structural_gap",
                    description="blocked",
                    priority=10,
                )
            ],
        )
        self.assertIsNone(plan_next_action(state))

    def test_executor_logs_and_replays_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.jsonl"
            executor = ToolExecutor(
                registry={
                    "structural_overview": lambda **kwargs: {
                        "storyboard_path": "/tmp/storyboard.jpg",
                        **kwargs,
                    }
                },
                trace_path=trace_path,
            )
            state = PlannerState(
                video_id="vid-001",
                issues=[
                    AgentIssue(
                        issue_id="issue-0",
                        issue_type="structural_gap",
                        description="gap",
                        priority=10,
                    )
                ],
            )
            action = plan_next_action(state)
            self.assertIsNotNone(action)
            action = cast(PlannedAction, action)
            state, result = executor.execute(state, action)
            state = executor.record_decision(
                state,
                issue_id="issue-0",
                decision="accept",
                rationale="looks good",
                confidence=0.9,
            )
            replay = executor.replay_trace()
        self.assertEqual(
            cast(dict[str, object], result)["storyboard_path"], "/tmp/storyboard.jpg"
        )
        self.assertEqual(len(state.tool_calls), 1)
        self.assertEqual(len(replay), 2)
        self.assertEqual(replay[0]["kind"], "tool_call")
        self.assertEqual(replay[1]["kind"], "decision")

    def test_mark_and_resolve_issue(self) -> None:
        state = PlannerState(
            video_id="vid-001",
            issues=[
                AgentIssue(
                    issue_id="issue-0",
                    issue_type="grouping_ambiguity",
                    description="group",
                    priority=1,
                )
            ],
        )
        state = mark_issue_attempted(state, "issue-0")
        state = resolve_issue(state, "issue-0")
        self.assertEqual(state.issues[0].attempts, 1)
        self.assertEqual(state.issues[0].status, "resolved")

    def test_planner_maps_new_ambiguity_issue_types_to_tools(self) -> None:
        state = PlannerState(
            video_id="vid-001",
            issues=[
                AgentIssue(
                    issue_id="issue-cut",
                    issue_type="cut_ambiguity",
                    description="cut ambiguity",
                    priority=1,
                ),
                AgentIssue(
                    issue_id="issue-desc",
                    issue_type="description_stale",
                    description="description stale",
                    priority=2,
                ),
            ],
        )
        action = plan_next_action(state)
        self.assertIsNotNone(action)
        action = cast(PlannedAction, action)
        self.assertEqual(action.tool_name, "refine_candidate_cuts")


if __name__ == "__main__":
    unittest.main()
