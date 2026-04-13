from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import streamlit as st

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.observability import build_score_id, create_trace_score
from v2a_inspect.settings import settings
from v2a_inspect.workflows import InspectOptions, InspectState
from v2a_inspect.review import (
    apply_route_override,
    approve_validation_issue,
    merge_generation_groups,
    persist_bundle,
    rename_source,
    split_generation_group,
)

from .session import reset_state


def render_page_header() -> None:
    st.title("🔍 V2A Inspect — Tool-First Multitrack Description Inspector")
    st.markdown(
        "시각 증거 기반 구조화 파이프라인으로 **multitrack description bundle**을 만들고, "
        "**agentic repair**와 **사람 검토**까지 이어지는 연구용 검사 도구입니다.  \n"
        "오디오 입력/생성 없음 — video-only evidence, grouping, routing, validation만 수행합니다."
    )
    st.divider()


def render_sidebar(authenticator: Any) -> InspectOptions:
    with st.sidebar:
        st.header("⚙️ 분석 옵션")

        fps = st.slider("Analysis FPS", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        st.caption("초당 분석 프레임 수. 높을수록 정밀하지만 느림")

        pipeline_mode = cast(
            Literal["tool_first_foundation", "agentic_tool_first"],
            st.selectbox(
                "Pipeline Mode",
                ["agentic_tool_first", "tool_first_foundation"],
                index=["agentic_tool_first", "tool_first_foundation"].index(settings.visual_pipeline_mode),
                format_func=lambda value: {
                    "agentic_tool_first": "agentic_tool_first — deferred during semantic reset",
                    "tool_first_foundation": "tool_first_foundation — bundle-first Gemini semantics",
                }[value],
            ),
        )
        st.caption("모든 경로는 silent-video bundle-first 파이프라인만 사용합니다.")

        prompt_type = cast(Literal["default", "extended"], st.selectbox("Prompt Type", ["default", "extended"], index=0))
        st.caption("`default`: 간결 | `extended`: Foley 상세")

        if st.button("🔄 Reset", use_container_width=True, type="secondary"):
            reset_state()
            st.rerun()

        authenticator.logout("Logout", "sidebar")

    return InspectOptions(fps=fps, pipeline_mode=pipeline_mode, scene_analysis_mode=prompt_type)


def render_results(
    *,
    video_path: str,
    clip_dir: str,
    inspect_state: InspectState | None,
) -> None:
    del video_path
    st.divider()
    st.header("Step 2: 분석 결과 요약")

    _render_state_messages(inspect_state)
    _render_langfuse_summary(inspect_state)

    bundle = inspect_state.get("multitrack_bundle") if inspect_state else None
    if isinstance(bundle, MultitrackDescriptionBundle):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🪟 Evidence windows", len(bundle.evidence_windows))
        c2.metric("🔊 Physical sources", len(bundle.physical_sources))
        c3.metric("🎞️ Sound events", len(bundle.sound_events))
        c4.metric("🧩 Generation groups", len(bundle.generation_groups))
        st.divider()
        st.header("Step 3: Multitrack bundle review")
        _render_bundle_review(bundle=bundle, inspect_state=inspect_state, clip_dir=clip_dir)
        return
    st.info("No multitrack bundle is available yet.")


def render_footer() -> None:
    st.divider()
    st.caption("V2A Inspect | bundle-first Gemini semantics | No audio input or generation")


def _render_state_messages(inspect_state: InspectState | None) -> None:
    if inspect_state is None:
        return
    warnings = inspect_state.get("warnings", [])
    progress_messages = inspect_state.get("progress_messages", [])
    if warnings:
        with st.expander("⚠️ 워크플로우 경고", expanded=True):
            for message in warnings:
                st.warning(message)
    if progress_messages:
        with st.expander("🧭 워크플로우 로그", expanded=False):
            for message in progress_messages:
                st.write(f"- {message}")


def _render_langfuse_summary(inspect_state: InspectState | None) -> None:
    if inspect_state is None:
        return
    trace_id = inspect_state.get("trace_id")
    if not trace_id:
        return
    st.caption(f"Langfuse trace id: `{trace_id}`")
    with st.expander("🧪 Langfuse Review", expanded=False):
        quality_key = "langfuse_overall_grouping_quality"
        approval_key = "langfuse_approved_for_export"
        quality_score = st.slider(
            "Overall grouping quality",
            min_value=1,
            max_value=5,
            value=int(st.session_state.get(quality_key, 3)),
            key=quality_key,
        )
        approved = st.checkbox(
            "Approved for export",
            value=bool(st.session_state.get(approval_key, False)),
            key=approval_key,
        )
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Save overall score", key="langfuse_save_overall_score"):
                success = create_trace_score(
                    trace_id=trace_id,
                    name="overall_grouping_quality",
                    value=float(quality_score),
                    data_type="NUMERIC",
                    score_id=build_score_id(trace_id, "overall_grouping_quality"),
                    metadata={"scale": "1-5"},
                    flush=True,
                )
                if success:
                    st.success("Saved overall grouping score to Langfuse.")
                else:
                    st.warning("Langfuse is not configured, so the score was not sent.")
        with col_right:
            if st.button("Save approval", key="langfuse_save_approval"):
                success = create_trace_score(
                    trace_id=trace_id,
                    name="approved_for_export",
                    value=1.0 if approved else 0.0,
                    data_type="BOOLEAN",
                    score_id=build_score_id(trace_id, "approved_for_export"),
                    metadata={"approved": approved},
                    flush=True,
                )
                if success:
                    st.success("Saved export approval to Langfuse.")
                else:
                    st.warning("Langfuse is not configured, so the score was not sent.")


def _render_bundle_review(
    *,
    bundle: MultitrackDescriptionBundle,
    inspect_state: InspectState,
    clip_dir: str,
) -> None:
    st.caption("Evidence windows, source tracks, event segments, generation groups, validation issues, and persisted review edits.")
    with st.expander("📦 Final bundle JSON", expanded=False):
        st.json(bundle.model_dump(mode="json"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence windows", len(bundle.evidence_windows))
    c2.metric("Physical sources", len(bundle.physical_sources))
    c3.metric("Sound events", len(bundle.sound_events))
    c4.metric("Generation groups", len(bundle.generation_groups))

    with st.expander("🪟 Evidence windows & sources", expanded=False):
        for window in bundle.evidence_windows:
            st.markdown(
                f"- `{window.window_id}` {window.start_time:.1f}s–{window.end_time:.1f}s | frames={len(window.sampled_frame_ids)} | artifacts={len(window.artifact_refs)}"
            )
        for source in bundle.physical_sources:
            top_label = source.label_candidates[0].label if source.label_candidates else "unresolved"
            st.markdown(
                f"- `{source.source_id}` label={top_label} kind={source.kind} spans={len(source.spans)} tracks={len(source.track_refs)} crops={len(source.crop_refs)}"
            )

    with st.expander("🔉 Event segments & generation groups", expanded=True):
        for group in bundle.generation_groups:
            route_label = group.route_decision.model_type if group.route_decision is not None else "unresolved"
            route_conf = f"{group.route_decision.confidence:.0%}" if group.route_decision is not None else "—"
            description = group.canonical_description or "(unresolved description)"
            st.markdown(f"### `{group.group_id}` — {description}")
            st.caption(
                f"route={route_label} conf={route_conf} | events={len(group.member_event_ids)} | ambience={len(group.member_ambience_ids)}"
            )
            override_default = route_label if route_label in {"TTA", "VTA"} else "TTA"
            override = st.selectbox(
                f"Route override for {group.group_id}",
                [override_default, "TTA", "VTA"],
                key=f"bundle_route_override_{group.group_id}",
            )
            if st.button(f"Apply route override {group.group_id}", key=f"apply_route_override_{group.group_id}"):
                updated = apply_route_override(bundle, group_id=group.group_id, model_type=override, author="ui")
                _persist_review_bundle(updated, inspect_state, clip_dir)
                st.rerun()

    with st.expander("🛠 Review edits", expanded=False):
        source_options = [source.source_id for source in bundle.physical_sources]
        if source_options:
            selected_source = st.selectbox("Rename source", source_options, key="rename_source_id")
            renamed_label = st.text_input("New source label", key="rename_source_label")
            if st.button("Apply source rename", key="apply_source_rename") and renamed_label.strip():
                updated = rename_source(bundle, source_id=selected_source, new_label=renamed_label.strip(), author="ui")
                _persist_review_bundle(updated, inspect_state, clip_dir)
                st.rerun()

        group_ids = [group.group_id for group in bundle.generation_groups]
        if group_ids:
            selected_group = st.selectbox("Split generation group", group_ids, key="split_group_id")
            selected_group_obj = next(group for group in bundle.generation_groups if group.group_id == selected_group)
            split_events = st.multiselect(
                "Events to split into a new group",
                selected_group_obj.member_event_ids,
                key="split_group_events",
            )
            if st.button("Apply split", key="apply_group_split") and split_events:
                updated = split_generation_group(bundle, group_id=selected_group, event_ids=split_events, author="ui")
                _persist_review_bundle(updated, inspect_state, clip_dir)
                st.rerun()

        if len(group_ids) >= 2:
            groups_to_merge = st.multiselect("Merge generation groups", group_ids, key="merge_group_ids")
            if st.button("Apply merge", key="apply_group_merge") and len(groups_to_merge) >= 2:
                updated = merge_generation_groups(bundle, source_group_ids=groups_to_merge, author="ui")
                _persist_review_bundle(updated, inspect_state, clip_dir)
                st.rerun()

    with st.expander("✅ Validation issues", expanded=False):
        for index, issue in enumerate(bundle.validation.issues):
            issue_id = issue.issue_id or f"issue-{index:04d}"
            reviewed = issue_id in bundle.validation.reviewed_issue_ids
            badge = "✅" if reviewed else "⚠️"
            st.markdown(
                f"- {badge} `{issue_id}` {issue.issue_type} ({issue.severity}) — {issue.message} | action={issue.recommended_action}"
            )
            if not reviewed and st.button(f"Approve {issue_id}", key=f"approve_issue_{issue_id}"):
                updated = approve_validation_issue(bundle, issue_id=issue_id, author="ui")
                _persist_review_bundle(updated, inspect_state, clip_dir)
                st.rerun()

    if bundle.review_metadata.applied_edits:
        with st.expander("📝 Persisted review edits", expanded=False):
            for edit in bundle.review_metadata.applied_edits:
                st.write(f"- {edit.action} target={edit.target_ids} rationale={edit.rationale}")


def _persist_review_bundle(
    bundle: MultitrackDescriptionBundle,
    inspect_state: InspectState,
    clip_dir: str,
) -> None:
    bundle_path = inspect_state.get("review_bundle_path") or st.session_state.get("review_bundle_path")
    if not bundle_path:
        bundle_path = str(Path(clip_dir) / "review_bundle.json")
    bundle.artifacts.review_bundle_path = bundle_path
    persist_bundle(bundle, bundle_path)
    st.session_state.multitrack_bundle = bundle
    st.session_state.review_bundle_path = bundle_path
    if inspect_state is not None:
        inspect_state["multitrack_bundle"] = bundle
