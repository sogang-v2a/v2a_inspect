from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import streamlit as st

from v2a_inspect.observability import WorkflowTraceContext
from v2a_inspect.review import persist_bundle
from v2a_inspect.runner import run_inspect
from v2a_inspect.settings import settings
from v2a_inspect.ui.auth import require_authentication
from v2a_inspect.ui.render import (
    render_footer,
    render_page_header,
    render_results,
    render_sidebar,
)
from v2a_inspect.ui.session import (
    ensure_process_resources,
    get_analysis_semaphore,
    get_langfuse_session_id,
    initialize_session_state,
    reset_state,
)
from v2a_inspect.ui.video import (
    get_video_duration,
    save_uploaded_file,
    validate_video_file,
)
from v2a_inspect.workflows import InspectOptions


def main() -> None:
    st.set_page_config(page_title="V2A Inspect", page_icon="🔍", layout="wide")

    authenticator = require_authentication()
    ensure_process_resources()
    initialize_session_state()

    render_page_header()
    options = render_sidebar(authenticator)
    render_upload_step(options)

    bundle = st.session_state.get("multitrack_bundle")
    if bundle is not None:
        render_results(
            video_path=st.session_state.get("video_path") or "",
            clip_dir=st.session_state.get("clip_dir") or "",
            inspect_state=st.session_state.get("inspect_state"),
        )

    render_footer()


def render_upload_step(options: InspectOptions) -> None:
    st.header("Step 1: 영상 업로드 및 분석")

    uploaded_file = st.file_uploader(
        "영상 파일 선택",
        type=["mp4", "mov", "avi", "mkv"],
        help="MP4, MOV, AVI, MKV 형식 지원",
    )

    if uploaded_file is None:
        return

    is_new_video = (
        st.session_state.video_path is None
        or not Path(st.session_state.video_path).exists()
        or Path(st.session_state.video_path).name != uploaded_file.name
    )
    if is_new_video:
        reset_state()
        st.session_state.video_path = save_uploaded_file(uploaded_file)

        if not validate_video_file(st.session_state.video_path):
            st.error("유효한 영상 파일이 아닙니다.")
            reset_state()
            st.stop()

        duration = get_video_duration(st.session_state.video_path)
        if duration is not None and duration > 60.0:
            st.error(f"영상 길이가 {duration:.1f}초입니다. 최대 60초까지 허용됩니다.")
            reset_state()
            st.stop()

    st.video(uploaded_file)

    analyze_disabled = st.session_state.multitrack_bundle is not None
    if st.button(
        "🔍 Analyze & Group",
        type="primary",
        disabled=analyze_disabled,
        help="이미 분석된 경우 Reset 후 재실행하세요",
    ):
        run_analysis(st.session_state.video_path, options)

    if analyze_disabled:
        st.success(
            "✅ 분석 완료. 아래에서 결과를 확인하세요. (재분석하려면 Reset 클릭)"
        )


def run_analysis(video_path: str, options: InspectOptions) -> None:
    clip_dir = tempfile.mkdtemp(prefix="v2a_inspect_clips_")
    st.session_state.clip_dir = clip_dir

    semaphore = get_analysis_semaphore()
    acquired = semaphore.acquire(timeout=settings.ui_analysis_acquire_timeout_seconds)
    if not acquired:
        st.error("서버가 바쁩니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    try:
        with st.status("분석 진행 중...", expanded=True) as status:
            try:
                state = run_inspect(
                    video_path,
                    options=options,
                    progress_callback=status.write,
                    warning_callback=lambda message: status.write(f"⚠️ {message}"),
                    trace_context=_build_ui_trace_context(options),
                )
                for message in state.get("progress_messages", []):
                    status.write(message)
                for message in state.get("warnings", []):
                    status.write(f"⚠️ {message}")
                st.session_state.inspect_state = state
                bundle = state.get("multitrack_bundle")
                if bundle is not None:
                    bundle_path = Path(clip_dir) / "review_bundle.json"
                    bundle.artifacts.review_bundle_path = str(bundle_path)
                    persist_bundle(bundle, bundle_path)
                    st.session_state.multitrack_bundle = bundle
                    st.session_state.review_bundle_path = str(bundle_path)
                if bundle is None:
                    raise ValueError(
                        "Inspect workflow completed without a multitrack bundle."
                    )

                message = (
                    f"✅ 번들 생성 완료: "
                    f"{len(bundle.physical_sources)}개 source / "
                    f"{len(bundle.sound_events)}개 event / "
                    f"{len(bundle.generation_groups)}개 generation group"
                )

                status.write(message)
                status.update(label="분석 완료!", state="complete")
                st.rerun()
            except TimeoutError:
                status.update(label="Timeout", state="error")
                st.error(
                    "영상 처리 시간이 초과되었습니다. 더 짧은 영상을 사용해주세요."
                )
            except Exception as exc:  # noqa: BLE001
                status.update(label="분석 실패", state="error")
                st.error(f"오류: {exc}")
                st.code(traceback.format_exc())
    finally:
        semaphore.release()


def _build_ui_trace_context(options: InspectOptions) -> WorkflowTraceContext:
    username = st.session_state.get("username")
    return WorkflowTraceContext(
        source="ui",
        operation="analyze",
        user_id=str(username) if username else None,
        session_id=get_langfuse_session_id(),
        tags=(),
        metadata={
            "pipeline_mode": options.pipeline_mode,
            "fps": options.fps,
            "auth_mode": settings.auth_mode,
        },
    )


if __name__ == "__main__":
    main()
