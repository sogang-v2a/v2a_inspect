"""
V2A Inspect — 트랙 그루핑 검증 시스템
Gemini 장면 분석 + 크로스씬 트랙 그루핑 결과 시각화.
오디오 생성 없음 (TangoFlux 미로드).
"""
import os
import sys
import glob
import shutil
import tempfile
import time
import threading
from pathlib import Path

import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
import streamlit as st
import streamlit_authenticator as stauth

# Pipeline imports (local pipeline/ directory)
_pipeline_dir = str(Path(__file__).parent / "pipeline")
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

_inspect_dir = str(Path(__file__).parent)
if _inspect_dir not in sys.path:
    sys.path.insert(0, _inspect_dir)

from pipeline_client import InspectPipeline                         # noqa: E402
from track_grouper import group_tracks, GroupedAnalysis, RawTrack, ModelSelection   # noqa: E402
from analyze_video_scenes import VideoSceneAnalysis                  # noqa: E402

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="V2A Inspect",
    page_icon="🔍",
    layout="wide",
)

# =============================================================================
# Authentication
# =============================================================================
load_dotenv(Path(__file__).parent / ".env.secure")

_credentials_path = Path(__file__).parent / "credentials.yaml"
with open(_credentials_path) as _f:
    _auth_config = yaml.load(_f, Loader=SafeLoader)

_cookie_key = os.getenv("AUTH_COOKIE_KEY")
if not _cookie_key:
    st.error("AUTH_COOKIE_KEY가 .env.secure에 설정되지 않았습니다.")
    st.stop()

authenticator = stauth.Authenticate(
    _auth_config["credentials"],
    "v2a_inspect_cookie",
    _cookie_key,
    1,
)
authenticator.login()

if st.session_state["authentication_status"] is False:
    st.error("Username 또는 Password가 올바르지 않습니다.")
    st.stop()
elif st.session_state["authentication_status"] is None:
    st.warning("Username과 Password를 입력해주세요.")
    st.stop()

# =============================================================================
# Utility functions
# =============================================================================

def save_uploaded_file(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temp directory and return its path."""
    temp_dir = tempfile.mkdtemp(prefix="v2a_inspect_upload_")
    safe_name = "".join(
        c for c in Path(uploaded_file.name).name if c.isalnum() or c in "._-"
    )
    if not safe_name:
        safe_name = "video.mp4"
    file_path = os.path.join(temp_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_clip(video_path: str, start: float, end: float, clip_dir: str) -> str | None:
    """
    Extract a subclip [start, end] from video_path using moviepy.
    Result is cached by (start, end) inside clip_dir.
    Returns path on success, None on failure.
    """
    out_path = os.path.join(clip_dir, f"clip_{start:.3f}_{end:.3f}.mp4")
    if os.path.exists(out_path):
        return out_path
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(video_path) as src:
            duration = src.duration
            actual_end = min(end, duration)
            actual_start = max(0.0, start)
            if actual_start >= actual_end:
                return None
            sub = src.subclipped(actual_start, actual_end)
            sub.write_videofile(out_path, logger=None, audio=False)
        return out_path
    except Exception:
        return None


def validate_video_file(path: str) -> bool:
    """Check that file starts with valid video container magic bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        return b"ftyp" in header or header[:4] == b"RIFF" or header[:4] == b"\x1a\x45\xdf\xa3"
    except OSError:
        return False


def get_video_duration(video_path: str) -> float | None:
    """Return video duration in seconds, or None on failure."""
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(video_path) as clip:
            return clip.duration
    except Exception:
        return None


def reset_state():
    """Clear all session state and remove temp directories."""
    clip_dir = st.session_state.get("clip_dir")
    if clip_dir and os.path.isdir(clip_dir):
        shutil.rmtree(clip_dir, ignore_errors=True)
    upload_path = st.session_state.get("video_path")
    if upload_path:
        upload_dir = os.path.dirname(upload_path)
        if "v2a_inspect_upload_" in upload_dir:
            shutil.rmtree(upload_dir, ignore_errors=True)
    for key in ("video_path", "scene_analysis", "grouped", "clip_dir"):
        st.session_state[key] = None
    st.session_state["model_overrides"] = {}


# =============================================================================
# Process-level shared resources (stateless, no user data)
# =============================================================================

@st.cache_resource
def _get_gemini_client():
    """Stateless HTTP client — one per process, holds no user data."""
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in .env.secure")
    import google.genai as genai
    return genai.Client(api_key=api_key)


@st.cache_resource
def _get_analysis_semaphore():
    """Limit concurrent Gemini analyses to 2."""
    return threading.Semaphore(2)


def _cleanup_stale_temp(max_age_seconds: int = 3600):
    """Remove v2a_inspect temp dirs older than max_age_seconds."""
    now = time.time()
    tmp_base = tempfile.gettempdir()
    for prefix in ["v2a_inspect_upload_", "v2a_inspect_clips_"]:
        for d in glob.glob(os.path.join(tmp_base, prefix + "*")):
            try:
                if os.path.isdir(d) and (now - os.path.getmtime(d)) > max_age_seconds:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


@st.cache_resource
def _start_cleanup_thread():
    """Daemon thread: clean stale temp dirs every 30 minutes."""
    def _loop():
        while True:
            time.sleep(1800)
            _cleanup_stale_temp()
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


_cleanup_stale_temp()
_start_cleanup_thread()

# =============================================================================
# Session state initialisation
# =============================================================================
for _key, _default in [
    ("video_path", None),
    ("scene_analysis", None),
    ("grouped", None),
    ("clip_dir", None),
    ("model_overrides", {}),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# =============================================================================
# Title
# =============================================================================
st.title("🔍 V2A Inspect — 트랙 그루핑 검증 시스템")
st.markdown(
    "Gemini 장면 분석과 크로스씬 트랙 그루핑 결과를 시각화하여 "
    "**사람이 직접 검증**할 수 있는 검사 도구입니다.  \n"
    "오디오 생성 없음 — 분석과 그루핑 단계만 실행합니다."
)
st.divider()

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("⚙️ 분석 옵션")

    fps = st.slider("Analysis FPS", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    st.caption("초당 분석 프레임 수. 높을수록 정밀하지만 느림")

    prompt_type = st.selectbox("Prompt Type", ["default", "extended"], index=0)
    st.caption("`default`: 간결 | `extended`: Foley 상세")

    enable_vlm_verify = st.checkbox("VLM 그룹 검증 사용", value=True)
    st.caption("Gemini VLM이 실제 영상 프레임으로 그룹핑 결과를 시각적으로 확인")

    enable_model_select = st.checkbox("TTA/VTA 모델 자동 선정", value=False)
    st.caption(
        "Gemini VLM이 각 씬의 동적 특성(싱크 중요도 vs 트랙 분리 중요도)을 분석하여 "
        "TTA 또는 VTA 모델을 자동 판정"
    )

    st.divider()

    with st.expander("🔄 파이프라인 구조", expanded=True):
        st.markdown(
            """
```
📹 Video Upload
      │
      ▼
🤖 Gemini Scene Analysis
   FPS · Prompt Type
      │
      ▼
 VideoSceneAnalysis
  ├─ Scene 0
  │   ├─ background_sound
  │   └─ objects (≤2)
  └─ Scene N ...
      │
      ▼
🔗 Cross-Scene Text Grouping
   (Gemini batch call)
      │
      ▼  (VLM verify ON)
👁️ VLM Group Verification
   (Gemini + video frames)
      │
      ▼
📦 GroupedAnalysis
  ├─ groups (canonical desc)
  └─ track_assignments
```
"""
        )

    st.divider()

    if st.button("🔄 Reset", use_container_width=True, type="secondary"):
        reset_state()
        st.rerun()

    authenticator.logout("Logout", "sidebar")

# =============================================================================
# Step 1: Upload & Analyze
# =============================================================================
st.header("Step 1: 영상 업로드 및 분석")

uploaded_file = st.file_uploader(
    "영상 파일 선택",
    type=["mp4", "mov", "avi", "mkv"],
    help="MP4, MOV, AVI, MKV 형식 지원",
)

if uploaded_file is not None:
    # Detect new video
    is_new = (
        st.session_state.video_path is None
        or not os.path.exists(st.session_state.video_path)
        or Path(st.session_state.video_path).name != uploaded_file.name
    )
    if is_new:
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

    analyze_disabled = st.session_state.grouped is not None
    if st.button(
        "🔍 Analyze & Group",
        type="primary",
        disabled=analyze_disabled,
        help="이미 분석된 경우 Reset 후 재실행하세요",
    ):
        video_path = st.session_state.video_path
        clip_dir = tempfile.mkdtemp(prefix="v2a_inspect_clips_")
        st.session_state.clip_dir = clip_dir

        sem = _get_analysis_semaphore()
        acquired = sem.acquire(timeout=120)
        if not acquired:
            st.error("서버가 바쁩니다. 잠시 후 다시 시도해주세요.")
            st.stop()
        try:
            with st.status("분석 진행 중...", expanded=True) as status:
                try:
                    st.write("Gemini 클라이언트 초기화 중...")
                    pipeline = InspectPipeline(
                        fps=fps, prompt_type=prompt_type, client=_get_gemini_client()
                    )

                    st.write(f"영상 업로드 및 장면 분석 중 (fps={fps}, prompt={prompt_type})...")
                    scene_analysis, video_file = pipeline.analyze_video(
                        video_path, return_file_object=True
                    )
                    st.session_state.scene_analysis = scene_analysis
                    st.write(
                        f"✅ 장면 분석 완료: {len(scene_analysis.scenes)}개 씬, "
                        f"{scene_analysis.total_duration:.1f}s"
                    )

                    steps = []
                    if enable_vlm_verify:
                        steps.append("VLM 검증")
                    if enable_model_select:
                        steps.append("TTA/VTA 선정")
                    step_str = " + ".join(steps) if steps else "텍스트 그루핑만"
                    st.write(f"트랙 그루핑 중 ({step_str})...")
                    grouped = group_tracks(
                        scene_analysis,
                        pipeline.client,
                        video_file=video_file,
                        enable_vlm_verify=enable_vlm_verify,
                        enable_model_select=enable_model_select,
                        progress_callback=st.write,
                    )
                    st.session_state.grouped = grouped
                    n_model_assigned = sum(
                        1 for t in grouped.raw_tracks if t.model_selection is not None
                    )
                    msg = f"✅ 그루핑 완료: {len(grouped.raw_tracks)}개 raw 트랙 → {len(grouped.groups)}개 그룹"
                    if n_model_assigned:
                        msg += f" | 모델 판정 {n_model_assigned}개 트랙"
                    st.write(msg)

                    status.update(label="분석 완료!", state="complete")
                    st.rerun()

                except TimeoutError:
                    status.update(label="Timeout", state="error")
                    st.error("영상 처리 시간이 초과되었습니다. 더 짧은 영상을 사용해주세요.")
                except Exception as exc:
                    status.update(label="분석 실패", state="error")
                    st.error(f"오류: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
        finally:
            sem.release()

    if analyze_disabled:
        st.success("✅ 분석 완료. 아래에서 결과를 확인하세요. (재분석하려면 Reset 클릭)")

# =============================================================================
# Step 2 & 3: Results (shown only after analysis)
# =============================================================================
if st.session_state.grouped is not None:
    grouped: GroupedAnalysis = st.session_state.grouped
    scene_analysis: VideoSceneAnalysis = st.session_state.scene_analysis
    video_path: str = st.session_state.video_path
    clip_dir: str = st.session_state.clip_dir or ""

    # ── Step 2: Stats + JSON ──────────────────────────────────────────────
    st.divider()
    st.header("Step 2: 분석 결과 요약")

    n_scenes = len(scene_analysis.scenes)
    n_backgrounds = n_scenes          # always 1 background per scene
    n_objects = sum(len(s.objects) for s in scene_analysis.scenes)
    n_raw = len(grouped.raw_tracks)
    n_groups = len(grouped.groups)
    n_multi = sum(1 for g in grouped.groups if len(g.member_ids) > 1)
    n_verified = sum(1 for g in grouped.groups if g.vlm_verified)
    n_merged = n_raw - n_groups       # how many tracks were merged
    n_model_vta = sum(1 for t in grouped.raw_tracks if t.model_selection and t.model_selection.model_type == "VTA")
    n_model_tta = sum(1 for t in grouped.raw_tracks if t.model_selection and t.model_selection.model_type == "TTA")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🎬 씬 수", n_scenes)
    c2.metric("🌲 배경 트랙", n_backgrounds)
    c3.metric("🎯 객체 트랙", n_objects)
    c4.metric("📦 Raw 트랙 수", n_raw)
    c5.metric(
        "🔗 최종 그룹 수",
        n_groups,
        delta=f"-{n_merged} 병합" if n_merged > 0 else "병합 없음",
        delta_color="normal" if n_merged > 0 else "off",
    )
    c6.metric("✅ VLM 검증 그룹", n_verified)

    if n_model_vta + n_model_tta > 0:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🟢 TTA 트랙", n_model_tta)
        mc2.metric("🔵 VTA 트랙", n_model_vta)
        mc3.metric(
            "⚠️ 그룹 내 이견",
            sum(1 for g in grouped.groups if g.model_selection and g.model_selection.confidence < 0.6),
        )

    st.caption(
        f"멀티멤버 그룹 {n_multi}개 (같은 개체로 판단된 트랙들이 하나의 그룹으로 묶임)"
    )

    # JSON expanders
    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("📋 씬 분석 JSON (raw)", expanded=False):
            st.json(scene_analysis.model_dump())

    with col_right:
        with st.expander("📋 트랙 그룹 JSON", expanded=False):
            st.json(
                {
                    "groups": {
                        g.group_id: {
                            "canonical_description": g.canonical_description,
                            "member_ids": g.member_ids,
                            "vlm_verified": g.vlm_verified,
                        }
                        for g in grouped.groups
                    },
                    "track_assignments": grouped.track_to_group,
                }
            )

    # ── Step 3: Grouping Verification ─────────────────────────────────────
    st.divider()
    st.header("Step 3: 그루핑 검증")
    st.caption(
        "각 그룹의 **canonical description**과 멤버별 원본 description·영상 클립을 나란히 배치합니다.  \n"
        "멀티멤버 그룹에서 같은 개체인지 직접 확인하세요."
    )

    tracks_by_id: dict[str, RawTrack] = {t.track_id: t for t in grouped.raw_tracks}

    for group in grouped.groups:
        members = [
            tracks_by_id[mid]
            for mid in group.member_ids
            if mid in tracks_by_id
        ]
        is_multi = len(members) > 1

        if group.vlm_verified:
            badge = "✅ VLM 검증됨"
        elif is_multi:
            badge = "🔗 텍스트 그루핑"
        else:
            badge = "⬜ 싱글턴"

        short_desc = (
            group.canonical_description[:60] + "..."
            if len(group.canonical_description) > 60
            else group.canonical_description
        )

        with st.expander(
            f"{badge}  `{group.group_id}` — {short_desc}",
            expanded=is_multi,
        ):
            # Group header
            hcol_a, hcol_b = st.columns([4, 1])
            with hcol_a:
                st.markdown(f"**Canonical description:**  \n> {group.canonical_description}")
            with hcol_b:
                st.markdown(f"**{badge}**")
                st.caption(f"멤버 {len(members)}개")
                # Model selection badge
                if group.model_selection:
                    gms = group.model_selection
                    model_icon = "🔵" if gms.model_type == "VTA" else "🟢"
                    conflict_flag = " ⚠️" if gms.confidence < 0.6 else ""
                    rule_tag = " ⚡규칙" if gms.rule_based else ""
                    st.markdown(
                        f"{model_icon} **{gms.model_type}**{conflict_flag}{rule_tag}  \n"
                        f"conf: {gms.confidence:.0%}  \n"
                        f"vta={gms.vta_score:.1f} / tta={gms.tta_score:.1f}"
                    )
                    if gms.confidence < 0.6:
                        st.caption("⚠️ 그룹 내 멤버 간 모델 이견 있음")
                # Manual override selectbox
                current_override = st.session_state.model_overrides.get(group.group_id, "(자동)")
                override = st.selectbox(
                    "모델 오버라이드",
                    ["(자동)", "TTA", "VTA"],
                    index=["(자동)", "TTA", "VTA"].index(current_override),
                    key=f"model_override_{group.group_id}",
                )
                if override != current_override:
                    st.session_state.model_overrides[group.group_id] = override

            st.markdown("---")

            if not is_multi:
                # ── Singleton ──
                track = members[0]
                kind_icon = "🌲" if track.kind == "background" else "🎯"
                st.markdown(
                    f"{kind_icon} `{track.track_id}` | "
                    f"Scene {track.scene_index} | "
                    f"{track.start:.1f}s – {track.end:.1f}s | "
                    f"*{track.kind}*"
                )
                st.info(track.description)
                if track.model_selection:
                    ms = track.model_selection
                    t_icon = "🔵" if ms.model_type == "VTA" else "🟢"
                    rule_tag = " ⚡규칙" if ms.rule_based else ""
                    st.caption(
                        f"{t_icon} **{ms.model_type}**{rule_tag} ({ms.confidence:.0%})  \n"
                        f"vta={ms.vta_score:.1f} / tta={ms.tta_score:.1f}  \n"
                        f"{ms.reasoning}"
                    )
                if video_path and clip_dir:
                    clip_path = extract_clip(video_path, track.start, track.end, clip_dir)
                    if clip_path:
                        st.video(clip_path)
                    else:
                        st.warning("영상 클립 추출 실패")

            else:
                # ── Multi-member: side by side ──
                max_cols = min(len(members), 4)  # cap at 4 columns
                cols = st.columns(max_cols)

                for i, track in enumerate(members):
                    col = cols[i % max_cols]
                    kind_icon = "🌲" if track.kind == "background" else "🎯"
                    with col:
                        st.markdown(f"#### {kind_icon} `{track.track_id}`")
                        st.caption(
                            f"Scene {track.scene_index} | "
                            f"{track.start:.1f}s – {track.end:.1f}s | "
                            f"*{track.kind}*"
                        )
                        # Original (pre-canonical) description highlighted
                        st.info(track.description)

                        # Per-track model selection badge
                        if track.model_selection:
                            ms = track.model_selection
                            t_icon = "🔵" if ms.model_type == "VTA" else "🟢"
                            rule_tag = " ⚡규칙" if ms.rule_based else ""
                            st.caption(
                                f"{t_icon} **{ms.model_type}**{rule_tag} ({ms.confidence:.0%})  \n"
                                f"vta={ms.vta_score:.1f} / tta={ms.tta_score:.1f}  \n"
                                f"{ms.reasoning}"
                            )

                        if video_path and clip_dir:
                            clip_path = extract_clip(
                                video_path, track.start, track.end, clip_dir
                            )
                            if clip_path:
                                st.video(clip_path)
                            else:
                                st.warning("클립 추출 실패")

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption(
    "V2A Inspect | Gemini Scene Analysis + Cross-Scene Track Grouping | No Audio Generation"
)
