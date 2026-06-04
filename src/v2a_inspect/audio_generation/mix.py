"""
오디오 믹싱 및 영상 합성 모듈.

AudioPlan 기반으로 생성된 오디오 파일들을 원본 비디오에 합성합니다.

주요 기능:
  - Volume 적용: 각 AudioPlanItem의 volume 값 반영
  - Pan 적용: 스테레오 패닝 (-1.0 좌 ~ 1.0 우)
  - 시간 정렬: 각 오디오를 정확한 타임라인 위치에 배치
"""

from __future__ import annotations

import logging
from pathlib import Path

from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from moviepy.audio.fx import MultiplyVolume, AudioFadeOut
from dataclasses import dataclass, field


@dataclass
class AudioPlanItem:
    item_id: str
    type: str
    time: tuple[float, float]
    description: str
    volume: float = 0.8
    intensity: float = 0.5
    pan: float = 0.0
    confidence: float = 1.0
    track_id: str | None = None
    generation_model: str = "t2a"


@dataclass
class AudioPlan:
    items: list[AudioPlanItem] = field(default_factory=list)
    total_duration: float = 0.0


logger = logging.getLogger(__name__)


def mix_audio_into_video(
    video_path: str,
    audio_plan: AudioPlan,
    generated_audio: dict[str, str],
    output_path: str,
    *,
    keep_original_audio: bool = False,
) -> str | None:
    """
    생성된 오디오 트랙들을 원본 비디오에 합성합니다.

    Parameters
    ----------
    video_path : str
        원본 비디오 파일 경로.
    audio_plan : AudioPlan
        오디오 플랜 (타이밍, 볼륨, 팬 정보).
    generated_audio : dict[str, str]
        item_id → 생성된 wav 파일 경로 매핑.
    output_path : str
        출력 비디오 파일 경로.
    keep_original_audio : bool
        True이면 원본 비디오의 오디오를 유지합니다.

    Returns
    -------
    str | None
        성공 시 출력 파일 경로, 실패 시 None.
    """
    if not generated_audio:
        logger.error("No generated audio to mix.")
        return None

    try:
        video = VideoFileClip(video_path)
        audio_clips: list = []

        # 원본 오디오 유지 옵션
        if keep_original_audio and video.audio is not None:
            audio_clips.append(video.audio)

        # 각 AudioPlanItem에 대해 오디오 클립 생성 및 배치
        n_mixed = 0
        for item in audio_plan.items:
            if item.type == "silence":
                continue

            wav_path = generated_audio.get(item.item_id)
            if not wav_path or not Path(wav_path).exists():
                continue

            duration = item.time[1] - item.time[0]
            clip = AudioFileClip(wav_path)
            clip = clip.subclipped(0, min(clip.duration, duration))
            clip = clip.with_start(item.time[0])

            effects = []
            # 1. Volume 적용
            if item.volume != 1.0:
                effects.append(MultiplyVolume(item.volume))

            # 자연스러운 페이드아웃 추가 (0.15초)
            fade_duration = min(0.15, clip.duration / 2)
            if fade_duration > 0:
                effects.append(AudioFadeOut(fade_duration))

            if effects:
                clip = clip.with_effects(effects)

            # 2. Stereo Pan 적용
            if item.pan != 0.0:
                clip = _apply_pan(clip, item.pan)

            audio_clips.append(clip)
            n_mixed += 1

        if not audio_clips:
            logger.warning("No audio clips to mix into video.")
            video.close()
            return None

        # 오디오 합성
        final_audio = CompositeAudioClip(audio_clips)
        if final_audio.duration > video.duration:
            final_audio = final_audio.subclipped(0, video.duration)

        video = video.with_audio(final_audio)

        # 비디오 출력
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing mixed video: %s (%d tracks)", output_path, n_mixed)
        video.write_videofile(
            output_path, audio_codec="aac", fps=video.fps, logger=None
        )

        # 리소스 정리
        video.close()
        for clip in audio_clips:
            if hasattr(clip, "close"):
                clip.close()

        return output_path

    except Exception as e:
        logger.error("Failed to mix video: %s", e)
        return None


# ── Pan 효과 ──────────────────────────────────────────────────────────────────


def _apply_pan(clip, pan: float):
    """
    Simple stereo pan: -1.0 = left only, 0.0 = center, 1.0 = right only.
    Uses channel volume weighting.
    """
    try:
        left_vol = max(0.0, 1.0 - pan)
        right_vol = max(0.0, 1.0 + pan)
        peak = max(left_vol, right_vol, 1e-6)
        left_vol /= peak
        right_vol /= peak

        def _pan_audio(get_frame, t):
            import numpy as np

            frame = get_frame(t)
            if frame.ndim == 1:
                frame = np.c_[frame, frame]  # mono → stereo
            frame = frame.copy().astype(float)
            frame[:, 0] *= left_vol
            frame[:, 1] *= right_vol
            return frame

        return clip.transform(_pan_audio, apply_to="audio")
    except Exception:
        return clip
