"""
오디오 생성 클라이언트.

- OpenAI TTS: 대사(dialogue) 오디오 생성
- ElevenLabs: SFX / Ambience / Music 오디오 생성
- Dummy: API 키 없을 때 사일런트 fallback

환경 변수:
  OPENAI_API_KEY — OpenAI TTS 호출에 필요
  ELEVENLABS_API_KEY — ElevenLabs 호출에 필요
"""

from __future__ import annotations

import logging
import os
import re
import tempfile

import numpy as np
import scipy.io.wavfile as wavfile
import requests

logger = logging.getLogger(__name__)


# ── OpenAI TTS ────────────────────────────────────────────────────────────────


def generate_dialogue_openai(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate speech using OpenAI TTS API with adaptive speed and voice."""
    import openai

    match = re.search(r'["\'](.+?)["\']', text)
    spoken_text = match.group(1) if match else text

    # 화자 설명 키워드에 따라 음성 선택
    desc_lower = text.lower()
    if any(
        w in desc_lower
        for w in [
            "female",
            "woman",
            "girl",
            "lady",
            "여성",
            "여자",
            "소녀",
        ]
    ):
        voice = "nova"
    elif any(
        w in desc_lower
        for w in [
            "deep",
            "monster",
            "large man",
            "giant",
            "괴물",
            "거인",
            "거친",
        ]
    ):
        voice = "onyx"
    elif any(
        w in desc_lower
        for w in [
            "male",
            "man",
            "boy",
            "guy",
            "남성",
            "남자",
            "소년",
        ]
    ):
        voice = "echo"
    else:
        voice = "alloy"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. Falling back to dummy audio.")
        return generate_dummy_audio(duration or 1.0, out_path)

    client = openai.OpenAI(api_key=api_key)

    # 발화 속도 자동 조절: 주어진 duration에 맞춤
    speed = 1.0
    if duration and duration > 0.1:
        standard_duration = max(len(spoken_text) / 12.0, 0.5)
        raw_speed = standard_duration / duration
        speed = max(0.25, min(raw_speed, 4.0))

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=spoken_text,
        speed=speed,
    ) as response:
        response.stream_to_file(out_path)
    return out_path


# ── ElevenLabs SFX ────────────────────────────────────────────────────────────


def generate_sfx_elevenlabs(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate sound effects using ElevenLabs API."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not found. Falling back to dummy audio.")
        return generate_dummy_audio(duration or 1.0, out_path)

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=api_key)
        dur_seconds = min(max(duration, 0.5), 30.0) if duration else None

        audio_generator = client.text_to_sound_effects.convert(
            text=text,
            duration_seconds=dur_seconds,
        )

        with open(out_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        return out_path
    except Exception as e:
        logger.error("ElevenLabs SFX generation failed: %s", e)
        return generate_dummy_audio(duration or 1.0, out_path)


# ── ElevenLabs Music ──────────────────────────────────────────────────────────


def generate_music_elevenlabs(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate background music using ElevenLabs API."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not found. Falling back to dummy audio.")
        return generate_dummy_audio(duration or 1.0, out_path)

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=api_key)
        dur_ms = int(min(max(duration, 3.0), 30.0) * 1000) if duration else 10000

        audio_generator = client.music.compose(
            prompt=text,
            music_length_ms=dur_ms,
        )

        with open(out_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        return out_path
    except Exception as e:
        if "paid_plan_required" in str(e) or "402" in str(e):
            logger.error(
                "ElevenLabs Music API requires a paid plan. Falling back to dummy audio."
            )
        else:
            logger.error("ElevenLabs Music generation failed: %s", e)
        return generate_dummy_audio(duration or 1.0, out_path)


# ── Dummy (fallback) ──────────────────────────────────────────────────────────


def generate_dummy_audio(
    duration_sec: float, out_path: str, sample_rate: int = 44100
) -> str:
    """Generate a placeholder silent/beep audio file (fallback)."""
    if duration_sec <= 0:
        duration_sec = 0.1

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    fade_len = int(0.1 * sample_rate)
    if audio.size > fade_len:
        audio[fade_len:] = 0

    wavfile.write(out_path, sample_rate, (audio * 32767).astype(np.int16))
    return out_path


# ── V2A MMAudio API ───────────────────────────────────────────────────────────


def generate_v2a_hunyuan(
    video_id: str,
    fps: float,
    time: tuple[float, float],
    text: str,
    out_path: str,
    duration: float | None = None,
) -> str:
    """Generate audio using HunyuanVideo-Foley via remote API."""
    api_url = os.getenv("V2A_INSPECT_SERVER_URL", "http://localhost:8000")
    # If the user explicitly set the old variable, use it.
    if os.getenv("HUNYUAN_V2A_API_URL"):
        api_url = os.getenv("HUNYUAN_V2A_API_URL")

    try:
        data = {
            "video_id": video_id,
            "prompt": text,
            "start_frame_index": int(time[0] * fps),
            "end_frame_index": int(time[1] * fps),
            "guidance_scale": 4.5,
            "num_inference_steps": 50,
        }
        logger.info("Calling Hunyuan V2A API: %s", api_url)

        endpoint = f"{api_url}/infer/hunyuan/generate-v2a"
        if api_url.endswith("/generate_v2a"):
            # They might have put the old endpoint directly
            endpoint = api_url
            # Fallback to the old logic if they explicitly wanted the old API that accepts files.
            # But we don't have the file here. This will probably fail if it's the old API.
            # We assume it's the new API.

        response = requests.post(endpoint, json=data)

        if response.status_code == 200:
            with open(out_path, "wb") as f_out:
                f_out.write(response.content)

            return out_path
        else:
            logger.error(
                "Hunyuan API failed with %d: %s", response.status_code, response.text
            )
            return generate_dummy_audio(duration or 1.0, out_path)
    except Exception as e:
        logger.error("Failed to call Hunyuan API: %s", e)
        return generate_dummy_audio(duration or 1.0, out_path)


# ── Router ────────────────────────────────────────────────────────────────────


def generate_audio_for_item(
    kind: str,
    description: str,
    out_path: str,
    duration: float,
    video_path: str | None = None,
    time: tuple[float, float] | None = None,
    generation_model: str = "t2a",
    video_id: str | None = None,
    fps: float = 30.0,
) -> str | None:
    """
    오디오 타입에 따라 적절한 생성 API로 라우팅합니다.

    Parameters
    ----------
    kind : str
        "sfx", "ambience", "music", "dialogue" 중 하나
    description : str
        오디오 생성을 위한 텍스트 프롬프트
    out_path : str
        생성된 오디오 파일 저장 경로
    duration : float
        대상 오디오 길이(초)
    video_id : str | None
        대상 비디오 ID (V2A 용)
    fps : float
        대상 비디오의 초당 프레임 수 (V2A 용)

    Returns
    -------
    str | None
        생성된 파일 경로 또는 실패 시 None
    """
    try:
        if generation_model == "v2a" and video_id and time:
            return generate_v2a_hunyuan(
                video_id, fps, time, description, out_path, duration
            )

        if kind == "dialogue":
            if '"' in description or "'" in description:
                return generate_dialogue_openai(
                    description, out_path, duration=duration
                )
            else:
                return generate_sfx_elevenlabs(description, out_path, duration=duration)
        elif kind in ("sfx", "ambience"):
            return generate_sfx_elevenlabs(description, out_path, duration=duration)
        elif kind == "music":
            return generate_music_elevenlabs(description, out_path, duration=duration)
        else:
            return generate_dummy_audio(duration, out_path)
    except Exception as e:
        logger.error("Audio generation failed for '%s': %s", kind, e)
        return None
