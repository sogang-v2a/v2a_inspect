"""
Lightweight Gemini-only pipeline client for the Inspect UI.
Does NOT load TangoFlux — analysis and grouping only.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# Make the local pipeline directory importable
_pipeline_dir = str(Path(__file__).parent / "pipeline")
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

from analyze_video_scenes import VideoSceneAnalysis, PROMPT, EXTENDED_PROMPT  # noqa: E402


def _state_name(state) -> str:
    """Normalize file state to string regardless of SDK version (str or enum)."""
    if hasattr(state, "name"):
        return state.name
    return str(state)


class InspectPipeline:
    """Gemini-only wrapper: scene analysis + track grouping, no audio generation."""

    MODEL = "gemini-3-pro-preview"

    def __init__(self, fps: float = 2.0, prompt_type: str = "default", client=None):
        self.fps = fps
        self.prompt_type = prompt_type
        self.client = client if client is not None else self._load_client()

    def _load_client(self) -> genai.Client:
        env_path = Path(__file__).parent / ".env.secure"
        load_dotenv(str(env_path))
        api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API_KEY not found. Add it to .env.secure in the v2a_ui root directory."
            )
        return genai.Client(api_key=api_key)

    def get_prompt(self) -> str:
        return EXTENDED_PROMPT if self.prompt_type == "extended" else PROMPT

    def analyze_video(
        self,
        video_path: str,
        fps: float | None = None,
        prompt: str | None = None,
        max_wait_seconds: int = 300,
        return_file_object: bool = False,
    ):
        """
        Upload video to Gemini, wait for processing, run structured scene analysis.

        Returns:
            VideoSceneAnalysis  (or (VideoSceneAnalysis, video_file) if return_file_object=True)
        """
        if fps is None:
            fps = self.fps
        if prompt is None:
            prompt = self.get_prompt()

        # --- Upload ---
        print(f"[inspect] Uploading: {video_path}")
        video_file = self.client.files.upload(file=video_path)

        # --- Poll until ACTIVE ---
        waited = 0
        while _state_name(video_file.state) == "PROCESSING":
            if waited >= max_wait_seconds:
                raise TimeoutError(
                    f"Video processing timed out after {max_wait_seconds}s"
                )
            time.sleep(2)
            waited += 2
            video_file = self.client.files.get(name=video_file.name)

        if _state_name(video_file.state) != "ACTIVE":
            raise RuntimeError(
                f"Video upload ended in unexpected state: {video_file.state}"
            )

        print(f"[inspect] Video ready ({waited}s). Analyzing scenes (fps={fps})...")

        # --- Generate content ---
        video_part = types.Part(
            file_data=types.FileData(file_uri=video_file.uri),
            video_metadata=types.VideoMetadata(fps=fps),
        )
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=180_000),  # 3 min
                response_mime_type="application/json",
                response_schema=VideoSceneAnalysis,
            ),
        )

        scene_analysis = VideoSceneAnalysis.model_validate_json(response.text)
        print(
            f"[inspect] Done: {len(scene_analysis.scenes)} scenes, "
            f"{scene_analysis.total_duration:.1f}s total"
        )

        if return_file_object:
            return scene_analysis, video_file
        return scene_analysis
