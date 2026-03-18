from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import os
import time
import json


# --- Pydantic Models ---

class TimeRange(BaseModel):
    start: float = Field(
        description="Start time in seconds with 0.1s precision (e.g., 1.3, 4.7, 12.1)"
    )
    end: float = Field(
        description="End time in seconds with 0.1s precision (e.g., 2.8, 6.4, 15.3)"
    )

class SceneObject(BaseModel):
    description: str = Field(
        description="Object description as a short text clip. Must include specific count (e.g. '2 people playing guitars', '1 dog running')"
    )
    time_range: TimeRange = Field(
        description="Time range when this object appears. Must be within the scene's time range."
    )
    group_id: Optional[str] = Field(default=None, description="Track group ID assigned post-analysis for temporal consistency")
    canonical_description: Optional[str] = Field(default=None, description="Unified canonical description for the group")


class Scene(BaseModel):
    scene_index: int = Field(description="0-based scene index")
    time_range: TimeRange = Field(description="Time range of this scene")
    background_sound: str = Field(
        description="Short text clip describing appropriate background sound/music for this scene. Include extra objects beyond the main 2 as part of background sound."
    )
    objects: list[SceneObject] = Field(
        description="Main objects in the scene. Maximum 2. Each must include specific counts. If more than 2 notable objects exist, describe extras in background_sound.",
        max_length=2,
    )
    background_group_id: Optional[str] = Field(default=None, description="Group ID for background track assigned post-analysis")
    background_canonical: Optional[str] = Field(default=None, description="Unified canonical description for the background group")


class VideoSceneAnalysis(BaseModel):
    total_duration: float = Field(description="Total video duration in seconds")
    scenes: list[Scene] = Field(description="List of scenes detected in the video")


# --- Main ---

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt from the prompts/ directory."""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


PROMPT = _load_prompt("scene_analysis_default.txt")

EXTENDED_PROMPT = _load_prompt("scene_analysis_extended.txt")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze video scenes using Gemini API")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--prompt_type", type=str, default="extended", choices=["default", "extended"])
    args = parser.parse_args()

    start_time = time.time()
    load_dotenv(".env.secure")
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables. Please set it in .env.secure")
    client = genai.Client(api_key=api_key)

    # Upload video
    file_path = args.video_path
    print(f"Uploading {file_path}...")
    video_file = client.files.upload(file=file_path)

    while video_file.state == "PROCESSING":
        print("Processing...", end="\r")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    print("Upload complete. Analyzing scenes...")

    # video_metadata with fps goes on the Part, not on GenerateContentConfig
    video_part = types.Part(
        file_data=types.FileData(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
        ),
        video_metadata=types.VideoMetadata(fps=2.0),
    )

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[video_part, EXTENDED_PROMPT if args.prompt_type == "extended" else PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoSceneAnalysis,
        ),
    )

    # Parse and save
    result = VideoSceneAnalysis.model_validate_json(response.text)
    output_path = "results/scene_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds with 2 seconds sleep intervals")


if __name__ == "__main__":
    main()
