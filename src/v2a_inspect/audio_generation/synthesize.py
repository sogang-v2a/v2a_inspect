"""
V2A Inspect → V2A Audio Synthesis and Video Mix CLI.

Takes a SoundTimeline (timeline.json) and a video file, generates audio
using LLM TTS or specialized models, and mixes it into an output video.

Usage:
  uv run v2a-synthesize --timeline timeline.json --video videos/my_video.mp4 --output output.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from v2a_inspect.models.sound_timeline import SoundTimeline
from v2a_inspect.audio_generation.client import generate_audio_for_item
from v2a_inspect.audio_generation.mix import mix_audio_into_video, AudioPlan, AudioPlanItem
from moviepy import VideoFileClip

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="v2a-synthesize",
        description="Generates audio from a SoundTimeline and mixes it into a video.",
    )
    parser.add_argument(
        "--timeline",
        required=False,
        help="Path to the timeline.json file containing the SoundTimeline",
    )
    parser.add_argument(
        "--asset",
        required=False,
        help="Path to the video-asset.json file containing the VideoAsset (Overrides --timeline)",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the original video file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output video path (default: same directory as video)",
    )
    parser.add_argument(
        "--keep-original-audio",
        action="store_true",
        default=False,
        help="Keep the original audio of the video (default: replace)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Detailed logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.timeline and not args.asset:
        print("Error: Must provide either --timeline or --asset", file=sys.stderr)
        return 1

    if args.asset:
        asset_path = Path(args.asset)
        if not asset_path.exists():
            print(f"Error: asset json not found: {asset_path}", file=sys.stderr)
            return 1
        print(f"[1/4] Loading VideoAsset: {asset_path}", file=sys.stderr)
        from v2a_inspect.models import VideoAsset
        asset_data = json.loads(asset_path.read_text(encoding="utf-8"))
        video_asset = VideoAsset(**asset_data)
        timeline = video_asset.sound_timeline
        if not timeline:
            print("Error: VideoAsset does not contain a sound_timeline", file=sys.stderr)
            return 1
        
    else:
        timeline_path = Path(args.timeline)
        if not timeline_path.exists():
            print(f"Error: timeline.json not found: {timeline_path}", file=sys.stderr)
            return 1

        print(f"[1/4] Loading timeline: {timeline_path}", file=sys.stderr)
        timeline_data = json.loads(timeline_path.read_text(encoding="utf-8"))
        timeline = SoundTimeline(**timeline_data)

    video_path = args.video
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 1
        
    print("[2/4] Converting SoundTimeline to AudioPlan...", file=sys.stderr)
    try:
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_duration = video_clip.duration
        video_clip.close()
    except Exception as e:
        print(f"Error: Could not read video file: {e}", file=sys.stderr)
        return 1
        
    if not fps or fps <= 0:
        fps = 30.0
        
    audio_plan = AudioPlan(total_duration=video_duration)
    
    # Map track_id to SoundTrack for easy lookup
    track_map = {track.sound_track_id: track for track in timeline.sound_tracks}
    
    for event in timeline.sound_events:
        track = track_map.get(event.sound_track_id)
        if not track:
            continue
            
        start_time = event.start_frame_index / fps
        end_time = event.end_frame_index / fps
        
        # Adjust end_time if it's less than or equal to start_time
        if end_time <= start_time:
            end_time = start_time + 0.1
            
        # Map generation mode (vta -> v2a, tta -> t2a)
        gen_mode = track.generation_mode
        if gen_mode == "vta":
            gen_model = "v2a"
        elif gen_mode == "tta":
            gen_model = "t2a"
        else:
            gen_model = gen_mode

        desc = f"[{track.label}] {event.description}"

        item = AudioPlanItem(
            item_id=str(event.sound_event_id),
            type=track.track_type,
            time=(start_time, end_time),
            description=desc,
            volume=0.8,
            track_id=str(track.sound_track_id),
            generation_model=gen_model
        )
        audio_plan.items.append(item)
        
    audio_plan.items.sort(key=lambda x: x.time[0])
    
    n_items = len(audio_plan.items)
    print(f"  → {n_items} audio events scheduled.", file=sys.stderr)
    
    if n_items == 0:
        print("Warning: No audio items to generate.", file=sys.stderr)
        return 0

    print(f"[3/4] Generating {n_items} audio tracks...", file=sys.stderr)
    audio_dir = Path(tempfile.mkdtemp(prefix="v2a_synth_audio_"))
    generated_audio: dict[str, str] = {}

    for i, item in enumerate(audio_plan.items, 1):
        duration = item.time[1] - item.time[0]
        out_path = str(audio_dir / f"{item.item_id}.wav")

        desc_preview = item.description[:60] + ("..." if len(item.description) > 60 else "")
        print(
            f"  [{i}/{n_items}] {item.item_id} ({item.type}, "
            f"{item.time[0]:.1f}s-{item.time[1]:.1f}s): {desc_preview}",
            file=sys.stderr,
        )

        result_path = generate_audio_for_item(
            kind=item.type,
            description=item.description,
            out_path=out_path,
            duration=duration,
            video_path=video_path,
            time=item.time,
            generation_model=item.generation_model,
        )
        if result_path:
            generated_audio[item.item_id] = result_path

    n_generated = len(generated_audio)
    print(f"  → Generated {n_generated}/{n_items} audio files.", file=sys.stderr)

    if n_generated == 0:
        print("Error: No audio files were generated.", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(video_path).parent / f"synthesized_{timestamp}.mp4")

    print(f"[4/4] Mixing audio into video → {output_path}", file=sys.stderr)
    result = mix_audio_into_video(
        video_path=video_path,
        audio_plan=audio_plan,
        generated_audio=generated_audio,
        output_path=output_path,
        keep_original_audio=args.keep_original_audio,
    )

    if result:
        print(f"\n✅ Synthesis complete: {Path(result).resolve()}", file=sys.stderr)
        return 0
    else:
        print("\n❌ Synthesis failed.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
