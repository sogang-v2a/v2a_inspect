from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2a_inspect.tools.types import (
    FrameBatch,
    SampledFrame,
    Sam3EntityTrack,
    Sam3TrackPoint,
)
from v2a_inspect_server.tool_registry import build_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_direct_tool_surface(self) -> None:
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda frame_batches, prompts_by_scene=None: {
                    "tracks": []
                },
                recover_with_text_prompt=lambda frame_batches, text_prompt: {
                    "tracks": []
                },
            ),
            embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
            label_client=SimpleNamespace(
                score_image_labels=lambda image_paths, labels: []
            ),
        )
        registry = build_tool_registry(fake_runtime)
        self.assertIn("structural_overview", registry)
        self.assertIn("densify_window_sampling", registry)
        self.assertIn("refine_candidate_cuts", registry)
        self.assertIn("recover_foreground_sources", registry)
        self.assertIn("recover_with_text_prompt", registry)
        self.assertIn("validate_bundle", registry)

    def test_crop_and_embedding_tools_are_callable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.jpg"
            from PIL import Image

            Image.new("RGB", (64, 64), color="white").save(frame_path)
            fake_runtime = SimpleNamespace(
                sam3_client=SimpleNamespace(
                    extract_entities=lambda frame_batches, prompts_by_scene=None: {
                        "tracks": []
                    },
                    recover_with_text_prompt=lambda frame_batches, text_prompt: {
                        "tracks": []
                    },
                ),
                embedding_client=SimpleNamespace(
                    embed_images=lambda track_image_paths: track_image_paths
                ),
                label_client=SimpleNamespace(
                    score_image_labels=lambda image_paths, labels: []
                ),
            )
            registry = build_tool_registry(fake_runtime)
            frame_batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(
                            scene_index=0,
                            timestamp_seconds=0.0,
                            image_path=str(frame_path),
                        )
                    ],
                )
            ]
            tracks = [
                Sam3EntityTrack(
                    track_id="trk0",
                    scene_index=0,
                    start_seconds=0.0,
                    end_seconds=0.0,
                    confidence=0.9,
                    points=[
                        Sam3TrackPoint(timestamp_seconds=0.0, bbox_xyxy=[0, 0, 10, 10])
                    ],
                )
            ]
            track_crops = registry["crop_tracks"].handler(
                frame_batches=frame_batches,
                tracks=tracks,
                output_dir=str(Path(tmp_dir) / "crops"),
            )
            self.assertEqual(len(track_crops), 1)
            embedded = registry["embed_track_crops"].handler(
                track_image_paths={"trk0": [track_crops[0].crop_path]}
            )
            self.assertEqual(embedded, {"trk0": [track_crops[0].crop_path]})

    def test_recover_foreground_sources_uses_scene_prompt_recovery_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.jpg"
            from PIL import Image

            Image.new("RGB", (64, 64), color="white").save(frame_path)
            fake_runtime = SimpleNamespace(
                runtime_profile="cpu_dev",
                should_release_clients=False,
                release_client=lambda name: None,
                sam3_client=SimpleNamespace(
                    extract_entities=lambda frame_batches, prompts_by_scene=None, score_threshold=0.35: SimpleNamespace(
                        strategy="scene_prompt_seeded",
                        tracks=[
                            SimpleNamespace(
                                track_id="trk0",
                                scene_index=0,
                                start_seconds=0.0,
                                end_seconds=0.0,
                                confidence=0.9,
                                label_hint=(prompts_by_scene or {0: ["object"]})[0][0],
                                points=[],
                            )
                        ],
                        model_copy=lambda update=None, **kwargs: None,
                    )
                ),
                embedding_client=SimpleNamespace(
                    embed_images=lambda track_image_paths: track_image_paths
                ),
                label_client=SimpleNamespace(
                    score_image_labels=lambda image_paths, labels: [
                        SimpleNamespace(label=labels[0], score=0.9),
                        SimpleNamespace(label=labels[1], score=0.8),
                    ]
                ),
            )
            registry = build_tool_registry(fake_runtime)
            frame_batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(
                            scene_index=0,
                            timestamp_seconds=0.0,
                            image_path=str(frame_path),
                        )
                    ],
                )
            ]
            recovered = registry["recover_foreground_sources"].handler(
                frame_batches=frame_batches,
                prompt_vocabulary=["vehicle", "person", "object"],
            )
            self.assertEqual(recovered["prompts_by_scene"][0][0], "vehicle")
            self.assertEqual(recovered["track_set"].strategy, "scene_prompt_recovery")

    def test_extract_entities_uses_scene_prompt_candidates_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.jpg"
            from PIL import Image

            Image.new("RGB", (64, 64), color="white").save(frame_path)
            captured: dict[str, object] = {}

            fake_runtime = SimpleNamespace(
                runtime_profile="mig10_safe",
                should_release_clients=True,
                release_client=lambda name: captured.setdefault("released", []).append(name),
                sam3_client=SimpleNamespace(
                    extract_entities=lambda frame_batches, prompts_by_scene=None, score_threshold=0.35, **kwargs: captured.update(
                        {
                            "prompts_by_scene": prompts_by_scene,
                            "score_threshold": score_threshold,
                            "extra_kwargs": kwargs,
                        }
                    )
                    or SimpleNamespace(strategy="scene_prompt_seeded", tracks=[])
                ),
                embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: track_image_paths),
                label_client=SimpleNamespace(
                    score_image_labels=lambda image_paths, labels: [
                        SimpleNamespace(label="cat", score=0.11),
                        SimpleNamespace(label="animal", score=0.09),
                    ]
                ),
            )
            registry = build_tool_registry(fake_runtime)
            frame_batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(
                            scene_index=0,
                            timestamp_seconds=0.0,
                            image_path=str(frame_path),
                        )
                    ],
                )
            ]
            registry["extract_entities"].handler(frame_batches=frame_batches)

        self.assertEqual(captured["prompts_by_scene"], {0: ["cat", "animal", "object"]})
        self.assertEqual(captured["score_threshold"], 0.25)
        self.assertEqual(captured["released"], ["label"])

    def test_extract_entities_keeps_label_client_resident_in_full_gpu_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.jpg"
            from PIL import Image

            Image.new("RGB", (64, 64), color="white").save(frame_path)
            captured: dict[str, object] = {}

            fake_runtime = SimpleNamespace(
                runtime_profile="full_gpu",
                should_release_clients=False,
                release_client=lambda name: captured.setdefault("released", []).append(name),
                sam3_client=SimpleNamespace(
                    extract_entities=lambda frame_batches, prompts_by_scene=None, score_threshold=0.35, **kwargs: captured.update(
                        {
                            "prompts_by_scene": prompts_by_scene,
                            "score_threshold": score_threshold,
                        }
                    )
                    or SimpleNamespace(strategy="scene_prompt_seeded", tracks=[])
                ),
                embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: track_image_paths),
                label_client=SimpleNamespace(
                    score_image_labels=lambda image_paths, labels: [
                        SimpleNamespace(label="cat", score=0.11),
                        SimpleNamespace(label="animal", score=0.09),
                    ]
                ),
            )
            registry = build_tool_registry(fake_runtime)
            frame_batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(
                            scene_index=0,
                            timestamp_seconds=0.0,
                            image_path=str(frame_path),
                        )
                    ],
                )
            ]
            registry["extract_entities"].handler(frame_batches=frame_batches)

        self.assertEqual(captured["prompts_by_scene"], {0: ["cat", "animal", "object"]})
        self.assertIsNone(captured.get("released"))


if __name__ == "__main__":
    unittest.main()
