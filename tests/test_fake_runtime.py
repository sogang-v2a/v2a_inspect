from __future__ import annotations

import unittest

from v2a_inspect.testing import build_fake_tooling_runtime
from v2a_inspect.tools.types import FrameBatch, SampledFrame


class FakeRuntimeTests(unittest.TestCase):
    def test_fake_runtime_returns_deterministic_tool_outputs(self) -> None:
        runtime = build_fake_tooling_runtime()
        batches = [
            FrameBatch(
                scene_index=0,
                frames=[
                    SampledFrame(
                        scene_index=0,
                        timestamp_seconds=0.0,
                        image_path="/tmp/fake0.jpg",
                    ),
                    SampledFrame(
                        scene_index=0,
                        timestamp_seconds=0.5,
                        image_path="/tmp/fake1.jpg",
                    ),
                ],
            )
        ]

        track_set = runtime.sam3_client.extract_entities(batches)
        embeddings = runtime.embedding_client.embed_images(
            {"fake-track-0": ["/tmp/fake0.jpg", "/tmp/fake1.jpg"]}
        )
        label_scores = runtime.label_client.score_image_labels(
            image_paths=["/tmp/fake0.jpg"],
            labels=["person", "vehicle"],
        )

        self.assertEqual(track_set.provider, "fake-sam3")
        self.assertEqual(len(track_set.tracks), 1)
        self.assertEqual(embeddings[0].model_name, "fake-dinov2")
        self.assertEqual(label_scores[0].label, "person")


if __name__ == "__main__":
    unittest.main()
