from __future__ import annotations

import unittest
from pathlib import Path

from v2a_inspect.contracts import load_gold_set_manifest


class GoldSetManifestTests(unittest.TestCase):
    def test_manifest_loads_with_human_notes(self) -> None:
        manifest = load_gold_set_manifest(
            Path("tests/fixtures/gold_set/manifest.json")
        )
        self.assertEqual(manifest.version, "stage0-v1")
        self.assertGreaterEqual(len(manifest.clips), 6)
        for clip in manifest.clips:
            self.assertTrue(clip.visible_sources)
            self.assertTrue(clip.event_notes)
            self.assertTrue(clip.grouping_notes)
            self.assertTrue(clip.routing_notes)


if __name__ == "__main__":
    unittest.main()
