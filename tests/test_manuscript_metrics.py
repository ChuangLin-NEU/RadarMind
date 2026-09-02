"""Manuscript metric contract: source files → snapshot → expected displays."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_v2.snapshot import build_snapshot  # noqa: E402

SNAP = ROOT / "results" / "final_delivery" / "metric_snapshot_full.json"
CONTRACT = ROOT / "tests" / "expected_manuscript_snapshot.json"


class ManuscriptMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(SNAP.read_text(encoding="utf-8"))
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))["display"]
        cls.live = build_snapshot(ROOT)

    def test_each_contract_key_traces_to_source(self) -> None:
        for key, expected_display in self.contract.items():
            stored = self.stored["metrics"][key]
            live = self.live["metrics"][key]
            self.assertEqual(
                stored["display"],
                expected_display,
                f"{key}: snapshot display drifted from manuscript contract",
            )
            self.assertEqual(
                live["display"],
                stored["display"],
                f"{key}: live registry rebuild differs from stored snapshot",
            )
            self.assertEqual(
                live["value"],
                stored["value"],
                f"{key}: live value differs from stored snapshot",
            )


if __name__ == "__main__":
    unittest.main()
