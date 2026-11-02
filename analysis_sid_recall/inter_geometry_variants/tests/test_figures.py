"""A supported figure may not render the empty state (Issue #135)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw as stats  # noqa: E402


class FigureTests(unittest.TestCase):
    def test_every_supported_scatter_has_points(self):
        audit_path = HERE.parent / "figure_audit.json"
        if not audit_path.is_file():
            self.skipTest("figure_audit.json has not been produced yet")
        for name, block in json.loads(audit_path.read_text()).items():
            content = (HERE.parent / name).read_text()
            if "no defined values" in content:
                self.fail(f"{name} is empty")
            if block.get("defined", 0):
                self.assertGreater(block["defined"], 0)

    def test_the_helper_counts_points(self):
        rows = [{"series": "s", "recall": 0.1, "x_l2": 1.0},
                {"series": "s", "recall": 0.2, "x_l2": 2.0}]
        path = Path("/tmp/_scatter_probe.svg")
        self.assertEqual(stats.scatter_level_svg(rows, 2, "x_l", path, "x"), 2)


if __name__ == "__main__":
    unittest.main()
