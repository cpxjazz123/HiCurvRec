"""The audit base is the archive plus only the live pairs that carry a real path.

Reproduces the fault it guards against: ``collect.py`` records both the
repository path of a SID table and the copy it writes under
``analysis_sid_recall/sids/``. After the 2026 history rewrite the original paths
of 147 archived rows no longer resolve, so only the archived snapshot holds
them; meanwhile the copies' flattened filenames contain ``stage2_RQ-VAE`` and a
plain substring match happily selects them. Replacing the archive with a fresh
collection therefore loses evidence, and letting a copy-only row in trades a
real path for nothing.

Run: python3 -m unittest analysis_sid_recall.tests.test_pair_audit_sources
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis_sid_recall.pair_audit import audit_rows, repository_paths  # noqa: E402

FIELDS = ["sid_sha256", "sid_file", "sid_paths", "variant", "run_dir", "code_path",
          "sid_resolution", "pair_confidence", "test_recall@10"]

COPY = "analysis_sid_recall/sids/results__stage2_RQ-VAE__TIGER_a__item_sids__aa.json"
REAL = "results/stage2_RQ-VAE/TIGER_a/item_sids.json"


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(sha: str, paths: str, variant: str = "TIGER_a") -> dict:
    return {"sid_sha256": sha, "sid_file": "x", "sid_paths": paths, "variant": variant,
            "run_dir": "results/stage3_T5Train/%s/logs/d/run" % variant,
            "code_path": REAL, "sid_resolution": "date", "pair_confidence": "verified",
            "test_recall@10": "0.05"}


class RepositoryPaths(unittest.TestCase):
    def test_the_collected_copy_is_not_a_repository_path(self):
        self.assertEqual(repository_paths(row("a", COPY + " | " + REAL)), [REAL])

    def test_a_copy_only_row_offers_nothing(self):
        self.assertEqual(repository_paths(row("a", COPY)), [])


class AuditRows(unittest.TestCase):
    def test_the_archive_is_kept_and_a_tracked_new_pair_is_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, live = Path(tmp) / "full_2.csv", Path(tmp) / "sid_recall.csv"
            write(base, [row("aa", REAL, "TIGER_archived")])
            write(live, [row("aa", COPY + " | " + REAL, "TIGER_archived"),
                         row("bb", REAL, "TIGER_new")])
            rows, appended = audit_rows(base, live, is_tracked=lambda path: path == REAL)
            self.assertEqual([r["sid_sha256"] for r in rows], ["aa", "bb"])
            self.assertEqual(appended, 1)

    def test_a_copy_only_live_row_is_not_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, live = Path(tmp) / "full_1.csv", Path(tmp) / "sid_recall.csv"
            write(base, [row("aa", REAL, "TIGER_archived")])
            write(live, [row("cc", COPY, "TIGER_copy_only")])
            rows, appended = audit_rows(base, live, is_tracked=lambda path: True)
            self.assertEqual([r["sid_sha256"] for r in rows], ["aa"])
            self.assertEqual(appended, 0)

    def test_an_untracked_repository_path_is_not_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, live = Path(tmp) / "full_1.csv", Path(tmp) / "sid_recall.csv"
            write(base, [row("aa", REAL, "TIGER_archived")])
            write(live, [row("dd", REAL, "TIGER_untracked")])
            rows, appended = audit_rows(base, live, is_tracked=lambda path: False)
            self.assertEqual(appended, 0)
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
