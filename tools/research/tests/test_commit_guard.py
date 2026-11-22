"""The commit preflight must catch files that `commit.draft()` would have to diff.

The guard reads only Git's path lists and `lstat`; a large generated table is never read. These tests
run in a disposable Git repository under `results/research/_generated_commit_guard_test/` (CLAUDE.md
86: no `/tmp`), create small files, and check both the failing and the passing paths.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "commit_guard.py"
SCRATCH_ROOT = ROOT / "results" / "research" / "_generated_commit_guard_test"


class CommitGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="guard-",
            dir=str(SCRATCH_ROOT),
        )
        self.repo = Path(self._temporary.name)
        self.git("init", "-q")
        self.git("config", "user.email", "guard-test@example.invalid")
        self.git("config", "user.name", "guard test")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def guard(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), "--repo", str(self.repo), "--max-mb", "0.001", *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_large_untracked_file_fails(self) -> None:
        (self.repo / "big.csv").write_text("x" * 4096)
        result = self.guard()
        self.assertEqual(result.returncode, 1)
        self.assertIn("big.csv", result.stderr)

    def test_small_untracked_file_passes(self) -> None:
        (self.repo / "small.csv").write_text("x" * 16)
        result = self.guard()
        self.assertEqual(result.returncode, 0)
        self.assertIn("ok", result.stdout)

    def test_ignored_large_file_passes(self) -> None:
        (self.repo / ".gitignore").write_text("*.csv\n")
        (self.repo / "big.csv").write_text("x" * 4096)
        result = self.guard()
        self.assertEqual(result.returncode, 0)

    def test_large_modified_tracked_file_fails(self) -> None:
        path = self.repo / "tracked.csv"
        path.write_text("original\n")
        self.git("add", "tracked.csv")
        self.git("commit", "-qm", "initial")
        path.write_text("x" * 4096)
        result = self.guard()
        self.assertEqual(result.returncode, 1)
        self.assertIn("tracked.csv", result.stderr)

    def test_explicit_allow_passes(self) -> None:
        (self.repo / "evidence.csv").write_text("x" * 4096)
        result = self.guard("--allow", "evidence.csv")
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
