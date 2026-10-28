"""The copy baseline is the last complete mechanism, not the frozen seed.

From the second round on, a mechanism is built on the previous mechanism that
finished the whole chain: implemented, reviewed, trained and evaluated in
Stage 3. These checks pin the resolver, the fallback, and the one framework-owned
constant the copy is allowed to change.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import baseline as baseline_module


def _trainer(arm: str) -> str:
    return 'ARM_NAME = "%s"\nRESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME\n' % arm


def _round(root: Path, name: str, jobs) -> None:
    directory = root / "results" / "research" / "issue121" / "rounds" / name
    directory.mkdir(parents=True)
    (directory / "jobs.json").write_text(json.dumps({"schema": 1, "jobs": jobs}))


def _arm(root: Path, idea_id: str) -> Path:
    tree = root / "stage2_RQ-VAE" / ("TIGER_" + idea_id)
    tree.mkdir(parents=True)
    (tree / "train_rqvae.py").write_text(_trainer("TIGER_" + idea_id), encoding="utf-8")
    return tree


class PreviousImplementationTests(unittest.TestCase):

    def test_the_latest_complete_mechanism_is_the_baseline(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            old = _arm(root, "old-idea")
            new = _arm(root, "new-idea")
            _round(root, "round_20260101_000000_1",
                   [{"idea_id": "old-idea", "kind": "stage3", "status": "done"}])
            _round(root, "round_20260202_000000_2",
                   [{"idea_id": "new-idea", "kind": "stage3", "status": "done"}])
            self.assertEqual(baseline_module.previous_implementation(root), new.resolve())
            self.assertNotEqual(baseline_module.previous_implementation(root), old.resolve())

    def test_a_round_whose_stage3_never_finished_is_not_a_baseline(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            _arm(root, "done-idea")
            _arm(root, "running-idea")
            _round(root, "round_20260101_000000_1",
                   [{"idea_id": "done-idea", "kind": "stage3", "status": "done"}])
            _round(root, "round_20260202_000000_2",
                   [{"idea_id": "running-idea", "kind": "stage2", "status": "running"},
                    {"idea_id": "running-idea", "kind": "stage3", "status": "pending"}])
            picked = baseline_module.previous_implementation(root)
            self.assertEqual(picked, (root / "stage2_RQ-VAE" / "TIGER_done-idea").resolve())

    def test_a_round_without_jobs_is_skipped(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            _arm(root, "known")
            _round(root, "round_20260101_000000_1",
                   [{"idea_id": "known", "kind": "stage3", "status": "done"}])
            (root / "results" / "research" / "issue121" / "rounds" /
             "round_20260303_000000_3").mkdir()
            self.assertEqual(baseline_module.previous_implementation(root),
                             (root / "stage2_RQ-VAE" / "TIGER_known").resolve())

    def test_no_complete_mechanism_means_no_answer(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertIsNone(baseline_module.previous_implementation(Path(work)))


class CopyTests(unittest.TestCase):

    def test_the_copy_renames_the_arm_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            source = _arm(root, "previous-idea")
            (source / "model").mkdir()
            (source / "model" / "m.py").write_text("x = 1\n", encoding="utf-8")
            destination = root / "workspace" / "TIGER_candidate"
            provenance = baseline_module.copy_baseline(destination, baseline=source)
            self.assertEqual(provenance["arm_renamed"],
                             {"from": "TIGER_previous-idea", "to": "TIGER_candidate"})
            text = (destination / "train_rqvae.py").read_text(encoding="utf-8")
            self.assertIn('ARM_NAME = "TIGER_candidate"', text)
            self.assertTrue(baseline_module._same_source_except_arm(source, destination))

    def test_a_copy_that_changes_the_mechanism_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            source = _arm(root, "previous-idea")
            destination = root / "workspace" / "TIGER_candidate"
            provenance = baseline_module.copy_baseline(destination, baseline=source)
            self.assertEqual(provenance["baseline"], str(source.resolve()))
            (destination / "train_rqvae.py").write_text(
                (destination / "train_rqvae.py").read_text(encoding="utf-8") + "\n# edit\n",
                encoding="utf-8")
            self.assertFalse(baseline_module._same_source_except_arm(source, destination))

    def test_the_copy_carries_the_previous_mechanisms_tests(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            source = _arm(root, "previous-idea")
            (source / "tests").mkdir()
            (source / "tests" / "test_geometry.py").write_text("x = 1\n", encoding="utf-8")
            destination = root / "workspace" / "TIGER_candidate"
            baseline_module.copy_baseline(destination, baseline=source)
            self.assertTrue((destination / "tests" / "test_geometry.py").is_file())


if __name__ == "__main__":
    unittest.main()


class CanonicalGeometryTests(unittest.TestCase):
    """A candidate receives the canonical module and may not redefine the geometry."""

    def test_the_copy_carries_the_canonical_module(self):
        from tools.research.rounds import baseline as baseline_module
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            source = root / "stage2_RQ-VAE" / "TIGER_baseline"
            source.mkdir(parents=True)
            (source / "train_rqvae.py").write_text(
                'ARM_NAME = "TIGER_baseline"\n', encoding="utf-8")
            destination = root / "trees" / "round" / "idea"
            baseline_module.copy_baseline(destination, baseline=source)
            self.assertTrue((destination / "model" / "poincare_contract.py").is_file())

    def test_a_candidate_that_redefines_the_geometry_is_refused(self):
        from tools.research.stage2 import checks
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "TIGER_candidate"
            (tree / "model").mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (tree / "model" / "poincare_contract.py").write_text(
                (Path(__file__).resolve().parents[1] / "geometry" /
                 "poincare_contract.py").read_text(encoding="utf-8"), encoding="utf-8")
            report = checks.check_geometry_contract(tree)
            self.assertEqual(report["status"], "PASS", report)
            (tree / "model" / "mine.py").write_text(
                "def expmap0(v, c):\n    return v\n", encoding="utf-8")
            report = checks.check_geometry_contract(tree)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("expmap0", report["detail"])
