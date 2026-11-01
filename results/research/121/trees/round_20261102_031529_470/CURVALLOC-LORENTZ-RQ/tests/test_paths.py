"""The trainer's paths: the root marker, the arm directory and the frozen data.

The repository root is found by a marker, and a marker that is not unique makes a
run from a workspace under ``results/`` write to ``results/results/...``.
"""
import re
import unittest
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
TRAINER = TREE / "train_rqvae.py"


def repository_root() -> Path:
    """The repository root, by the same unambiguous marker the trainer uses."""
    for parent in Path(__file__).resolve().parents:
        if all((parent / marker).is_dir()
               for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return parent
    raise RuntimeError("no repository root above " + str(TREE))


def _resolver_source() -> str:
    source = TRAINER.read_text(encoding="utf-8")
    start = source.index("def _repository_root")
    end = source.index("REPO_ROOT = _repository_root()")
    return source[start:end]


def _resolve(file_path: Path) -> Path:
    namespace = {"Path": Path, "__file__": str(file_path)}
    exec(_resolver_source(), namespace)
    return namespace["_repository_root"]()


class RootMarkerTests(unittest.TestCase):

    def test_the_marker_is_unambiguous(self):
        source = _resolver_source()
        for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding"):
            self.assertIn(marker, source)

    def test_the_root_resolves_from_every_location_the_framework_uses(self):
        repo = repository_root()
        locations = [
            repo / "stage2_RQ-VAE" / "TIGER_poincare-frechet-successor-codebook" / "train_rqvae.py",
            repo / "stage2_RQ-VAE" / "_runs" / "run_x" / "arm" / "src" / "train_rqvae.py",
            repo / "results" / "research" / "issue121" / "trees" / "round" / "idea" / "train_rqvae.py",
            repo / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "train_rqvae.py",
        ]
        for location in locations:
            self.assertEqual(_resolve(location).name, repo.name, str(location))

    def test_the_results_directory_is_not_taken_for_the_root(self):
        """results/ holds a stage2_RQ-VAE of its own and must not win the search."""
        repo = repository_root()
        inner = repo / "results" / "stage2_RQ-VAE" / "_runs" / "r" / "a" / "src" / "train_rqvae.py"
        self.assertNotEqual(_resolve(inner), repo / "results")


class PathConstantTests(unittest.TestCase):

    def test_the_arm_and_the_result_directory_agree(self):
        source = TRAINER.read_text(encoding="utf-8")
        arm = re.search(r'^ARM_NAME\s*=\s*"([^"]+)"', source, re.M).group(1)
        self.assertTrue(arm.startswith("TIGER_"))
        self.assertIn('RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME', source)

    def test_no_source_file_holds_an_absolute_path(self):
        pattern = re.compile(r'["\'](/home/|/fs04/|/tmp/)')
        for path in list(TREE.glob("*.py")) + list((TREE / "model").glob("*.py")):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")),
                              "absolute path in " + path.name)

    def test_no_parent_counting(self):
        for path in list(TREE.glob("*.py")) + list((TREE / "model").glob("*.py")):
            self.assertNotIn("parents[", path.read_text(encoding="utf-8"), path.name)

    def test_the_frozen_inputs_are_the_project_ones(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('results" / "stage0_build_parquet" / "train.parquet"', source)
        self.assertIn('"stage1_GeneEmbedding" / "output" / "sentence_t5.npy"', source)

    def test_the_mechanism_is_not_started_by_the_candidate(self):
        """The launcher owns torchrun; the candidate must not fork it itself."""
        source = TRAINER.read_text(encoding="utf-8")
        for forbidden in ("--nproc_per_node=", "--master_port="):
            self.assertNotIn(forbidden, source.replace('_LAUNCHER["script"]', ""))


if __name__ == "__main__":
    unittest.main()
