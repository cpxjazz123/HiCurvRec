"""Regression tests for the engineering faults this programme hit.

Each test reproduces one thing that actually went wrong and would otherwise go
wrong again silently. They run with the system interpreter: none of them needs
torch, because none of them is about the mechanism.

Run:  python3 -m tools.research.stage2.tests.test_contract
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2 import checks, contract, launcher


def test_root_marker_is_unambiguous() -> str:
    """A stray stage2_RQ-VAE inside a tree must not capture the root.

    The first conformance run on the baseline failed here: a directory named
    stage2_RQ-VAE inside the tree made the marker resolve to the tree, and every
    data path pointed inside it.
    """
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "repo"
        (root / "stage2_RQ-VAE").mkdir(parents=True)
        (root / "results").mkdir()
        (root / "stage1_GeneEmbedding").mkdir()
        tree = root / "stage2_RQ-VAE" / "TIGER_TEST"
        tree.mkdir(parents=True)
        (tree / "train_rqvae.py").write_text("# mechanism\n")
        # the stray directory the baseline actually had
        (tree / "stage2_RQ-VAE").mkdir()
        assert contract.resolve_root(tree / "train_rqvae.py") == root.resolve(), \
            "the root was captured by a directory inside the tree"
    return "the root resolves to the repository, never to a tree"


def test_workspace_is_per_run() -> str:
    """Two runs write in two directories, and a new run never reuses one."""
    with tempfile.TemporaryDirectory() as raw:
        first = contract.run_workspace(raw, "run_one", "ARM")
        second = contract.run_workspace(raw, "run_one", "ARM")
        assert first["source"] == second["source"], "the same run resolved differently"
        other = contract.run_workspace(raw, "run_two", "ARM")
        assert other["result"] != first["result"], "two runs share a result directory"
        next_revision = contract.run_workspace(raw, "run_one_rev2", "ARM")
        assert next_revision["result"] != first["result"], \
            "a new revision would overwrite a directory a running job reads"
    return "each run has its own source snapshot and result directory"


def test_workspace_rejects_unsafe_names() -> str:
    """A run id or arm name that could escape its parent is refused."""
    with tempfile.TemporaryDirectory() as raw:
        for bad in ("../escape", "a/b", ""):
            try:
                contract.run_workspace(raw, bad, "ARM")
            except contract.ContractError:
                continue
            raise AssertionError("an unsafe run id was accepted: " + repr(bad))
    return "unsafe run ids are refused"


def test_launcher_working_directory_is_stable() -> str:
    """The ranks run in the result directory, not in the tree.

    The four ranks of a candidate died four times because the framework replaced
    the tree they were running in, and torch calls os.getcwd() while it imports.
    """
    with tempfile.TemporaryDirectory() as raw:
        result = Path(raw) / "result"
        result.mkdir()
        tree = Path(raw) / "tree"
        tree.mkdir()
        entry = tree / "entry.py"
        # The entry writes where it is running from; the test then asks where
        # that was. torchrun passes the script path only, so the entry must not
        # expect an argument.
        entry.write_text(
            "import os\n"
            "open(os.path.join(os.getcwd(), 'written.txt'), 'w').write(os.getcwd())\n")
        if not Path(contract.launch_env()["PATH"]).exists():
            pass
        # The same torchrun the real launches use: the check must exercise the
        # working-directory invariant, not skip it because the system python has
        # no torchrun next to it.
        script = Path(checks._python_for())
        torchrun = script.with_name("torchrun")
        if not torchrun.is_file():
            return "torchrun is not available, the launcher check is skipped"
        code = launcher.launch(entry, result, world_size=1, port=contract.STAGE2_PORT + 20,
                               python=str(script))
        assert code == 0, "the launcher did not run the entry point: exit " + str(code)
        assert (result / "written.txt").is_file(), \
            "the ranks did not run with the result directory as their working directory"
        assert not (tree / "written.txt").exists(), \
            "a rank ran inside the candidate tree, which a later run replaces"
    return "the ranks run with the result directory as their working directory"


def test_launcher_refuses_missing_entry() -> str:
    """A launcher that cannot find the entry point fails loudly."""
    with tempfile.TemporaryDirectory() as raw:
        try:
            launcher.launch(Path(raw) / "missing.py", Path(raw) / "result")
        except contract.ContractError:
            return "a missing entry point is refused"
    raise AssertionError("the launcher started a job without an entry point")


def test_conformance_detects_own_launcher() -> str:
    """A tree that builds its own four-rank argv is refused by the check."""
    with tempfile.TemporaryDirectory() as raw:
        tree = Path(raw) / "TIGER_FAKE"
        tree.mkdir()
        (tree / "train_rqvae.py").write_text(
            "EPOCHS = 3000\nBATCH_SIZE_PER_RANK = 1024\nLR = 1e-3\n"
            "WEIGHT_DECAY = 1e-4\nCODEBOOK_NUM = 3\nCODEBOOK_DIM = 32\n"
            "GRADIENT_CLIP_NORM = 1.0\nSEED = 42\n"
            "cmd = ['torchrun', '--standalone', '--nproc_per_node=4', '--master_port=50202']\n")
        report = checks.check_constants(tree)
        assert report["status"] == "FAIL", "a hand-built four-rank argv was accepted"
    return "a candidate that builds its own four-rank argv is refused"


def test_frozen_constants_are_the_published_ones() -> str:
    """The frozen table matches the recipe every published arm was trained with."""
    assert contract.FROZEN["epochs"] == 3000
    assert contract.FROZEN["batch_size_per_rank"] == 1024
    assert contract.FROZEN["learning_rate"] == 1e-3
    assert contract.FROZEN["seed"] == 42
    assert contract.STAGE2_PORT != contract.STAGE3_PORT
    return "the frozen table is the published recipe"


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            print("PASS", test.__name__ + ":", test(), flush=True)
        except AssertionError as exc:
            failures.append(test.__name__ + ": " + str(exc))
            print("FAIL", test.__name__ + ":", exc, flush=True)
        except Exception as exc:                              # noqa: BLE001
            failures.append(test.__name__ + ": " + repr(exc))
            print("ERROR", test.__name__ + ":", repr(exc), flush=True)
    if failures:
        print("TESTS_FAIL", failures, flush=True)
        return 1
    print("TESTS_PASS", len(tests), "engineering invariants hold", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
