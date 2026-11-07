"""The conformance run must not write into the run it is checking.

The fault this refuses: ``check_entry`` redirects a candidate's results into a
scratch directory, but a trainer computes ``METRICS_PATH`` from ``LOG_DIR`` at
import time. Patching ``LOG_DIR`` afterwards therefore leaves the metrics
stream pointing at the REAL result directory, and the one-epoch conformance run
truncates the 3000-epoch stream that a published arm's report cites as its
evidence. It happened: the arm's own stream became a single row.

The test builds a throwaway tree whose trainer appends one row to its
``METRICS_PATH`` in ``main()``, runs the real ``check_entry`` on it, and
requires the tree's own metrics file to be untouched.

Run:  python3 -m tools.research.stage2.tests.test_entry_check_scratch
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2 import checks, contract  # noqa: E402

TRAINER = '''
"""A tree that resolves the repository root the way every arm does.

The metrics stream lives under the REPOSITORY's result tree, not under the
tree's own directory: that is what makes the conformance run able to clobber a
published arm's evidence when the check redirects LOG_DIR but not METRICS_PATH.
"""
import json
from pathlib import Path


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("no repository root")


REPO_ROOT = _repository_root()
ARM_NAME = "_test_entry_scratch"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME
LOG_DIR = RESULT_DIR / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
OUTPUT_SID = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON = RESULT_DIR / "item_sids.json"
CHECKPOINT = RESULT_DIR / "rqvae_best.pth"


def main():
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as sink:
        sink.write(json.dumps({"event": "train", "epoch": 1}) + "\\n")
    OUTPUT_JSON.write_text(json.dumps({"0": [1, 2, 3, 1]}))
'''


def run() -> int:
    root = contract.resolve_root()
    tree = root / "stage2_RQ-VAE" / "_test_entry_scratch"
    failures = []
    try:
        tree.mkdir(parents=True, exist_ok=True)
        (tree / "train_rqvae.py").write_text(TRAINER)
        marker = (root / "results" / "stage2_RQ-VAE" / "_test_entry_scratch"
                  / "logs" / "training_metrics.jsonl")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"event": "train", "epoch": 0}) + "\n")
        before = marker.read_text()

        report = checks.check_entry(tree)

        after = marker.read_text()
        if after != before:
            failures.append("check_entry overwrote the tree's metrics stream: "
                            + after.strip()[:120])
        if report["status"] not in ("PASS", "FAIL"):
            failures.append("check_entry returned an unexpected status: "
                            + str(report["status"]))
    finally:
        import shutil
        shutil.rmtree(tree, ignore_errors=True)
        for leftover in (root / "stage2_RQ-VAE").glob("_admission__test_entry_scratch*"):
            shutil.rmtree(leftover, ignore_errors=True)
        import shutil as _shutil
        _shutil.rmtree(root / "results" / "stage2_RQ-VAE" / "_test_entry_scratch",
                       ignore_errors=True)

    if failures:
        for failure in failures:
            print("FAIL:", failure)
        return 1
    print("PASS: the conformance run leaves the checked tree's metrics stream alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
