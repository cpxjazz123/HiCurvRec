"""The diagnosis-provenance acceptance runner (issue #105, workstream A).

The project's Python 3.9 environment has torch but no pytest, and the repository's own
convention is a standalone script that can be run directly. This is that script:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/tests/run_diagnosis_provenance_tests.py

CPU only. No GPU, no network, no training. It exits non-zero when a check fails.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from tools.mvg import invariants  # noqa: E402


def _case() -> dict:
    return {
        "source": torch.ones(2, 3, dtype=torch.float64),
        "target": torch.ones(2, 3, dtype=torch.float64),
        "codebook": torch.ones(4, 3, dtype=torch.float64),
        "mask": torch.ones(2, dtype=torch.float64),
    }


def test_diagnosis_records_its_provenance() -> None:
    def entry(source, target, codebook, mask, **parameters):
        return (source * 1.0).sum()

    diagnosis = invariants.diagnose_gradient_failure(
        entry, _case(), {"source": "nonzero", "codebook": "zero"}, {}, provenance="prototype"
    )
    assert diagnosis["source"]["provenance"] == "prototype", diagnosis
    assert diagnosis["codebook"]["provenance"] == "prototype", diagnosis


def test_unknown_provenance_is_refused() -> None:
    def entry(source, target, codebook, mask, **parameters):
        return (source * 1.0).sum()

    try:
        invariants.diagnose_gradient_failure(entry, _case(), {"source": "nonzero"}, {},
                                             provenance="vibes")
    except ValueError:
        return
    raise AssertionError("an unknown provenance must be refused")


def test_conflicting_prototype_and_module_is_unresolved() -> None:
    prototype = {"codebook": {"provenance": "prototype", "verdict": "impossible specification"}}
    integrated = {"codebook": {"provenance": "integrated", "verdict": "delivered"}}
    verdict = invariants.reconcile_diagnoses(prototype, integrated)
    assert verdict["state"] == "unresolved", verdict
    assert verdict["governing_provenance"] == "integrated", verdict
    assert verdict["conflicts"][0]["target"] == "codebook", verdict


def test_agreeing_levels_are_consistent() -> None:
    level = {"provenance": "integrated", "verdict": "delivered"}
    verdict = invariants.reconcile_diagnoses(
        {"codebook": level}, {"codebook": dict(level, provenance="prototype")}
    )
    assert verdict["state"] == "consistent", verdict
    assert not verdict["conflicts"], verdict


def main() -> int:
    tests: List[Tuple[str, Callable[[], None]]] = [
        (name, value) for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
