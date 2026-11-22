"""The frozen baseline carries no hyperbolic code.

The baseline is the Euclidean RQ-VAE every candidate is copied from. It used to
keep two hyperbolic leftovers: `model/utils.py` defined the Poincare operators and
two hyperbolic quantisers, and `model/poincare_contract.py` held a copy of the
framework's geometry module. Nothing imported either of them -- the trainer reaches
`model.layers` and `model.model` only -- so they were dead weight in the tree that
every candidate inherits.

The operator names are the ones `tools/research/stage2/checks.py` forbids a
candidate from defining, so the same list is the rule here.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"

#: Every hyperbolic operator and quantiser the baseline used to define.
HYPERBOLIC = ("artanh", "proj_to_ball", "mobius_add", "lambda_x",
              "expmap0", "logmap0", "poincare_distance",
              "HVectorQuantization", "HResidualVectorQuantization")


def _python_files():
    for path in sorted(BASELINE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_baseline_file_defines_a_hyperbolic_operator() -> str:
    hits = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in HYPERBOLIC:
            if ("def " + name) in text or ("class " + name) in text:
                hits.append("%s defines %s" % (path.relative_to(BASELINE), name))
    if hits:
        raise AssertionError("the baseline still carries hyperbolic code: " + "; ".join(hits))
    return "no baseline file defines a hyperbolic operator or quantiser"


def test_the_baseline_holds_no_copy_of_the_geometry_contract() -> str:
    copy = BASELINE / "model" / "poincare_contract.py"
    if copy.is_file():
        raise AssertionError("the baseline still ships its own copy of the geometry module")
    return "the geometry module lives in tools/research/geometry/, not in the baseline"


def test_the_contract_consumer_points_at_the_framework_module() -> str:
    source = (REPO / "tools" / "research" / "science" / "community_curvature" /
              "model.py").read_text(encoding="utf-8")
    if "TIGER_RQ-VAE" in source:
        raise AssertionError("the consumer still reads the contract out of the baseline tree")
    canonical = REPO / "tools" / "research" / "geometry" / "poincare_contract.py"
    if not canonical.is_file():
        raise AssertionError("the canonical geometry module is missing: " + str(canonical))
    return "the contract consumer reads tools/research/geometry/poincare_contract.py"


def test_the_trainer_never_imports_the_hyperbolic_module() -> str:
    for relative in ("train_rqvae.py", "model/__init__.py", "model/model.py", "model/layers.py"):
        text = (BASELINE / relative).read_text(encoding="utf-8")
        for needle in ("from .utils import", "from model.utils import", "poincare_contract"):
            if needle in text:
                raise AssertionError("%s imports %r; the baseline's training path must not"
                                     % (relative, needle))
    return "the training path imports neither utils.py nor the geometry module"


def main() -> int:
    failures = 0
    for test in (test_no_baseline_file_defines_a_hyperbolic_operator,
                 test_the_baseline_holds_no_copy_of_the_geometry_contract,
                 test_the_contract_consumer_points_at_the_framework_module,
                 test_the_trainer_never_imports_the_hyperbolic_module):
        try:
            note = test()
            print("PASS %s: %s" % (test.__name__, note))
        except AssertionError as exc:
            failures += 1
            print("FAIL %s: %s" % (test.__name__, exc))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
