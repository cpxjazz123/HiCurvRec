"""Backfill the Issue #118 failure registry with the defects #117 actually hit.

Every entry below is a defect that blocked a real run, the commit that fixed it,
and the regression that now proves it stays fixed. Recording them once turns the
registry into a ledger a new run can be checked against, instead of a list of
anecdotes; re-running this script is idempotent because entries deduplicate by
their fingerprint.
"""
from __future__ import annotations

from pathlib import Path

from ..core import failures

STATE = Path(__file__).resolve().parents[3] / "results" / "research" / "issue117"
REGRESSION = "tools/research/tests/test_contract_preflight.py"
VERIFICATION = "tools/research/verification/tests/test_grouped_runner.py"

DEFECTS = [
    {"stage": "prompt", "error_class": "MATH", "rule": "FLAT_LIMIT_D",
     "expected": "limit of d_c^2 is 4*||x-y||^2 (and of d_c is 2*||x-y||)",
     "actual": "prompt told agents the unsquared distance tends to 4*||x-y||^2",
     "fix_commit": "2f4fb8f9",
     "regression_test": REGRESSION + "::MathLayerTests::test_the_unsquared_limit_is_a_counterexample"},
    {"stage": "math", "error_class": "MATH", "rule": "CLAMP_DOMINATES_DOMAIN",
     "expected": "c*|x-y|^2 >> delta at the declared curvature",
     "actual": "delta clamp drove the flat limit to 2*delta/c",
     "fix_commit": "36294d71",
     "regression_test": REGRESSION + "::MathLayerTests::test_a_clamp_that_dominates_the_curvature_is_a_counterexample"},
    {"stage": "falsifier", "error_class": "FALSIFIER", "rule": "FALSE_POSITIVE_NAMES",
     "expected": "field references and container names are not diagnostic claims",
     "actual": "checker rejected stage3_behavioral_test.x, OR, stage2_diagnostics, minimum_effect",
     "fix_commit": "0d6710e5",
     "regression_test": "tools/research/tests/test_contracts.py::FalsifierTests"},
    {"stage": "review", "error_class": "HASH_BINDING", "rule": "SUITE_KEY_MISMATCH",
     "expected": "one canonical suite identity",
     "actual": "manifest keyed math, reviewer keyed test_math.py",
     "fix_commit": "243c28e3",
     "regression_test": VERIFICATION + "::SuiteKeyNormalisationTests"},
    {"stage": "import", "error_class": "PYTHON_RUNTIME", "rule": "PEP604_WITHOUT_FUTURE",
     "expected": "research package imports under the deployed python3.9",
     "actual": "Path | None raised unsupported operand at import time",
     "fix_commit": "bcf15a60",
     "regression_test": ".github/workflows/research-verification.yml::deployed-python"},
    {"stage": "admission", "error_class": "PYTHON_RUNTIME", "rule": "RELATIVE_IMPORT_STANDALONE",
     "expected": "admission.py loads when imported by file path",
     "actual": "attempted relative import with no known parent package",
     "fix_commit": "f4dc7cb2",
     "regression_test": "tools/research/verification/tests/test_grouped_runner.py"},
    {"stage": "tests", "error_class": "DDP", "rule": "TOY_DDP",
     "expected": "two ranks train the candidate model through the DDP wrapper",
     "actual": "draft used Linear(3,2) or called ddp.module(x), bypassing the reducer",
     "fix_commit": "bdbe2556",
     "regression_test": "tools/research/verification/cases/issue117-geometry-transition-residual-poincare-v1/test_ddp.py"},
    {"stage": "tests", "error_class": "ENTRYPOINT", "rule": "MOCKED_INITIALISATION",
     "expected": "the real trainer runs with real KMeans initialisation",
     "actual": "draft mocked init_codebook, leaving torch.empty codebooks and a NaN loss",
     "fix_commit": "bdbe2556",
     "regression_test": "tools/research/verification/cases/issue117-geometry-transition-residual-poincare-v1/test_entrypoint.py"},
    {"stage": "tests", "error_class": "MATH", "rule": "ORACLE_4DELTA",
     "expected": "arcosh(1+delta)^2 = 2*delta",
     "actual": "draft expected 4*delta",
     "fix_commit": "bdbe2556",
     "regression_test": "tools/research/verification/cases/issue117-geometry-transition-residual-poincare-v1/test_math.py"},
    {"stage": "impl", "error_class": "ENTRYPOINT", "rule": "SAMPLER_NOT_NORMALISED",
     "expected": "edge sampling proportional to the row-normalised T_ab",
     "actual": "cumulative last entry forced to 1.0 while T_ab sums to the source count",
     "fix_commit": "7e72a1ac",
     "regression_test": "tools/research/verification/cases/issue117-geometry-transition-residual-poincare-v1/test_entrypoint.py::TransitionSamplerTests"},
    {"stage": "review", "error_class": "HASH_BINDING", "rule": "STALE_DRAFT_PROSE",
     "expected": "the committed, materialised tests are the reviewed evidence",
     "actual": "author draft prose quoted pre-correction values and blocked approval",
     "fix_commit": "365d8879",
     "regression_test": "tools/research/verification/tests/test_grouped_runner.py"},
    {"stage": "stage2", "error_class": "MATH", "rule": "COLLAPSE_UNGUARDED",
     "expected": "an objective term that keeps the latent informative",
     "actual": "3000 epochs produced one distinct three-token SID for 24587 items (ER 1.0)",
     "fix_commit": "111ac1d6",
     "regression_test": REGRESSION + "::CollapseGuardTests"},
    {"stage": "smoke", "error_class": "ENTRYPOINT", "rule": "FORWARD_LOSS_UNPROBED",
     "expected": "the smoke probes a loss returned from forward()",
     "actual": "no compute_loss, so probed_terms stayed empty and a healthy run was refused",
     "fix_commit": "2f81d3be",
     "regression_test": "tools/genrec-mcp/server.py::_probed_forward"},
]


def run() -> dict:
    recorded = []
    for defect in DEFECTS:
        entry = dict(defect)
        entry.setdefault("status", "verified")
        entry.setdefault("run_id", "")
        recorded.append(failures.record(STATE, entry, repeat=False))
    return failures.summary(STATE)


if __name__ == "__main__":  # no-argument research entry point
    print(run())
