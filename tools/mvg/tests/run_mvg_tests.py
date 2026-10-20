"""The MVG acceptance runner (issue #101 §B).

The project's Python 3.9 environment has torch but no pytest, and the repository's own
convention is a standalone script that can be run directly. This is that script:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/tests/run_mvg_tests.py

CPU only. No GPU, no network, no training. It exits non-zero when a check fails.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mvg import gates as G  # noqa: E402
from tools.mvg import mutants as MUT  # noqa: E402
from tools.mvg import runner as RUN  # noqa: E402
from tools.mvg import spec as SPEC  # noqa: E402

TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r9_mvg_reference"
SPEC_PATH = SPEC.SPECS_DIR / "masked_directed_codebook_margin.json"

#: The defects the issue names, one for one.
NAMED_DEFECTS = (
    "gw_to_mse",
    "transpose_pc_cc",
    "latent_only_zero_codebook_gradient",
    "no_op_loss",
    "symmetric_for_directed",
    "detached_gradient",
    "wrong_masked_ddp_normalization",
)

CHECKS: List[Tuple[str, Callable[[], None]]] = []


def check(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def register(function: Callable[[], None]) -> Callable[[], None]:
        CHECKS.append((name, function))
        return function

    return register


def _load_spec() -> dict:
    return SPEC.load_spec(SPEC_PATH)


@check("the shipped specification is trustworthy")
def _spec_is_valid() -> None:
    problems = SPEC.validate_spec(_load_spec())
    assert problems == [], problems
    assert SPEC.schema()["required"], "the normative schema declares no required keys"


@check("the specification carries a valid CGV seven-group contract")
def _cgv_contract_is_valid() -> None:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "cgv"))
    import contract as cgv_contract  # noqa: E402

    doc = _load_spec()
    assert cgv_contract.validate(doc["cgv_contract"]) == []
    assert cgv_contract.spec_sha256(doc["cgv_contract"])


@check("the specification declares the named forbidden substitutions")
def _forbidden_substitutions() -> None:
    ids = {entry["id"] for entry in _load_spec()["forbidden_substitutions"]}
    assert ids == {
        "no_detached_path",
        "no_distance_mse_substitution",
        "no_symmetric_averaging",
        "no_batch_size_denominator",
    }, ids


@check("M0 passes on the reference tree")
def _m0() -> None:
    result = G.gate_m0(G.MechanismUnderTest(TREE, _load_spec()))
    assert result.state == G.PASSED, result.diagnostics
    assert result.measurements["static_audit_is_sufficient_alone"] is False


@check("M1 reproduces the independent reference")
def _m1() -> None:
    result = G.gate_m1(G.MechanismUnderTest(TREE, _load_spec()))
    assert result.state == G.PASSED, result.diagnostics
    assert result.measurements["cases_checked"] >= 4
    assert result.measurements["max_deviation"] <= 1e-9


@check("M2 measures the derivative of the declared loss")
def _m2() -> None:
    result = G.gate_m2(G.MechanismUnderTest(TREE, _load_spec()))
    assert result.state == G.PASSED, result.diagnostics
    assert result.measurements["finite_difference_max_deviation"] <= 1e-5
    assert result.measurements["all_masked_out"]["loss"] == 0.0
    assert result.measurements["all_masked_out"]["finite_gradients"] is True


@check("M3 holds the declared properties")
def _m3() -> None:
    result = G.gate_m3(G.MechanismUnderTest(TREE, _load_spec()))
    assert result.state == G.PASSED, result.diagnostics
    assert abs(result.measurements["swapped_loss"] - result.measurements["loss"]) > 1e-9


@check("M4 reaches the trained parameters and the exported pathway")
def _m4() -> None:
    result = G.gate_m4(G.MechanismUnderTest(TREE, _load_spec()))
    assert result.state == G.PASSED, result.diagnostics
    assert result.measurements["frozen_host_gap"] <= 1e-9
    assert result.measurements["encoder_gradient_change"] > 0.0
    assert result.measurements["codebook_gradient_max_abs"] > 0.0
    assert result.measurements["exported_token_changes_after_one_step"] > 0


@check("M5 rejects every named defect")
def _m5() -> None:
    result = RUN.gate_m5(G.MechanismUnderTest(TREE, _load_spec()))
    rows = {row["mutant"]: row for row in result.measurements["mutants"]}
    assert set(rows) == set(NAMED_DEFECTS), sorted(rows)
    for name, row in rows.items():
        assert row["caught"], f"{name} was not rejected: {row}"
        assert row["must_fail_gate"] in row["failed_gates"], row


@check("the mutation catalogue covers the named defects")
def _catalogue() -> None:
    assert {mutant.id for mutant in MUT.MUTANTS} == set(NAMED_DEFECTS)


@check("the report states every gate and its digests")
def _report() -> None:
    with tempfile.TemporaryDirectory(prefix="mvg-report-") as tmp:
        report = RUN.run(TREE, SPEC_PATH, Path(tmp))
        assert set(report["gates"]) == {"M0", "M1", "M2", "M3", "M4", "M5"}
        assert report["mechanism"]["spec_sha256"]
        assert report["mechanism"]["tree_code_sha256"]
        assert report["mechanism"]["commit"]
        assert report["coverage"]["mutants_injected"] == len(NAMED_DEFECTS)
        assert report["coverage"]["mutants_rejected"] == len(NAMED_DEFECTS)
        assert report["residual_uncertainty"]
        written = json.loads((Path(tmp) / "mvg_report.json").read_text())
        assert written["verdict"] == report["verdict"]


def main() -> int:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch is present in every environment that runs this
        pass
    failures = 0
    for name, function in CHECKS:
        try:
            function()
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}")
            print("      " + str(exc).replace("\n", "\n      ")[:1500])
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
        else:
            print(f"ok    {name}")
    print()
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
