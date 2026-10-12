import contextlib
import io

import torch

from supervision_location import LEVELS, TABLE, run


def test_l2_l3_auc_are_expected_exact_baseline_identities():
    """EXPECTED identity: zero-gradient residual path makes deep-only AUC baseline-identical."""
    with contextlib.redirect_stdout(io.StringIO()):
        rows = run()
    by_key = {(row["arm"], row["level"], row["pair_set"]): row for row in rows}
    for arm in ("l2only", "l3only"):
        for level in LEVELS:
            for pair_set in ("primary_valid", "secondary_train"):
                actual = by_key[(arm, level, pair_set)]["auc"]
                baseline = by_key[("TIGER", level, pair_set)]["auc"]
                assert actual == baseline, (
                    f"EXPECTED zero-gradient identity failed: {arm} {level} {pair_set} "
                    f"AUC {actual!r} != TIGER {baseline!r}"
                )


def test_deep_residual_straight_through_identity_has_zero_gradient():
    x = torch.tensor([1.25, -3.0, 4.5], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.5, 0.0, 8.0], dtype=torch.float64)
    quantized_st = x + (q - x).detach()
    level_one_residual = x - quantized_st
    level_one_residual.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_summary_contains_all_five_arms_and_three_levels():
    with contextlib.redirect_stdout(io.StringIO()):
        rows = run()
    assert {row["arm"] for row in rows} == {"TIGER", "Exp1a", "l1only", "l2only", "l3only"}
    assert {row["level"] for row in rows} == set(LEVELS)
    assert {row["pair_set"] for row in rows} == {"primary_valid", "secondary_train"}
    assert TABLE.exists()


if __name__ == "__main__":
    tests = [
        test_l2_l3_auc_are_expected_exact_baseline_identities,
        test_deep_residual_straight_through_identity_has_zero_gradient,
        test_summary_contains_all_five_arms_and_three_levels,
    ]
    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:
            failed.append((test.__name__, error))
            print(f"FAIL {test.__name__}: {error}")
    print(f"{len(tests) - len(failed)} passed, {len(failed)} failed")
    raise SystemExit(bool(failed))
