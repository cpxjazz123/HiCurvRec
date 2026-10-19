"""Track E (issue #84) tests: the projection's claims, each falsifiable on its own.

The closed-form single-half-space projection is the INDEPENDENT reference. The iterative solver is
checked against it, and the feasibility and false-guarantee cases are checked against arithmetic a
reader can verify by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"),):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_e_voronoi_projection as E  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def test_already_feasible_vector_is_untouched() -> None:
    # <g, x> = 0.7 >= 0.5, so this point already satisfies the constraint and must not move
    vector = torch.tensor([1.0, 0.7, 0.0], dtype=torch.float64)
    gradient = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    projected, _ = E.dykstra_project(vector, [(gradient, 0.5)])
    check("a feasible vector is not moved", torch.equal(projected, vector), str(projected))


def test_single_halfspace_matches_closed_form() -> None:
    """The solver must agree with the analytic projection AND satisfy the projection property.

    Agreement alone is not enough: a shared sign error in both would also agree, which is exactly
    the failure this test once had. So feasibility and minimality are checked independently.
    """
    torch.manual_seed(0)
    for trial in range(20):
        vector = torch.randn(6, dtype=torch.float64)
        gradient = torch.randn(6, dtype=torch.float64)
        bound = float(gradient @ vector) + float(torch.rand(1)) * 2.0
        iterative, _ = E.dykstra_project(vector, [(gradient, bound)])
        analytic = E.project_halfspace(vector, gradient, bound)
        if not torch.allclose(iterative, analytic, atol=1e-8):
            check(f"trial {trial}: solver matches the closed form", False,
                  f"max diff {float((iterative - analytic).abs().max()):.3e}")
            return
        if float(gradient @ iterative) < bound - 1e-8:
            check(f"trial {trial}: the result is FEASIBLE", False,
                  f"<g,x>={float(gradient @ iterative):.6f} < bound={bound:.6f}")
            return
        if float((iterative - vector).norm()) > float((analytic - vector).norm()) + 1e-9:
            check(f"trial {trial}: the result is MINIMAL", False, "the solver moved further")
            return
    check("the solver matches the closed form and is feasible and minimal on 20 trials", True)


def test_infeasible_vector_is_pushed_onto_the_boundary() -> None:
    """The sign of the correction must be visible: an infeasible point must move to the boundary."""
    vector = torch.tensor([0.0, 0.0], dtype=torch.float64)
    gradient = torch.tensor([1.0, 0.0], dtype=torch.float64)
    projected = E.project_halfspace(vector, gradient, 2.0)
    check("an infeasible vector lands exactly on the boundary",
          abs(float(projected[0]) - 2.0) < 1e-12, str(projected))


def test_zero_gradient_constraint_is_inert() -> None:
    vector = torch.tensor([1.0, 2.0], dtype=torch.float64)
    gradient = torch.zeros(2, dtype=torch.float64)
    check("a zero gradient cannot move the update",
          torch.equal(E.project_halfspace(vector, gradient, 1e9), vector), "it moved")


def test_active_set_selects_only_at_risk_items() -> None:
    margins = torch.tensor([0.5, 0.001, 2.0, 0.0], dtype=torch.float64)
    selected = E.active_set(margins, 0.01)
    check("only items inside the band are guarded",
          selected.tolist() == [1, 3], str(selected.tolist()))


def test_realised_violation_is_detected_when_the_step_crosses() -> None:
    """A linearisation can look satisfied while the true margin still crosses."""
    positions = torch.tensor([0, 1])
    before = {"margin": torch.tensor([0.2, 0.3], dtype=torch.float64)}
    after = {"margin": torch.tensor([0.2, 0.05], dtype=torch.float64)}   # item 1 crossed to 0.05
    result = E.realised_violations(before, after, positions, 0.1)
    check("a crossed margin is reported as a violation",
          result["violated"] == 1 and result["violation_rate"] == 0.5, str(result))
    check("the crossed item is also counted as having crossed",
          result["crossed_any"] == 1, str(result))


def test_no_items_guarded_is_reported_not_silently_zero() -> None:
    empty = torch.tensor([], dtype=torch.long)
    result = E.realised_violations({"margin": torch.zeros(0)}, {"margin": torch.zeros(0)},
                                   empty, 0.1)
    check("an empty active set is stated, not reported as a clean result",
          result["items_guarded"] == 0 and "no item" in result["definition"], str(result))


def test_infeasible_intersection_is_reported_not_hidden() -> None:
    """Two half-spaces that cannot both hold: the solver must not claim it satisfied both."""
    torch.manual_seed(1)
    vector = torch.randn(4, dtype=torch.float64)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    second = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    projected, _ = E.dykstra_project(vector, [(first, 5.0), (second, 5.0)])
    satisfied_first = float(first @ projected) >= 5.0 - 1e-8
    satisfied_second = float(second @ projected) >= 5.0 - 1e-8
    check("an infeasible pair is not reported as both satisfied",
          not (satisfied_first and satisfied_second),
          f"first={satisfied_first} second={satisfied_second}")


def test_relative_deviation_is_zero_for_the_identity() -> None:
    vector = torch.randn(5, dtype=torch.float64)
    check("identity projection has zero deviation",
          E.relative_deviation(vector, vector.clone()) == 0.0, "nonzero")


def test_constraint_bound_formula() -> None:
    """rhs = delta - m - code_term, so a margin already above delta needs no correction."""
    gradient = torch.tensor([1.0], dtype=torch.float64)
    # exactly representable in binary, so the assertion tests the formula and not rounding
    margin, code_term, delta = 0.5, 0.25, 0.5
    constraints = E.build_constraints([gradient], torch.tensor([margin]), [code_term], delta)
    check("the bound is delta - m - code_term",
          abs(constraints[0][1] - (delta - margin - code_term)) < 1e-12, str(constraints[0][1]))
    # bound = 0.5 - 0.5 - 0.25 = -0.25 <= 0, so the zero vector is already feasible
    vector = torch.tensor([0.0], dtype=torch.float64)
    projected, _ = E.dykstra_project(vector, constraints)
    check("a feasible vector stays put when the margin already clears delta",
          torch.allclose(projected, vector, atol=1e-9), str(projected))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print(f"r7_e_voronoi_projection: {CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
