"""The native mixed quantizer, checked against closed forms that do not reuse it.

Every oracle here is derived independently of the implementation: the ball
distance by the standard closed form
``2 * artanh(sqrt(c) |x-y| / sqrt((1-c|x|^2)(1-c|y|^2))) / sqrt(c)``, the
Euclidean limit by the plain squared distance, the residual composition by the
left-cancellation law of Mobius addition, and the weight ordering by an
enumeration of the two candidate costs written out by hand.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.mixed_rq import (  # noqa: E402
    ALPHA_CANDIDATES,
    AlphaTable,
    EUCLIDEAN_DIM,
    MixedRQ,
    MixedVQLevel,
    TOTAL_DIM,
    encode_tokens,
    mixed_cost,
    quantized_readout,
    split,
)


def _ball_points(count: int, dim: int, curvature: float, radius: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(count, dim, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    radial = torch.rand(count, 1, generator=generator, dtype=torch.float64) * radius
    return direction * radial


def _closed_form_distance(x: torch.Tensor, y: torch.Tensor, curvature: float) -> torch.Tensor:
    """``d_c`` through the arc-cosh form, not through Mobius addition.

    ``d = arcosh(1 + 2 c |x-y|^2 / ((1 - c|x|^2)(1 - c|y|^2))) / sqrt(c)``. An
    earlier draft used the artanh form with the same argument, which is wrong by
    a normalisation: at ``y = 0`` it returns ``2 artanh(a / sqrt(1 - a^2))``
    instead of the distance ``2 artanh(a)``.
    """
    c = curvature
    squared_x = x.pow(2).sum(-1)
    squared_y = y.pow(2).sum(-1)
    difference = (x - y).pow(2).sum(-1)
    denominator = ((1 - c * squared_x) * (1 - c * squared_y)).clamp_min(1e-30)
    argument = 1.0 + 2.0 * c * difference / denominator
    return torch.acosh(argument.clamp_min(1.0 + 1e-15)) / math.sqrt(c)


def test_cost_matches_the_closed_form() -> str:
    """The cost equals 2[alpha E^2 + (1-alpha) d_c^2/4] with d_c from its definition."""
    for curvature in (1.0, 0.0):
        residual = _ball_points(6, TOTAL_DIM, curvature, 0.2, seed=17)
        codes = _ball_points(5, TOTAL_DIM, curvature, 0.2, seed=29)
        residual_e, residual_h = split(residual)
        codes_e, codes_h = split(codes)
        from model.poincare_contract import expmap0
        tangent_h = codes_h
        # The codebook's curved half is a tangent-space parameter, so the ball
        # point the distance is taken at is exp_0 of it - at c=0 the map is the
        # identity.
        ball_h = expmap0(codes_h, curvature) if curvature > 0 else codes_h
        for alpha_value in ALPHA_CANDIDATES:
            alpha = torch.full((residual.shape[0],), alpha_value, dtype=torch.float64)
            got = mixed_cost(residual_e, residual_h, codes_e, tangent_h, alpha, curvature)
            euclidean = (residual_e.unsqueeze(1) - codes_e.unsqueeze(0)).pow(2).sum(-1)
            if curvature == 0:
                curved = 4.0 * (residual_h.unsqueeze(1) - ball_h.unsqueeze(0)).pow(2).sum(-1)
            else:
                curved = _closed_form_distance(
                    residual_h.unsqueeze(1).expand(-1, ball_h.shape[0], -1).reshape(-1, EUCLIDEAN_DIM),
                    ball_h.unsqueeze(0).expand(residual_h.shape[0], -1, -1).reshape(-1, EUCLIDEAN_DIM),
                    curvature,
                ).pow(2).reshape(residual_h.shape[0], -1)
            want = 2.0 * (alpha_value * euclidean + (1.0 - alpha_value) * curved / 4.0)
            torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-12)
    return "the cost matches the closed form at c=1 and c=0 for every candidate weight"


def test_euclidean_limit_is_the_plain_squared_distance() -> str:
    """At c=0 with alpha=1/2 the cost is the 32-dimensional squared distance."""
    z = torch.randn(9, TOTAL_DIM, dtype=torch.float64)
    codes = torch.randn(7, TOTAL_DIM, dtype=torch.float64)
    residual_e, residual_h = split(z)
    codes_e, codes_h = split(codes)
    alpha = torch.full((z.shape[0],), 0.5, dtype=torch.float64)
    got = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha, curvature=0.0)
    want = (z.unsqueeze(1) - codes.unsqueeze(0)).pow(2).sum(-1)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    return "the c=0, alpha=1/2 limit recovers the 32-dimensional squared distance"


def test_the_weight_reverses_the_order_of_two_disagreeing_codes() -> str:
    """The counterexample the spec demands: E cost [1, 4], H cost [4, 1].

    At alpha=1/4 the curved half dominates and code 2 is nearer; at alpha=3/4
    the Euclidean half dominates and code 1 is nearer. A scalar weight can only
    do this because the code is one joint index.
    """
    residual = torch.zeros(1, TOTAL_DIM, dtype=torch.float64)
    residual_e, residual_h = split(residual)
    codes = torch.zeros(2, TOTAL_DIM, dtype=torch.float64)
    # c = 0 keeps the two halves commensurate, so the costs are exact.
    codes[0, 0] = 1.0            # ||eE||^2 = 1
    codes[1, 0] = 2.0            # ||eE||^2 = 4
    codes[0, EUCLIDEAN_DIM] = 2.0  # d^2/4 = 4
    codes[1, EUCLIDEAN_DIM] = 1.0  # d^2/4 = 1
    codes_e, codes_h = split(codes)
    low = mixed_cost(residual_e, residual_h, codes_e, codes_h,
                     torch.tensor([0.25], dtype=torch.float64), curvature=0.0)
    high = mixed_cost(residual_e, residual_h, codes_e, codes_h,
                      torch.tensor([0.75], dtype=torch.float64), curvature=0.0)
    torch.testing.assert_close(low, torch.tensor([[6.5, 3.5]], dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(high, torch.tensor([[3.5, 6.5]], dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)
    if int(torch.argmin(low, dim=-1)) == int(torch.argmin(high, dim=-1)):
        raise AssertionError("the weight did not change which joint code is nearest")
    return "alpha=0.25 picks code 2 and alpha=0.75 picks code 1"


def test_the_alpha_derivative_is_the_specified_difference() -> str:
    """``dC/dalpha = 2 (E^2 - d_c^2/4)``, measured numerically."""
    residual = _ball_points(5, TOTAL_DIM, 1.0, 0.2, seed=41)
    codes = _ball_points(4, TOTAL_DIM, 1.0, 0.2, seed=43)
    residual_e, residual_h = split(residual)
    codes_e, codes_h = split(codes)
    step = 1e-6
    alpha_low = torch.full((residual.shape[0],), 0.5 - step, dtype=torch.float64)
    alpha_high = torch.full((residual.shape[0],), 0.5 + step, dtype=torch.float64)
    numeric = (mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha_high, 1.0)
               - mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha_low, 1.0)) / (2 * step)
    euclidean = (residual_e.unsqueeze(1) - codes_e.unsqueeze(0)).pow(2).sum(-1)
    from model.poincare_contract import expmap0
    ball_h = expmap0(codes_h, 1.0)
    curved = _closed_form_distance(
        residual_h.unsqueeze(1).expand(-1, ball_h.shape[0], -1).reshape(-1, EUCLIDEAN_DIM),
        ball_h.unsqueeze(0).expand(residual_h.shape[0], -1, -1).reshape(-1, EUCLIDEAN_DIM),
        1.0,
    ).pow(2).reshape(residual_h.shape[0], -1)
    analytic = 2.0 * (euclidean - curved / 4.0)
    torch.testing.assert_close(numeric, analytic, rtol=1e-9, atol=1e-12)
    return "the measured derivative matches 2 (E^2 - d^2/4)"


def test_the_residual_composes_to_the_original_point() -> str:
    """The native residual is a true subtraction: the codes compose back.

    ``rE + sum(eE) = zE`` by left cancellation, and the curved half composes
    right-nested, ``e1 (+) (e2 (+) (e3 (+) rH)) = zH``, which is exactly why the
    readout uses that bracketing.
    """
    z = _ball_points(256, TOTAL_DIM, 1.0, 0.2, seed=53)
    level = MixedVQLevel(64)
    z_e, z_t = split(z)
    # The table has to be initialised the way training initialises it: a code
    # outside the ball is projected onto the boundary and the composition
    # identity then saturates, which says nothing about the residual rule.
    from model.poincare_contract import expmap0 as _expmap0
    level.init_codebook(z_e, _expmap0(z_t, 1.0), torch.device("cpu"))
    residual_e, residual_h = z_e, torch.tanh(z_t) * 0.0 + z_t
    from model.poincare_contract import expmap0, mobius_add
    residual_h = expmap0(z_t, 1.0)
    codes_e, codes_h = [], []
    for _ in range(3):
        alpha = torch.full((z.shape[0],), 0.5)
        (residual_e, residual_h), _, _, index, picked_e, picked_h = level(
            residual_e, residual_h, alpha)
        codes_e.append(picked_e.detach())
        codes_h.append(picked_h.detach())
    composed_e = residual_e + torch.stack(codes_e, 0).sum(0)
    # The level's table is float32, so the identity is checked to float32
    # precision, not to the float64 precision of the probe itself.
    torch.testing.assert_close(composed_e.float(), z_e.float(), rtol=1e-5, atol=1e-5)
    composed_h = residual_h
    for code in reversed(codes_h):
        composed_h = mobius_add(code, composed_h, 1.0)
    torch.testing.assert_close(composed_h.float(), expmap0(z_t, 1.0).float(), rtol=1e-4, atol=1e-4)
    return "both halves compose back to the encoded point"


def test_the_readout_is_the_nested_ball_sum() -> str:
    """``phi(q) = [sum eE, log_0(e1 (+) (e2 (+) e3))]`` with the right nesting."""
    from model.poincare_contract import expmap0, logmap0, mobius_add
    generator = torch.Generator().manual_seed(59)
    codes = [expmap0(torch.randn(4, EUCLIDEAN_DIM, generator=generator, dtype=torch.float64) * 0.05, 1.0)
             for _ in range(3)]
    euclidean = [torch.randn(4, EUCLIDEAN_DIM, generator=generator, dtype=torch.float64) for _ in range(3)]
    got = quantized_readout(euclidean, codes, 1.0)
    want_e = torch.stack(euclidean, 0).sum(0)
    want_h = mobius_add(codes[0], mobius_add(codes[1], codes[2], 1.0), 1.0)
    want = torch.cat([want_e, logmap0(want_h, 1.0)], dim=-1)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    return "the readout is the Euclidean sum and the right-nested ball sum"


def test_the_euclidean_arm_matches_a_plain_nearest_code_search() -> str:
    """The E arm (c=0) takes the same codes as a plain 32-dimensional argmin."""
    torch.manual_seed(71)
    table = AlphaTable()
    rq = MixedRQ([32, 32, 32], table, curvature=0.0)
    z = torch.randn(256, TOTAL_DIM)
    rq.init_codebooks(z, torch.device("cpu"))
    tokens = encode_tokens(z, rq)
    residual = z.clone()
    want = []
    for level in rq.levels:
        distances = (residual.unsqueeze(1) - level.get_code_embs().unsqueeze(0)).pow(2).sum(-1)
        index = torch.argmin(distances, dim=-1)
        want.append(index)
        residual = residual - level.get_code_embs()[index]
    want_tokens = torch.stack(want, dim=1)
    if not tokens.equal(want_tokens):
        raise AssertionError("the c=0 arm did not follow the plain nearest-code search")
    return "the c=0 arm reproduces a plain 32-dimensional residual search"


def main() -> int:
    failures = 0
    for test in (
        test_cost_matches_the_closed_form,
        test_euclidean_limit_is_the_plain_squared_distance,
        test_the_weight_reverses_the_order_of_two_disagreeing_codes,
        test_the_alpha_derivative_is_the_specified_difference,
        test_the_residual_composes_to_the_original_point,
        test_the_readout_is_the_nested_ball_sum,
        test_the_euclidean_arm_matches_a_plain_nearest_code_search,
    ):
        try:
            print("PASS " + test.__name__ + ": " + test())
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc))
        except Exception as exc:  # noqa: BLE001 - a crash is a failure of the check
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
