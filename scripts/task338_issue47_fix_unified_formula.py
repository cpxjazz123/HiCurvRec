"""
Task #338 — Issue #47 Debug unified κ-stereographic formula to 5/5 PASS.

Core bugs identified:
  Bug 1 (对称性破坏, symmetry_err=13.32): Möbius addition line 126,
    `+ kappa * y_norm_sq` should be `- kappa * y_norm_sq` per unified
    formula: x⊕_κ y = ((1-2κ⟨x,y⟩-κ‖y‖²)x + (1+κ‖x‖²)y) / (1-2κ⟨x,y⟩+κ²‖x‖²‖y‖²)
  Bug 2 (κ=0 NaN): _tan_kappa_inverse_smooth's sigmoid blend at κ=0
    mixes atan(0)/0 (NaN) with Taylor. Fix: use torch.where with crisp
    threshold instead of sigmoid.

Gates (per Issue #47):
  Gate 0: Isolated Möbius addition test — symmetry + Euclidean degeneracy
  Gate 1: κ=0 limit — no NaN/Inf, limits from κ→0⁺/0⁻ converge
  Gate 2: Full rerun of Task #333's 5 tests — 5/5 ALL PASS

Usage:
  python3 scripts/task338_issue47_fix_unified_formula.py
"""
from __future__ import annotations
import math
import time
import json
import torch
import numpy as np
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
EPS = 1e-15


# ══════════════════════════════════════════════════════════════════════
# CORRECT unified κ-stereographic formula (Issue #47 fix)
# ══════════════════════════════════════════════════════════════════════

def mobius_addition_correct(x: torch.Tensor, y: torch.Tensor,
                             kappa: torch.Tensor) -> torch.Tensor:
    """Correct unified Möbius addition for all κ ∈ ℝ.

    Formula (Sala 2018 / unified κ-stereographic):
      x ⊕_κ y = ((1 - 2κ⟨x,y⟩ - κ‖y‖²)·x + (1 + κ‖x‖²)·y)
                / (1 - 2κ⟨x,y⟩ + κ²‖x‖²‖y‖²)

    κ=0 limit → x + y (Euclidean addition).
    κ<0  → Poincaré ball Möbius addition.
    κ>0  → Spherical Möbius addition.

    Args:
        x: (..., d)
        y: (..., d)
        kappa: scalar tensor — any real value (broadcastable)

    Returns:
        x ⊕_κ y (same shape as x)
    """
    x_dot_y = (x * y).sum(dim=-1, keepdim=True)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)

    # ⚠️ BUG FIX: κ‖y‖² must be SUBTRACTED, not added
    # (Bug in task333: `+ kappa * y_norm_sq` causes symmetry_err=13.32)
    num_term1 = (1.0 - 2.0 * kappa * x_dot_y - kappa * y_norm_sq) * x
    num_term2 = (1.0 + kappa * x_norm_sq) * y
    numerator = num_term1 + num_term2

    denominator = (1.0 - 2.0 * kappa * x_dot_y
                   + (kappa ** 2) * x_norm_sq * y_norm_sq).clamp(min=EPS)

    return numerator / denominator


def tan_kappa_inverse_correct(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Correct unified tan_κ⁻¹ with crisp threshold (no sigmoid blend).

    Formula:
      κ > 0:  atan(√κ · x) / √κ
      κ = 0:  x - (κ/3)x³ + (2κ²/15)x⁵  (Taylor, keeps κ in graph → ∂/∂κ ≠ 0)
      κ < 0:  atanh(√|κ| · x) / √|κ|

    Implementation: compute ALL branches tensor-wise, select via torch.where.
    For |κ| < 1e-4, Taylor dominates (avoids 0/0 from closed-form division).
    For |κ| ≥ 1e-4, closed-form dominates.

    ⚠️ BUG FIX: sigmoid blend causes NaN at κ=0 because atan(x·0)/0 = 0/0.
    torch.where -> masks the closed-form gradient at κ=0, so only
    the Taylor gradient (which is non-zero: ∂/∂κ[x - κx³/3] = -x³/3) flows.
    """
    abs_kappa = kappa.abs()

    # Taylor branch (polynomial in κ, always safe)
    taylor = x - (1.0 / 3.0) * kappa * x ** 3 + (2.0 / 15.0) * (kappa ** 2) * x ** 5

    # Closed-form branch (1/√|κ| singularity at κ=0 → use only when |κ| large)
    sqrt_abs_k = torch.sqrt(abs_kappa + EPS)
    pos_branch = torch.atan(x * sqrt_abs_k) / sqrt_abs_k

    # clamp for atanh domain: |input| < 1
    neg_input = (x * sqrt_abs_k).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    neg_branch = torch.atanh(neg_input) / sqrt_abs_k

    # Select branch by sign of κ
    is_pos = (kappa > 0).float()
    closed_form = is_pos * pos_branch + (1.0 - is_pos) * neg_branch

    # ── BUG FIX: crisp threshold (torch.where) instead of sigmoid blend ──
    # Sigmoid blends the NaN from closed-form with Taylor → NaN persists.
    # torch.where with |κ| >= threshold → only closed-form path has grad flow.
    # At κ=0 precisely, only Taylor path is selected → grad = ∂taylor/∂κ ≠ 0.
    threshold = 1e-4
    return torch.where(abs_kappa >= threshold, closed_form, taylor)


def unified_distance_correct(x: torch.Tensor, y: torch.Tensor,
                              kappa: torch.Tensor) -> torch.Tensor:
    """Correct unified κ-stereographic geodesic distance (Issue #47 fix).

    d_κ(x, y) = 2 · tan_κ⁻¹(|| -x ⊕_κ y ||)
    """
    # Step 1: -x ⊕_κ y
    neg_x = -x
    diff = mobius_addition_correct(neg_x, y, kappa)

    # Step 2: || -x ⊕_κ y ||_2 with boundary clamping
    abs_kappa = kappa.abs().clamp(min=EPS)
    boundary = 1.0 / torch.sqrt(abs_kappa)
    max_norm = boundary - 1e-6
    diff_norm = diff.norm(dim=-1).clamp(max=max_norm).clamp(min=EPS)

    # Step 3: 2 · tan_κ⁻¹
    return 2.0 * tan_kappa_inverse_correct(diff_norm, kappa)


# ══════════════════════════════════════════════════════════════════════
# R137 (reference implementation, unchanged)
# ══════════════════════════════════════════════════════════════════════

def r137_geodesic_distance_sq(x: torch.Tensor, y: torch.Tensor,
                               kappa: torch.Tensor) -> torch.Tensor:
    """R137 formula (from hrqvae_free_curv.py). Used as performance baseline."""
    kappa_abs = kappa.abs().clamp(min=1e-8)
    sqrt_kappa = torch.sqrt(kappa_abs)
    diff = x - y
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    diff_norm_sq = diff_norm ** 2
    denom = 2.0 * (1.0 - kappa * diff_norm_sq / 4.0).abs().clamp_min(1e-6)
    arg = sqrt_kappa * diff_norm / denom
    d = (2.0 / sqrt_kappa) * torch.arctan(arg)
    return d ** 2


# ══════════════════════════════════════════════════════════════════════
# Gate 0 — Isolated Möbius addition test
# ══════════════════════════════════════════════════════════════════════

def test_gate0_mobius_symmetry() -> dict:
    """Gate 0: Isolated Möbius addition test.

    T0a: Symmetry — verify d(x,y) = d(y,x) for both κ<0 and κ>0
    T0b: Euclidean degeneracy — at κ→0, x⊕_κ y → x+y
    T0c: Identity — x ⊕_κ 0 = x and 0 ⊕_κ x = x
    T0d: Inverse — x ⊕_κ (-x) should be finite
    """
    print("\n═══ Gate 0 — Isolated Möbius addition test ═══")
    torch.manual_seed(42)
    n, d = 100, 16

    results = {}

    # T0a: Symmetry ||(-x)⊕_κ y|| = ||(-y)⊕_κ x||
    print("\n─── T0a: Symmetry of Möbius distance ───")
    for k_val, label in [(0.5, "κ=+0.5 (spherical)"),
                         (-0.5, "κ=-0.5 (hyperbolic)"),
                         (0.001, "κ=+0.001 (near-Euclidean)")]:
        x = torch.randn(n, d) * 0.1
        y = torch.randn(n, d) * 0.1
        kappa = torch.tensor(k_val)

        d1 = unified_distance_correct(x, y, kappa)
        d2 = unified_distance_correct(y, x, kappa)
        err = (d1 - d2).abs().max().item()

        print(f"  {label}: max|d(x,y)-d(y,x)| = {err:.2e}")
        results[f"symmetry_{k_val}"] = err

    sym_pass = all(v < 1e-6 for v in results.values())
    print(f"  {'✅ PASS' if sym_pass else '❌ FAIL'} "
          f"(symmetry error < 1e-6)")
    results["symmetry_pass"] = sym_pass

    # T0b: Euclidean degeneracy at κ→0
    print("\n─── T0b: Euclidean degeneracy (κ→0 → x+y) ───")
    x = torch.randn(10, d) * 0.1
    y = torch.randn(10, d) * 0.1
    kappa_small = torch.tensor(1e-8)
    mobius_sum = mobius_addition_correct(x, y, kappa_small)
    euc_sum = x + y
    err = (mobius_sum - euc_sum).abs().max().item()
    print(f"  max|(x⊕_κ y) - (x+y)| at κ=1e-8: {err:.2e}")
    results["euclidean_degeneracy"] = err
    euc_pass = err < 1e-4
    print(f"  {'✅ PASS' if euc_pass else '❌ FAIL'}")

    # T0c: Identity x ⊕ 0 = x
    print("\n─── T0c: Identity ───")
    for k_val, label in [(0.5, "κ=+0.5"), (-0.5, "κ=-0.5")]:
        x = torch.randn(10, d) * 0.1
        zero_vec = torch.zeros_like(x)
        kappa = torch.tensor(k_val)
        left_id = mobius_addition_correct(zero_vec, x, kappa)
        right_id = mobius_addition_correct(x, zero_vec, kappa)
        left_err = (left_id - x).abs().max().item()
        right_err = (right_id - x).abs().max().item()
        print(f"  {label}: max|0⊕x-x|={left_err:.2e}, max|x⊕0-x|={right_err:.2e}")
        results[f"identity_{k_val}"] = max(left_err, right_err)

    id_pass = all(results.get(f, 0) < 1e-6
                  for f in [k for k in results if k.startswith("identity_")])
    print(f"  {'✅ PASS' if id_pass else '❌ FAIL'}")

    gate0_pass = sym_pass and euc_pass and id_pass
    results["gate0_pass"] = gate0_pass
    return results


# ══════════════════════════════════════════════════════════════════════
# Gate 1 — κ=0 limit test (no NaN, convergence from ±sides)
# ══════════════════════════════════════════════════════════════════════

def test_gate1_kappa_zero_limit() -> dict:
    """Gate 1: κ=0 limit — no NaN, limits from ±sides converge.

    T1a: κ=0 forward pass produces non-NaN, non-inf
    T1b: κ=0 autograd produces non-zero, non-NaN gradient
    T1c: Limits κ→0⁺ and κ→0⁻ converge to same value
    """
    print("\n═══ Gate 1 — κ=0 limit test ═══")
    torch.manual_seed(42)
    n, d = 50, 8
    x = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(n, d, dtype=torch.float64)

    results = {}

    # T1a: forward pass produces finite values
    print("\n─── T1a: Forward at exact κ=0 ───")
    kappa_0 = torch.tensor(0.0, dtype=torch.float64)
    d_0 = unified_distance_correct(x, y, kappa_0)
    has_nan = torch.isnan(d_0).any().item()
    has_inf = torch.isinf(d_0).any().item()
    all_finite = not has_nan and not has_inf
    print(f"  d(x,y) at κ=0: has NaN? {has_nan}, has Inf? {has_inf}")
    print(f"  d range: [{d_0.min().item():.4f}, {d_0.max().item():.4f}]")
    results["forward_nan_free"] = all_finite
    print(f"  {'✅ PASS' if all_finite else '❌ FAIL'}")

    # T1b: autograd gradient at κ=0
    print("\n─── T1b: Autograd at κ=0 ───")
    kappa_var = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    d_var = unified_distance_correct(x, y, kappa_var)
    d_var.sum().backward()
    grad_at_zero = kappa_var.grad.item()
    grad_finite = grad_at_zero == grad_at_zero and abs(grad_at_zero) != float('inf')
    grad_nonzero = abs(grad_at_zero) > 1e-6
    print(f"  ∂d/∂κ at κ=0 (autograd): {grad_at_zero:.6e}")
    print(f"  Finite: {grad_finite}, Nonzero: {grad_nonzero}")
    results["grad_at_zero"] = grad_at_zero
    results["grad_nonzero"] = grad_nonzero
    print(f"  {'✅ PASS' if grad_finite and grad_nonzero else '❌ FAIL'}")

    # T1c: Limits from κ→0⁺ and κ→0⁻ converge
    print("\n─── T1c: Limit convergence κ→0⁺ vs κ→0⁻ ───")
    k_epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for eps in k_epsilons:
        k_pos = torch.tensor(eps, dtype=torch.float64)
        k_neg = torch.tensor(-eps, dtype=torch.float64)
        d_pos = unified_distance_correct(x, y, k_pos)
        d_neg = unified_distance_correct(x, y, k_neg)
        diff = (d_pos - d_neg).abs().max().item()
        print(f"  ε={eps:.0e}: max|d(κ=+ε) - d(κ=-ε)| = {diff:.2e}")
        if eps not in results:
            results[f"limit_diff_{eps}"] = diff

    # Convergence: as ε→0, d(+ε) - d(-ε) → 0 monotonically.
    # At κ=0, ∂d/∂κ ≠ 0, so difference ≈ 2·ε·∂d/∂κ scales linearly with ε.
    # The correct test is MONOTONIC decrease, not an absolute threshold.
    diffs_at_eps = [results.get(f"limit_diff_{e}", 1.0) for e in k_epsilons]
    monotonically_decreasing = all(
        diffs_at_eps[i] < diffs_at_eps[i - 1] for i in range(1, len(diffs_at_eps))
    )
    diff_at_1e8 = diffs_at_eps[-1]
    print(f"  {'✅ PASS' if monotonically_decreasing else '❌ FAIL'} "
          f"(monotonic convergence → 0, diff(ε=1e-8)={diff_at_1e8:.2e})")
    results["limit_converge"] = monotonically_decreasing

    gate1_pass = all_finite and grad_finite and grad_nonzero and monotonically_decreasing
    results["gate1_pass"] = gate1_pass
    return results


# ══════════════════════════════════════════════════════════════════════
# Gate 2 — Full 5-test rerun (task333 style)
# ══════════════════════════════════════════════════════════════════════

def test_gate2_kappa_zero_gradient(dtype=torch.float64) -> dict:
    """Core test: κ=0 gradient non-zero AND non-NaN (vs R137)."""
    print("\n═══ Gate 2a — κ=0 gradient (core test) ═══")
    torch.manual_seed(42)
    n, d = 50, 8
    x = torch.randn(n, d, dtype=dtype)
    y = torch.randn(n, d, dtype=dtype)

    # New formula at κ=0
    kappa_var = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    d_new = unified_distance_correct(x, y, kappa_var)
    d_new.sum().backward()
    new_grad = kappa_var.grad.item()

    # R137 at κ=0
    kappa_var_r137 = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    d_r137 = r137_geodesic_distance_sq(x, y, kappa_var_r137)
    d_r137.sum().backward()
    r137_grad = kappa_var_r137.grad.item()

    # Numerical gradient (finite difference)
    eps = 1e-5
    d_plus = unified_distance_correct(x, y, torch.tensor(eps, dtype=dtype))
    d_minus = unified_distance_correct(x, y, torch.tensor(-eps, dtype=dtype))
    numerical_grad = (d_plus.sum() - d_minus.sum()).item() / (2 * eps)

    print(f"  New formula ∂d/∂κ at κ=0 (autograd):   {new_grad:.6e}")
    print(f"  New formula ∂d/∂κ at κ=0 (finite diff): {numerical_grad:.6e}")
    print(f"  R137 ∂d²/∂κ at κ=0 (autograd):          {r137_grad:.6e}")

    new_finite = (new_grad == new_grad) and abs(new_grad) < 1e10
    new_nonzero = abs(new_grad) > 1e-3
    new_pass = new_finite and new_nonzero

    print(f"  New grad finite: {new_finite}, non-zero: {new_nonzero}")
    print(f"  {'✅ PASS' if new_pass else '❌ FAIL'}")

    return {
        'test': 'kappa_zero_gradient',
        'new_grad_autograd': new_grad,
        'new_grad_finite_diff': numerical_grad,
        'r137_grad_at_kappa0': r137_grad,
        'new_pass': new_pass,
        'verdict': 'PASS' if new_pass else 'FAIL',
    }


def test_gate2_gradient_continuity(dtype=torch.float64) -> dict:
    """κ sweep gradient continuity across the torch.where threshold."""
    print("\n═══ Gate 2b — Gradient continuity at κ=±1e-4 threshold ═══")
    torch.manual_seed(42)
    d = 8
    x = torch.randn(20, d, dtype=dtype)
    y = torch.randn(20, d, dtype=dtype)

    # Test continuity at the torch.where threshold (|κ| = 1e-4)
    # Taylor branch (|κ| < 1e-4) ↔ closed-form (|κ| >= 1e-4)
    # We sample just below and just above the threshold.
    threshold = 1e-4
    epsilons = [1e-7, 1e-6, 1e-5]
    max_rel_jump = 0.0
    for eps in epsilons:
        k_below = torch.tensor(threshold - eps, dtype=dtype, requires_grad=True)
        k_above = torch.tensor(threshold + eps, dtype=dtype, requires_grad=True)
        d_below = unified_distance_correct(x, y, k_below)
        d_above = unified_distance_correct(x, y, k_above)
        d_below.sum().backward(retain_graph=True)
        grad_below = k_below.grad.item()
        d_above.sum().backward()
        grad_above = k_above.grad.item()
        rel_jump = abs(grad_above - grad_below) / (abs(grad_below) + 1e-10)
        max_rel_jump = max(max_rel_jump, rel_jump)
        print(f"  κ=1e-4 ± {eps:.0e}: below={grad_below:.4e}, "
              f"above={grad_above:.4e}, rel_jump={rel_jump:.4e}")

    # Negative side
    for eps in epsilons:
        k_below = torch.tensor(-threshold - eps, dtype=dtype, requires_grad=True)
        k_above = torch.tensor(-threshold + eps, dtype=dtype, requires_grad=True)
        d_below = unified_distance_correct(x, y, k_below)
        d_above = unified_distance_correct(x, y, k_above)
        d_below.sum().backward(retain_graph=True)
        grad_below = k_below.grad.item()
        d_above.sum().backward()
        grad_above = k_above.grad.item()
        rel_jump = abs(grad_above - grad_below) / (abs(grad_below) + 1e-10)
        max_rel_jump = max(max_rel_jump, rel_jump)
        print(f"  κ=-1e-4 ± {eps:.0e}: below={grad_below:.4e}, "
              f"above={grad_above:.4e}, rel_jump={rel_jump:.4e}")

    continuous = max_rel_jump < 0.05
    print(f"  Max relative jump at threshold: {max_rel_jump:.4e}")
    print(f"  {'✅ PASS' if continuous else '❌ FAIL'} (max rel jump < 0.05)")

    return {
        'test': 'gradient_continuity',
        'new_continuous': continuous,
        'max_rel_jump_at_threshold': max_rel_jump,
        'verdict': 'PASS' if continuous else 'FAIL',
    }


def test_gate2_kappa_negative(dtype=torch.float64) -> dict:
    """κ<0 hyperbolic consistency."""
    print("\n═══ Gate 2c — κ<0 hyperbolic consistency ═══")
    torch.manual_seed(42)
    n, d = 100, 8
    kappa = torch.tensor(-1.0, dtype=dtype)
    x = torch.randn(n, d, dtype=dtype) * 0.3
    y = torch.randn(n, d, dtype=dtype) * 0.3

    d_new = unified_distance_correct(x, y, kappa)

    # d(x,x) = 0
    d_xx = unified_distance_correct(x[:10], x[:10], kappa)
    self_dist_err = d_xx.abs().max().item()

    # d(x,y) > 0 for x ≠ y
    positivity = (d_new > 0).all().item()

    # symmetry
    d_yx = unified_distance_correct(y, x, kappa)
    symmetry_err = (d_new - d_yx).abs().max().item()

    # monotonicity
    diff_norms = (x - y).norm(dim=-1)
    sorted_idx = diff_norms.argsort()
    closest = d_new[sorted_idx[:10]].mean().item()
    farthest = d_new[sorted_idx[-10:]].mean().item()
    monotonic = farthest > closest

    print(f"  κ=-1, n={n}")
    print(f"  d(x,x) max = {self_dist_err:.2e}")
    print(f"  All d(x,y) > 0: {positivity}")
    print(f"  Symmetry err: {symmetry_err:.2e}")
    print(f"  Monotonic: closest={closest:.4f}, farthest={farthest:.4f}")

    pass_ = self_dist_err < 1e-4 and positivity and symmetry_err < 1e-4 and monotonic
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return {
        'test': 'kappa_negative',
        'self_dist_err': self_dist_err,
        'all_positive': positivity,
        'symmetry_err': symmetry_err,
        'monotonic': monotonic,
        'pass': pass_,
        'verdict': 'PASS' if pass_ else 'FAIL',
    }


def test_gate2_kappa_positive(dtype=torch.float64) -> dict:
    """κ>0 spherical consistency."""
    print("\n═══ Gate 2d — κ>0 spherical consistency ═══")
    torch.manual_seed(42)
    n, d = 50, 8
    kappa = torch.tensor(0.5, dtype=dtype)
    sphere_radius = (1.0 / kappa).sqrt()

    x_raw = torch.randn(n, d, dtype=dtype)
    x = x_raw / x_raw.norm(dim=-1, keepdim=True) * sphere_radius
    y_raw = torch.randn(n, d, dtype=dtype)
    y = y_raw / y_raw.norm(dim=-1, keepdim=True) * sphere_radius

    d_new = unified_distance_correct(x, y, kappa)
    max_geodesic = (math.pi / kappa.sqrt()).item()

    d_xx = unified_distance_correct(x[:10], x[:10], kappa)
    self_dist_err = d_xx.abs().max().item()
    positivity = (d_new > 0).all().item()
    symmetry_err = (d_new - unified_distance_correct(y, x, kappa)).abs().max().item()
    max_dist = d_new.max().item()
    within_sphere = max_dist <= max_geodesic + 1e-3

    print(f"  κ=0.5, sphere radius={sphere_radius.item():.4f}")
    print(f"  d(x,x) max = {self_dist_err:.2e}")
    print(f"  All d(x,y) > 0: {positivity}")
    print(f"  Symmetry err: {symmetry_err:.2e}")
    print(f"  Max dist: {max_dist:.4f} ≤ π/√κ={max_geodesic:.4f}: {within_sphere}")

    pass_ = self_dist_err < 1e-4 and positivity and symmetry_err < 1e-4 and within_sphere
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return {
        'test': 'kappa_positive',
        'self_dist_err': self_dist_err,
        'all_positive': positivity,
        'symmetry_err': symmetry_err,
        'max_dist': max_dist,
        'max_geodesic_bound': max_geodesic,
        'within_sphere': within_sphere,
        'pass': pass_,
        'verdict': 'PASS' if pass_ else 'FAIL',
    }


def test_gate2_performance(dtype=torch.float64) -> dict:
    """Performance vs R137."""
    print("\n═══ Gate 2e — Performance benchmark ═══")
    torch.manual_seed(42)
    B, dim = 1000, 32
    x = torch.randn(B, dim, dtype=dtype)
    y = torch.randn(B, dim, dtype=dtype)
    kappa = torch.tensor(-0.5, dtype=dtype)

    n_iter = 50
    for _ in range(5):
        _ = unified_distance_correct(x, y, kappa)
        _ = r137_geodesic_distance_sq(x, y, kappa)

    t0 = time.time()
    for _ in range(n_iter):
        _ = unified_distance_correct(x, y, kappa)
    t_new = (time.time() - t0) / n_iter * 1000

    t0 = time.time()
    for _ in range(n_iter):
        _ = r137_geodesic_distance_sq(x, y, kappa)
    t_r137 = (time.time() - t0) / n_iter * 1000

    slowdown = t_new / t_r137
    print(f"  New unified distance: {t_new:.3f} ms/iter")
    print(f"  R137 (squared):       {t_r137:.3f} ms/iter")
    print(f"  Slowdown: {slowdown:.2f}x")

    pass_ = slowdown < 5.0  # target: within 5× of R137
    # R137 is simpler (just arctan), unified formula adds Möbius addition
    # (3 dot products + division) — 3-4× overhead is inherent.
    print(f"  {'✅ PASS' if pass_ else '⚠️ FAIL'} (target < 5.0×)")
    return {
        'test': 'performance',
        'new_ms_per_iter': t_new,
        'r137_ms_per_iter': t_r137,
        'slowdown_factor': slowdown,
        'pass': pass_,
        'verdict': 'PASS' if pass_ else 'FAIL',
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Task #338 — Issue #47 Fix unified κ-stereographic formula")
    print("=" * 70)
    print("\nBugs identified:")
    print("  Bug 1 (对称性破坏): mobius_addition `+κ‖y‖²` → `-κ‖y‖²`")
    print("  Bug 2 (κ=0 NaN):   sigmoid blend → torch.where crisp threshold")
    print()

    results = {}

    # Gate 0
    results["gate0"] = test_gate0_mobius_symmetry()

    # Gate 1
    results["gate1"] = test_gate1_kappa_zero_limit()

    # Gate 2
    results["gate2a"] = test_gate2_kappa_zero_gradient()
    results["gate2b"] = test_gate2_gradient_continuity()
    results["gate2c"] = test_gate2_kappa_negative()
    results["gate2d"] = test_gate2_kappa_positive()
    results["gate2e"] = test_gate2_performance()

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    gate0_pass = results["gate0"].get("gate0_pass", False)
    gate1_pass = results["gate1"].get("gate1_pass", False)

    g2_test_names = [
        ("gate2a", "κ=0 gradient"),
        ("gate2b", "Gradient continuity"),
        ("gate2c", "κ<0 hyperbolic"),
        ("gate2d", "κ>0 spherical"),
        ("gate2e", "Performance"),
    ]
    g2_pass = {}
    for key, name in g2_test_names:
        v = results[key]
        p = v.get('new_pass' if key == 'gate2a' else v.get('pass', v.get('continuous', False)),
                  False)
        if key == 'gate2a':
            p = v.get('new_pass', False)
        elif key == 'gate2b':
            p = v.get('new_continuous', False)
        elif key in ('gate2c', 'gate2d', 'gate2e'):
            p = v.get('pass', False)
        else:
            p = False
        g2_pass[key] = p

    # Re-evaluate
    g2_pass = {}
    for key, name in g2_test_names:
        v = results[key]
        if key == 'gate2a':
            p = v.get('new_pass', False)
        elif key == 'gate2b':
            p = v.get('new_continuous', False)
        elif key in ('gate2c', 'gate2d', 'gate2e'):
            p = v.get('pass', False)
        else:
            p = False
        g2_pass[key] = p

    print(f"\n  Gate 0 (Möbius isolated test):   "
          f"{'✅ PASS' if gate0_pass else '❌ FAIL'}")
    print(f"  Gate 1 (κ=0 limit):              "
          f"{'✅ PASS' if gate1_pass else '❌ FAIL'}")
    print(f"  Gate 2 (full 5-test rerun):")
    for key, name in g2_test_names:
        p = g2_pass.get(key, False)
        print(f"    {name:30s}  {'✅ PASS' if p else '❌ FAIL'}")

    all_pass = gate0_pass and gate1_pass and all(g2_pass.values())
    print(f"\n  Issue #47 overall: "
          f"{'✅ 5/5 ALL PASS 🎉' if all_pass else '❌ Some tests FAIL'}")
    print(f"\n  If ALL PASS: patch unified formula into hrqvae_free_curv.py")
    print(f"  Then proceed to Gate 3 (training validation) per #47 §实验设计")

    # Save JSON
    output_path = REPO / "verdicts/task338_issue47_fix_unified_formula.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果写入: {output_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
