"""
Task #333 — Issue #42 κ-Stereographic 统一公式 (MCKG Table 1 风格)

实现真正的 κ-stereographic 统一距离公式 (Sala 2018 / MCKG Table 1),
替代 R137 的 abs-based Berman-Metzler 2020 / Chlenski 2020 写法.

关键差异 (Issue #42 核心主张):
- R137: 用 .abs() 强制归零 κ=0 处梯度 → θ=0 死点 (数学 fixed point)
- MCKG Table 1: 用 tan_κ⁻¹ 的 Taylor 极限 ($x - \\frac{1}{3}\\kappa x^3$) 在 κ=0 处连续可导
- 距离公式 $d_\\kappa(x,y) = 2\\tan_\\kappa^{-1}(\\| -x \\oplus_\\kappa y \\|_2)$

注意: 这是不同的 κ-Stereographic 模型 (Sala 2018 vs Berman-Metzler 2020),
两者都有效, 只是归一化不同:
- R137: d_{κ=0} = ‖x-y‖ (1x Euclidean scale)
- MCKG Table 1: d_{κ=0} = 2‖x-y‖ (2x Euclidean scale)

主要测试:
1. κ=0 处 d/dκ ≠ 0 (vs R137 d/dκ = 0, θ=0 死点证据)
2. κ sweep 跨 [-1, 1] 梯度连续 (vs R137 sign flip)
3. κ<0 / κ>0 行为符合 κ-Stereographic (sphere/ball constraints)
"""
from __future__ import annotations
import math
import torch
import numpy as np
import json
import time
from pathlib import Path

EPS = 1e-15


def mobius_addition(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """MCKG Table 1 Möbius addition (单条 closed-form 适用于所有 κ).

    Formula (corrected from MCKG Table 1 - paper has x²/y² typo):
        x ⊕_κ y = [(1 - 2κ⟨x,y⟩ + κ‖y‖²)·x + (1 + κ‖x‖²)·y]
                  / [1 - 2κ⟨x,y⟩ + κ²·‖x‖²·‖y‖²]

    Note: MCKG Table 1 as written has (1 + κ‖y‖²)·y in second term; we correct
    to (1 + κ‖x‖²)·y to ensure symmetry d(x,y) = d(y,x).

    Limit κ=0 → x + y (regular Euclidean addition).
    """
    x_dot_y = (x * y).sum(dim=-1, keepdim=True)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)

    # Corrected formula: x² in second coefficient, -κ‖y‖² (not +κ‖y‖²)
    # This matches HG-Rec.md eq 1 for Poincaré κ=-c: -(κ)‖y‖² → +c‖y‖²
    num_term1 = (1.0 - 2.0 * kappa * x_dot_y - kappa * y_norm_sq) * x
    num_term2 = (1.0 + kappa * x_norm_sq) * y
    numerator = num_term1 + num_term2

    denominator = (1.0 - 2.0 * kappa * x_dot_y
                  + (kappa ** 2) * x_norm_sq * y_norm_sq).clamp(min=EPS)

    return numerator / denominator


def tan_kappa_inverse(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Unified tan_κ⁻¹ function (MCKG 论文 §3.1 补全).

    Definition:
        κ > 0:  κ^{-1/2} · atan(x · κ^{1/2})
        κ = 0:  x - (1/3)κ x³  (Taylor 极限, 保证 C¹ 连续)
        κ < 0:  |κ|^{-1/2} · atanh(x · |κ|^{1/2})

    关键: κ=0 分支用 Taylor 极限 (而非退化到 atan(0)/0),
    这样 d/dκ 在 κ=0 处非零 (vs R137 用 .abs() 强行归零).

    实现策略 (避免 autograd 0/0 NaN):
    - 用一个 smooth gate function g(|κ|): g=0 当 |κ|<threshold, g=1 当 |κ|>=threshold
    - g(|κ|) · closed_form + (1-g(|κ|)) · Taylor_expansion
    - closed_form 在 |κ|<threshold 时实际不参与 (因为乘以 0)
    - 关键: 不能用 .float() 做离散门 (会断梯度), 用 sigmoid 实现 smooth gate
    """
    abs_kappa = kappa.abs()

    # Always compute Taylor (always safe, polynomial in κ)
    # Higher-order terms for better accuracy:
    # tan_κ⁻¹(x) = x - (κ/3)x³ + (2κ²/15)x⁵ - (17κ³/315)x⁷ + O(κ⁴)
    taylor = x - (1.0 / 3.0) * kappa * x ** 3 + (2.0 / 15.0) * (kappa ** 2) * x ** 5

    # Closed-form (computed but masked when |κ| is small)
    sqrt_abs_k = torch.sqrt(abs_kappa + EPS)
    pos_branch = torch.atan(x * sqrt_abs_k) / sqrt_abs_k
    neg_input = (x * sqrt_abs_k).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    neg_branch = torch.atanh(neg_input) / sqrt_abs_k

    # Sign selection: κ>0 → pos_branch, κ<0 → neg_branch
    is_pos = (kappa > 0).float()
    closed_form = is_pos * pos_branch + (1.0 - is_pos) * neg_branch

    # ── BUG FIX (2026-07-30): crisp threshold (torch.where) instead of sigmoid blend ──
    # Sigmoid causes NaN at κ=0 because atan(0)/0 = 0/0 propagates through gate.
    # torch.where masks the closed-form gradient at κ=0, so only the Taylor gradient flows.
    threshold = 1e-4
    return torch.where(abs_kappa >= threshold, closed_form, taylor)


def unified_k_stereographic_distance(x: torch.Tensor, y: torch.Tensor,
                                     kappa: torch.Tensor) -> torch.Tensor:
    """MCKG Table 1 unified κ-Stereographic geodesic distance.

    d_κ(x, y) = 2 · tan_κ⁻¹(|| -x ⊕_κ y ||)

    Args:
        x: (B, d) 或 (d,)
        y: (B, d) 或 (d,)
        kappa: scalar tensor
    Returns:
        (B,) 或 scalar - geodesic distance d_κ
    """
    # Step 1: -x ⊕_κ y
    neg_x = -x
    diff = mobius_addition(neg_x, y, kappa)

    # Step 2: || -x ⊕_κ y ||_2
    # For κ<0: max norm = 1/√|κ| (Poincaré ball boundary)
    # For κ>0: max norm = 1/√κ (sphere, depends on parameterization)
    # For κ=0: no boundary (Euclidean)
    # Use safe clamp: avoid torch.where (which causes autograd NaN)
    abs_kappa = kappa.abs().clamp(min=EPS)
    # boundary = 1/√|κ|, but if κ=0, default to large value (no actual boundary)
    # Use abs_kappa + EPS in sqrt to avoid 1/0
    boundary = 1.0 / torch.sqrt(abs_kappa)  # safe due to clamp above
    max_norm = boundary - 1e-6
    diff_norm = diff.norm(dim=-1).clamp(max=max_norm).clamp(min=EPS)

    # Step 3: 2 · tan_κ⁻¹(||·||)
    return 2.0 * tan_kappa_inverse(diff_norm, kappa)


# ────────────────────────────────────────────────────────────
# R137 复刻 (Berman-Metzler 2020 / Chlenski 2020, 不同 κ-Stereo 归一化)
# ────────────────────────────────────────────────────────────
def r137_geodesic_distance_sq(x: torch.Tensor, y: torch.Tensor,
                               kappa: torch.Tensor) -> torch.Tensor:
    """R137 实际实现 (从 HG-Rec/model/hrqvae_free_curv.py:45-76 复刻).

    单条 closed-form 但用 .abs() 在 κ=0 不可导 → θ=0 死点根因.
    """
    kappa_abs = kappa.abs().clamp(min=1e-8)
    sqrt_kappa = torch.sqrt(kappa_abs)
    diff = x - y
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    diff_norm_sq = diff_norm ** 2
    denom = 2.0 * (1.0 - kappa * diff_norm_sq / 4.0).abs().clamp_min(1e-6)
    arg = sqrt_kappa * diff_norm / denom
    d = (2.0 / sqrt_kappa) * torch.arctan(arg)
    return d ** 2  # 返回 d² (跟原函数一致)


# ────────────────────────────────────────────────────────────
# 验证测试
# ────────────────────────────────────────────────────────────
def test_kappa_zero_gradient(dtype=torch.float64):
    """**核心测试**: 验证 κ=0 处新公式梯度 ≠ 0 (vs R137 = 0).

    这是 Issue #42 的核心主张: 真正的统一公式应该让 θ=0 init 的 κ 参数
    能接收到梯度, 避免数学 fixed point.
    """
    print("\n=== 核心 Test: κ=0 处梯度可导性 ===")
    torch.manual_seed(42)
    n = 50
    d = 8
    x = torch.randn(n, d, dtype=dtype)
    y = torch.randn(n, d, dtype=dtype)

    # New formula at κ=0
    kappa_var = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    d_new = unified_k_stereographic_distance(x, y, kappa_var)
    d_new.sum().backward()
    new_grad = kappa_var.grad.item()

    # R137 at κ=0
    kappa_var_r137 = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    d_r137 = r137_geodesic_distance_sq(x, y, kappa_var_r137)
    d_r137.sum().backward()
    r137_grad = kappa_var_r137.grad.item()

    # 数值梯度 (新公式) - finite difference 检查
    eps = 1e-5
    d_plus = unified_k_stereographic_distance(x, y, torch.tensor(eps, dtype=dtype))
    d_minus = unified_k_stereographic_distance(x, y, torch.tensor(-eps, dtype=dtype))
    numerical_grad = (d_plus.sum() - d_minus.sum()).item() / (2 * eps)

    print(f"  New formula ∂d/∂κ at κ=0 (autograd): {new_grad:.4e}")
    print(f"  New formula ∂d/∂κ at κ=0 (finite diff): {numerical_grad:.4e}")
    print(f"  R137 ∂d²/∂κ at κ=0 (autograd): {r137_grad:.4e}  (应≈0, 死点证据)")

    # 关键判定: 新公式梯度应该显著非零, R137 应该 ≈ 0
    new_pass = abs(new_grad) > 1e-3 and abs(numerical_grad) > 1e-3
    r137_dead_point = abs(r137_grad) < 1e-3  # R137 在 κ=0 是数学死点

    print(f"  New formula gradient non-zero: {new_pass}")
    print(f"  R137 dead point at κ=0: {r137_dead_point}")

    return {
        'test': 'kappa_zero_gradient',
        'new_grad_autograd': new_grad,
        'new_grad_finite_diff': numerical_grad,
        'r137_grad_at_kappa0': r137_grad,
        'new_pass': new_pass,
        'r137_dead_point': r137_dead_point,
        'verdict': 'PASS' if new_pass and r137_dead_point else 'FAIL',
    }


def test_gradient_continuity(dtype=torch.float64):
    """验证 κ sweep 中新公式梯度连续 (vs R137 在 κ=0 sign flip)."""
    print("\n=== Test: κ sweep 梯度连续性 [-1, 1] ===")
    torch.manual_seed(42)
    d = 8
    x = torch.randn(20, d, dtype=dtype)
    y = torch.randn(20, d, dtype=dtype)

    # Sample κ values across range, including κ=0
    kappas = torch.linspace(-0.99, 0.99, 21, dtype=dtype)
    new_grads = []
    r137_grads = []

    for k_val in kappas:
        # New formula
        k_var = torch.tensor(k_val.item(), dtype=dtype, requires_grad=True)
        d_new = unified_k_stereographic_distance(x, y, k_var)
        d_new.sum().backward()
        new_grads.append(k_var.grad.item())

        # R137
        k_var_r = torch.tensor(k_val.item(), dtype=dtype, requires_grad=True)
        d_r137 = r137_geodesic_distance_sq(x, y, k_var_r)
        d_r137.sum().backward()
        r137_grads.append(k_var_r.grad.item())

    # 关键指标: 跨 κ=0 梯度连续性
    # κ=0 在 21 点中是 index 10
    zero_idx = 10
    new_left = new_grads[zero_idx - 1]
    new_at_zero = new_grads[zero_idx]
    new_right = new_grads[zero_idx + 1]

    r137_left = r137_grads[zero_idx - 1]
    r137_at_zero = r137_grads[zero_idx]
    r137_right = r137_grads[zero_idx + 1]

    # 新公式: 在 κ=0 附近梯度变化应该连续 (相邻点差值小)
    new_jump_left = abs(new_at_zero - new_left)
    new_jump_right = abs(new_right - new_at_zero)
    new_continuous = new_jump_left < 0.1 * abs(new_at_zero) and new_jump_right < 0.1 * abs(new_at_zero)

    # R137: 在 κ=0 由于 .abs() 应该突变
    r137_jump_left = abs(r137_at_zero - r137_left)
    r137_jump_right = abs(r137_right - r137_at_zero)
    # R137 在 κ=0 应该有明显 jump (sign 或 magnitude)
    r137_discontinuous = r137_jump_left > 0.01 * abs(r137_left) or r137_jump_right > 0.01 * abs(r137_right)

    print(f"  κ sweep 21 points:")
    print(f"    New formula ∂d/∂κ at κ=0: {new_at_zero:.4e}  (左 {new_left:.4e}, 右 {new_right:.4e})")
    print(f"    R137 ∂d²/∂κ at κ=0: {r137_at_zero:.4e}  (左 {r137_left:.4e}, 右 {r137_right:.4e})")
    print(f"    New formula continuous at κ=0: {new_continuous}")
    print(f"    R137 discontinuous at κ=0: {r137_discontinuous} (死点特征)")

    # Sign change check
    new_sign_change = (new_left > 0) != (new_right > 0)
    r137_sign_change = (r137_left > 0) != (r137_right > 0)
    print(f"    New formula sign change: {new_sign_change} (应该: 连续, 无 sign change)")
    print(f"    R137 sign change: {r137_sign_change} (应该: sign change 因 .abs() 翻转)")

    return {
        'test': 'gradient_continuity',
        'new_continuous_at_kappa0': new_continuous,
        'r137_discontinuous_at_kappa0': r137_discontinuous,
        'new_at_kappa0': new_at_zero,
        'r137_at_kappa0': r137_at_zero,
        'new_jump_left': new_jump_left,
        'r137_jump_left': r137_jump_left,
    }


def test_kappa_negative_consistency(dtype=torch.float64):
    """验证 κ<0 时新公式的内部一致性 (符合 κ-Stereographic 性质)."""
    print("\n=== Test: κ<0 双曲空间性质 ===")
    torch.manual_seed(42)
    n = 100
    d = 8
    # 限制 ‖x‖, ‖y‖ < 1/√|κ| (Poincaré ball constraint)
    kappa = torch.tensor(-1.0, dtype=dtype)
    max_norm = 1.0 / 1.0 - 0.1  # 0.9 for c=1

    x = (torch.randn(n, d, dtype=dtype) * 0.3)
    y = (torch.randn(n, d, dtype=dtype) * 0.3)

    # 新公式
    d_new = unified_k_stereographic_distance(x, y, kappa)

    # 基本性质检查
    # 1. d(x, x) = 0
    d_xx = unified_k_stereographic_distance(x[:10], x[:10], kappa)
    self_dist_err = (d_xx).abs().max().item()

    # 2. d(x, y) > 0 for x ≠ y
    d_xy = unified_k_stereographic_distance(x, y, kappa)
    positivity = (d_xy > 0).all().item()

    # 3. 对称性 d(x, y) = d(y, x)
    d_yx = unified_k_stereographic_distance(y, x, kappa)
    symmetry_err = (d_new - d_yx).abs().max().item()

    # 4. d 与 ‖x-y‖ 单调性 (远点应该距离更大)
    diff_norms = (x - y).norm(dim=-1)
    # 取 10 个点, 找最远和最近
    sorted_idx = diff_norms.argsort()
    closest_dist = d_new[sorted_idx[:10]].mean().item()
    farthest_dist = d_new[sorted_idx[-10:]].mean().item()
    monotonic = farthest_dist > closest_dist

    print(f"  κ = -1, n = {n}")
    print(f"  d(x, x) max = {self_dist_err:.2e} (应该=0)")
    print(f"  All d(x, y) > 0: {positivity}")
    print(f"  Symmetry max |d(x,y) - d(y,x)| = {symmetry_err:.2e}")
    print(f"  Monotonic: closest mean = {closest_dist:.4f}, farthest mean = {farthest_dist:.4f}")
    print(f"  Monotonicity (farther > closer): {monotonic}")

    return {
        'test': 'kappa_negative_consistency',
        'self_dist_err': self_dist_err,
        'all_positive': positivity,
        'symmetry_err': symmetry_err,
        'monotonic': monotonic,
        'pass': self_dist_err < 1e-4 and positivity and symmetry_err < 1e-4 and monotonic,
    }


def test_kappa_positive_consistency(dtype=torch.float64):
    """验证 κ>0 时新公式的内部一致性 (球面性质)."""
    print("\n=== Test: κ>0 球面性质 ===")
    torch.manual_seed(42)
    n = 50
    d = 8
    kappa = torch.tensor(0.5, dtype=dtype)
    # 球面约束: ‖x‖² = 1/κ (球面半径)
    sphere_radius = (1.0 / kappa).sqrt()

    # 生成 sphere 上的点
    x_raw = torch.randn(n, d, dtype=dtype)
    x = x_raw / x_raw.norm(dim=-1, keepdim=True) * sphere_radius
    y_raw = torch.randn(n, d, dtype=dtype)
    y = y_raw / y_raw.norm(dim=-1, keepdim=True) * sphere_radius

    d_new = unified_k_stereographic_distance(x, y, kappa)

    # 球面距离上界: π/√κ (大圆距离)
    max_geodesic = (math.pi / kappa.sqrt()).item()

    # 基本性质
    d_xx = unified_k_stereographic_distance(x[:10], x[:10], kappa)
    self_dist_err = (d_xx).abs().max().item()

    d_xy = unified_k_stereographic_distance(x, y, kappa)
    positivity = (d_xy > 0).all().item()

    symmetry_err = (d_new - unified_k_stereographic_distance(y, x, kappa)).abs().max().item()

    # 在球面上最大距离应不超过 π/√κ
    max_dist = d_xy.max().item()
    within_sphere = max_dist <= max_geodesic + 1e-3

    print(f"  κ = 0.5, sphere radius = {sphere_radius.item():.4f}")
    print(f"  d(x, x) max = {self_dist_err:.2e}")
    print(f"  All d(x, y) > 0: {positivity}")
    print(f"  Symmetry max |d(x,y) - d(y,x)| = {symmetry_err:.2e}")
    print(f"  Max distance: {max_dist:.4f} ≤ π/√κ = {max_geodesic:.4f}: {within_sphere}")

    return {
        'test': 'kappa_positive_consistency',
        'self_dist_err': self_dist_err,
        'all_positive': positivity,
        'symmetry_err': symmetry_err,
        'max_dist': max_dist,
        'max_geodesic_bound': max_geodesic,
        'within_sphere': within_sphere,
        'pass': self_dist_err < 1e-4 and positivity and symmetry_err < 1e-4 and within_sphere,
    }


def test_performance(dtype=torch.float64):
    """基准测试: 新公式 vs R137 计算开销."""
    print("\n=== Test: 性能基准 (B=1000, d=32) ===")
    torch.manual_seed(42)
    B, d = 1000, 32
    x = torch.randn(B, d, dtype=dtype)
    y = torch.randn(B, d, dtype=dtype)
    kappa = torch.tensor(-0.5, dtype=dtype)

    n_iter = 50

    # Warm-up
    for _ in range(5):
        _ = unified_k_stereographic_distance(x, y, kappa)
        _ = r137_geodesic_distance_sq(x, y, kappa)

    # Benchmark new
    t0 = time.time()
    for _ in range(n_iter):
        _ = unified_k_stereographic_distance(x, y, kappa)
    t_new = (time.time() - t0) / n_iter * 1000

    # Benchmark R137
    t0 = time.time()
    for _ in range(n_iter):
        _ = r137_geodesic_distance_sq(x, y, kappa)
    t_r137 = (time.time() - t0) / n_iter * 1000

    print(f"  New formula (Möbius + tan_κ⁻¹): {t_new:.3f} ms/iter")
    print(f"  R137 (直接差 + arctan): {t_r137:.3f} ms/iter")
    print(f"  Slowdown: {t_new / t_r137:.2f}x")

    return {
        'test': 'performance',
        'new_ms_per_iter': t_new,
        'r137_ms_per_iter': t_r137,
        'slowdown_factor': t_new / t_r137,
    }


def main():
    """Run all tests + write verdict JSON."""
    print("=" * 70)
    print("Task #333 / Issue #42 — MCKG Table 1 κ-Stereographic 统一公式验证")
    print("=" * 70)

    results = {}
    results['test_kappa_zero_gradient'] = test_kappa_zero_gradient()
    results['test_gradient_continuity'] = test_gradient_continuity()
    results['test_kappa_negative'] = test_kappa_negative_consistency()
    results['test_kappa_positive'] = test_kappa_positive_consistency()
    results['test_performance'] = test_performance()

    # Summary
    print("\n" + "=" * 70)
    print("汇总:")
    print("=" * 70)
    grad_test = results['test_kappa_zero_gradient']
    print(f"  核心 Issue #42 验证 (κ=0 梯度):")
    print(f"    New formula ∂d/∂κ at κ=0: {grad_test['new_grad_autograd']:.4e}")
    print(f"    R137 ∂d²/∂κ at κ=0: {grad_test['r137_grad_at_kappa0']:.4e}")
    print(f"    New formula gradient non-zero: {grad_test['new_pass']}")
    print(f"    R137 dead point confirmed: {grad_test['r137_dead_point']}")
    print(f"    **Issue #42 核心主张: {'CONFIRMED ✓' if grad_test['new_pass'] and grad_test['r137_dead_point'] else 'NOT CONFIRMED ✗'}**")
    print()

    continuity = results['test_gradient_continuity']
    print(f"  κ sweep 梯度连续性:")
    print(f"    New formula continuous at κ=0: {continuity['new_continuous_at_kappa0']}")
    print(f"    R137 discontinuous at κ=0: {continuity['r137_discontinuous_at_kappa0']}")

    n_internal_pass = sum(1 for k in ['test_kappa_negative', 'test_kappa_positive']
                          if results[k].get('pass', False))
    print(f"  κ-Stereographic 内部一致性 tests: {n_internal_pass}/2 pass")

    # Save JSON
    output_path = Path(__file__).parent.parent / 'verdicts' / 'task333_issue42_unified_formula_verification.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果写入: {output_path}")


if __name__ == '__main__':
    main()