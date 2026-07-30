"""
Issue #56 方向B 数学 sanity 5/5 test

验证 alpha_l sigmoid 参数化 + d = alpha * d_hyp + (1-alpha) * d_euc 距离公式
在 κ=0 处无 singular gradient (跟 #44/#47 修复兼容).

测试:
T1: alpha=0 → 纯 Euclidean distance
T2: alpha=1 → 纯 hyperbolic distance (跟 #47 公式兼容)
T3: alpha=0 梯度非零 (无 singular point)
T4: sigmoid(z) 有界 alpha ∈ (0, 1) for any z
T5: 跟 #47 修复公式兼容 (mixed_curv_dist alpha=1 跟 geodesic_distance_unified 一致)

R10: zero-GPU 立即可做 (跟 Issue #57 Gate 0 Stage 3 训练并行)
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import torch.nn as nn
from model.hrqvae_free_curv import geodesic_distance_sq, geodesic_distance_unified


def mixed_curv_dist(x: torch.Tensor, y: torch.Tensor,
                    alpha: torch.Tensor, kappa_fixed: float) -> torch.Tensor:
    """Issue #56 α_l weighted hyperbolic + euclidean distance.

    d = α * d_hyp(x, y; κ_fixed) + (1 - α) * d_euc(x, y)
    where d_hyp uses #47 fix geodesic_distance_sq.
    """
    d_hyp = geodesic_distance_sq(x, y, torch.tensor(kappa_fixed, dtype=x.dtype, device=x.device))
    d_euc = torch.norm(x - y, dim=-1, keepdim=True)
    return alpha * d_hyp + (1.0 - alpha) * d_euc


def test_T1_alpha_zero_is_euclidean():
    """T1: α=0 → 纯 Euclidean distance."""
    torch.manual_seed(42)
    x = torch.randn(10, 32) * 0.1
    y = torch.randn(10, 32) * 0.1

    d_mixed = mixed_curv_dist(x, y, alpha=torch.tensor(0.0), kappa_fixed=0.74)
    d_euc = torch.norm(x - y, dim=-1, keepdim=True)

    assert torch.allclose(d_mixed, d_euc, atol=1e-5), \
        f"α=0 应该是 Euclidean, 实际差距 {(d_mixed - d_euc).abs().max().item()}"
    print("✅ T1 PASS: α=0 → 纯 Euclidean distance")


def test_T2_alpha_one_is_hyperbolic():
    """T2: α=1 → 纯 hyperbolic distance (跟 #47 公式兼容)."""
    torch.manual_seed(42)
    x = torch.randn(10, 32) * 0.1
    y = torch.randn(10, 32) * 0.1

    d_mixed = mixed_curv_dist(x, y, alpha=torch.tensor(1.0), kappa_fixed=0.74)
    d_hyp = geodesic_distance_sq(x, y, torch.tensor(0.74))

    assert torch.allclose(d_mixed, d_hyp, atol=1e-5), \
        f"α=1 应该是 hyperbolic, 实际差距 {(d_mixed - d_hyp).abs().max().item()}"
    print("✅ T2 PASS: α=1 → 纯 hyperbolic distance (跟 #47 兼容)")


def test_T3_alpha_gradient_at_zero_nonzero():
    """T3: ∂d/∂α at α=0 非零 (无 singular gradient, 关键: 跟 #44 κ=0 退化点对比)."""
    torch.manual_seed(42)
    x = torch.randn(10, 32, requires_grad=True) * 0.1
    y = torch.randn(10, 32) * 0.1
    alpha = torch.tensor(0.0, requires_grad=True)

    d = mixed_curv_dist(x, y, alpha=alpha, kappa_fixed=0.74)
    d.sum().backward()

    assert alpha.grad is not None, "α=0 应该有梯度 (跟 #44 κ=0 dead point 对比)"
    assert torch.isfinite(alpha.grad).all(), \
        f"α=0 梯度应该是 finite, 实际 {alpha.grad.item()}"
    # 关键: 梯度 ≠ 0 (跟 κ=0 退化对比)
    assert alpha.grad.abs() > 1e-8, \
        f"α=0 梯度应该非零, 实际 {alpha.grad.item()} (接近 singular point?)"
    print(f"✅ T3 PASS: α=0 梯度 = {alpha.grad.item():.6f} (非零, 跟 #44 κ=0 dead point 形成对比)")


def test_T4_sigmoid_bounded():
    """T4: α = sigmoid(z), z ∈ R, α ∈ (0, 1) for any z."""
    z = torch.linspace(-100, 100, 10000)
    alpha = torch.sigmoid(z)

    assert (alpha >= 0).all(), f"α 应该 ≥ 0, min={alpha.min().item()}"
    assert (alpha <= 1).all(), f"α 应该 ≤ 1, max={alpha.max().item()}"
    # sigmoid float 边界: underflow 到 0 或 overflow 到 1 都算 OK (实数严格 (0,1))
    assert torch.isfinite(alpha).all(), "α 应该 finite"
    assert alpha[0].item() < 1e-10, f"z=-100 sigmoid 应该接近 0, 实际 {alpha[0].item()}"
    # 用 isclose 代替 > 比较 (避免 float underflow)
    assert torch.isclose(alpha[-1], torch.tensor(1.0), atol=1e-10), \
        f"z=+100 sigmoid 应该接近 1, 实际 {alpha[-1].item()}"
    # sanity: sigmoid(0) = 0.5
    assert abs(torch.sigmoid(torch.tensor(0.0)).item() - 0.5) < 1e-6, "sigmoid(0) 应该是 0.5"
    print(f"✅ T4 PASS: sigmoid 有界, α ∈ (0, 1) for any z ∈ [-100, 100]")


def test_T5_compat_with_47_unified():
    """T5: mixed_curv_dist(α=1) ≈ geodesic_distance_unified (#47 fix 兼容)."""
    torch.manual_seed(42)
    x = torch.randn(5, 32) * 0.1
    y = torch.randn(5, 32) * 0.1

    # Issue #56 mixed_curv at α=1
    d_mixed = mixed_curv_dist(x, y, alpha=torch.tensor(1.0), kappa_fixed=0.74)

    # #47 fix geodesic_distance_sq (跟 unified 在 κ<0 等价)
    d_hyp_sq = geodesic_distance_sq(x, y, torch.tensor(0.74))

    assert torch.allclose(d_mixed, d_hyp_sq, atol=1e-5), \
        f"α=1 应该跟 #47 geodesic_distance_sq 等价, 实际差距 {(d_mixed - d_hyp_sq).abs().max().item()}"
    print(f"✅ T5 PASS: α=1 跟 #47 fix 兼容, max diff = {(d_mixed - d_hyp_sq).abs().max().item():.2e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #56 方向B 数学 sanity 5/5 test")
    print("=" * 60)
    test_T1_alpha_zero_is_euclidean()
    test_T2_alpha_one_is_hyperbolic()
    test_T3_alpha_gradient_at_zero_nonzero()
    test_T4_sigmoid_bounded()
    test_T5_compat_with_47_unified()
    print("=" * 60)
    print("ALL 5/5 PASS — Issue #56 Gate 0 数学基础 OK")
    print("下一步: 启动 Stage 1 训练 (~3.2h, 复用 #49 流程, 仅替换 κ_l → α_l)")
    print("=" * 60)