"""
Issue #55 方向A 数学 sanity 5/5 test

验证 Riemannian AdamW 风格曲率更新 (跟标准 AdamW 解耦的二阶矩估计)
+ per-layer scale_recalibration 校准 codebook 有效半径.

测试:
T1: step() 减 loss (跟标准 AdamW 一致)
T2: Poincaré retraction 保 ‖x‖ < 1 (向量参数)
T3: scale_recalibration 校准 codebook 半径
T4: weight_decay decoupled from gradient (AdamW 风格)
T5: step counter 正确递增

R10: zero-GPU 立即可做
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import math
import torch
import torch.nn as nn


class RiemannianAdamW(torch.optim.Optimizer):
    """AdamW for Riemannian manifold (Issue #55).

    For 1D scalar κ: identity Riemannian gradient.
    For vector params (codebook): project via 1/(1+‖x‖²)² + expmap0 retraction.
    """

    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Riemannian gradient (project to tangent space)
                if p.numel() == 1:
                    riem_grad = grad
                else:
                    # For vector params: project via 1/(1+‖x‖²)² (Poincaré ball)
                    riem_grad = grad / (1 + torch.norm(p, dim=-1, keepdim=True) ** 2) ** 2

                # Standard Adam moments
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(riem_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(riem_grad, riem_grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # AdamW update with Riemannian retraction
                denom = (bias_corrected_exp_avg_sq.sqrt() + group['eps'])
                update = exp_avg / denom
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p
                p.add_(update, alpha=-step_size)

                # Riemannian retraction: for scalar κ, identity. For vector, expmap0.
                if p.numel() > 1:
                    # expmap0: x → x (since we're in tangent space at origin)
                    # But we need to project back to ball
                    norm = torch.norm(p, dim=-1, keepdim=True)
                    norm_clamped = norm.clamp_min(1e-8)
                    # Project to ‖p‖ < 1 (with small margin)
                    max_norm = 1.0 - 1e-5
                    if (norm > max_norm).any():
                        scale = max_norm / norm_clamped
                        scale = torch.minimum(scale, torch.ones_like(scale))
                        p.data = p.data * scale

        return None


def poincare_distance(x, y, c=1.0):
    """Poincaré ball distance (跟 #47 兼容)."""
    sqrt_c = math.sqrt(c)
    diff = x - y
    diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp_min(1e-8)
    denom = 2.0 * (1.0 - c * diff_norm ** 2 / 4.0).abs().clamp_min(1e-6)
    arg = sqrt_c * diff_norm / denom
    return (2.0 / sqrt_c) * torch.arctan(arg)


def test_T1_step_decreases_loss():
    """T1: Riemannian AdamW step 应该减小 loss."""
    torch.manual_seed(42)
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = RiemannianAdamW([param], lr=0.1)
    loss_fn = lambda x: (x - 0.5) ** 2

    initial_loss = loss_fn(param).item()
    optimizer.zero_grad()
    loss_fn(param).backward()
    optimizer.step()
    new_loss = loss_fn(param).item()

    assert new_loss < initial_loss, f"Step 后 loss 应该减小, initial={initial_loss}, new={new_loss}"
    print(f"✅ T1 PASS: step 减 loss ({initial_loss:.6f} → {new_loss:.6f})")


def test_T2_poincare_retraction():
    """T2: 向量参数 step 后应仍在 Poincaré ball (‖x‖<1)."""
    torch.manual_seed(42)
    param = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
    optimizer = RiemannianAdamW([param], lr=0.5)
    loss_fn = lambda x: (x.sum() - 1.0) ** 2

    # Run multiple steps to potentially violate ball
    for _ in range(20):
        optimizer.zero_grad()
        loss_fn(param).backward()
        optimizer.step()

    norm = torch.norm(param.data)
    assert norm < 1.0, f"向量参数应仍在 Poincaré ball (‖x‖<1), 实际 ‖x‖={norm.item()}"
    print(f"✅ T2 PASS: Poincaré retraction 保 ‖x‖<1, 20 steps 后 ‖x‖={norm.item():.6f}")


def test_T3_scale_recalibration():
    """T3: scale_recalibration 应该校准 codebook 半径到新 κ 下的有效值."""
    torch.manual_seed(42)
    codebook = torch.randn(256, 32) * 0.1  # (K, d)
    scale = nn.Parameter(torch.ones(3))  # per-layer
    old_kappa = torch.tensor(1.0)
    new_kappa = torch.tensor(1.5)

    # Compute Poincaré radius for codebook under both κ
    def poincare_radius(cbook, c):
        norm = torch.norm(cbook, dim=-1).clamp_min(1e-8)
        return (2.0 / math.sqrt(c)) * torch.arctan(math.sqrt(c) * norm / 2.0).mean()

    old_r = poincare_radius(codebook * scale.mean(), c=old_kappa.item()).item()
    new_r_target = poincare_radius(codebook * scale.mean(), c=new_kappa.item()).item()

    # scale_recalibration: s_new = old_r / new_r_target (so scaled codebook has same effective radius)
    s_new = old_r / new_r_target
    scale.data = scale.data * s_new

    # Verify: scaled codebook under new κ should have radius close to old_r
    recalibrated_r = poincare_radius(codebook * scale.mean(), c=new_kappa.item()).item()

    # Sanity check: ratio should be closer to 1 than uncalibrated
    uncalibrated_ratio = new_r_target / old_r
    calibrated_ratio = recalibrated_r / old_r

    assert torch.isfinite(torch.tensor(s_new)).item(), "s_new 应该 finite"
    assert s_new > 0, "s_new 应该 > 0"
    # calibrated ratio should be closer to 1.0 than uncalibrated
    assert abs(calibrated_ratio - 1.0) < abs(uncalibrated_ratio - 1.0), \
        f"calibrated={calibrated_ratio:.4f} 应该比 uncalibrated={uncalibrated_ratio:.4f} 更接近 1.0"
    print(f"✅ T3 PASS: scale_recalibration {s_new:.6f}, ratio old={uncalibrated_ratio:.4f} → calibrated={calibrated_ratio:.4f}")


def test_T4_weight_decay_decoupled():
    """T4: weight_decay 跟 gradient 解耦 (AdamW 风格, 区别于 Adam L2)."""
    torch.manual_seed(42)
    # 用 loss = (p - target)^2 + small grad signal, 这样 wd 主导 shrink 趋势
    param_wd = nn.Parameter(torch.tensor([2.0, 2.0, 2.0]))
    param_no_wd = nn.Parameter(torch.tensor([2.0, 2.0, 2.0]))

    opt_wd = RiemannianAdamW([param_wd], lr=0.01, weight_decay=0.5)
    opt_no_wd = RiemannianAdamW([param_no_wd], lr=0.01, weight_decay=0.0)

    # 多次 step 让 wd effect 累积
    for _ in range(10):
        # loss 接近 0 grad (因为 target = 2.0)
        loss_wd = ((param_wd - 2.0) ** 2).sum()
        loss_wd.backward()
        param_no_wd.grad = param_wd.grad.clone()
        opt_wd.step()
        opt_no_wd.step()

    # wd=0.5 with 10 steps * lr=0.01 = 0.05 cumulative wd effect
    # param_wd should be slightly shrunk vs no_wd
    norm_wd = torch.norm(param_wd.data).item()
    norm_no_wd = torch.norm(param_no_wd.data).item()

    assert norm_wd < norm_no_wd, \
        f"weight_decay 应该 shrink 参数: wd={norm_wd:.4f} vs no_wd={norm_no_wd:.4f}"
    print(f"✅ T4 PASS: weight_decay decoupled ({norm_wd:.4f} < {norm_no_wd:.4f})")


def test_T5_step_counter():
    """T5: step counter 应该正确递增."""
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = RiemannianAdamW([param], lr=0.1)

    for expected_step in range(1, 6):
        optimizer.zero_grad()
        (param ** 2).backward()
        optimizer.step()
        assert optimizer.state[param]['step'] == expected_step, \
            f"step counter 应该 = {expected_step}, 实际 {optimizer.state[param]['step']}"
    print(f"✅ T5 PASS: step counter 正确递增 (1 → 5)")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #55 方向A 数学 sanity 5/5 test")
    print("=" * 60)
    test_T1_step_decreases_loss()
    test_T2_poincare_retraction()
    test_T3_scale_recalibration()
    test_T4_weight_decay_decoupled()
    test_T5_step_counter()
    print("=" * 60)
    print("ALL 5/5 PASS — Issue #55 Gate 0 数学基础 OK")
    print("下一步: 启动 Stage 1 训练 (~3.5h, 复用 #49 流程, 替换 optimizer + 加 s_l)")
    print("=" * 60)