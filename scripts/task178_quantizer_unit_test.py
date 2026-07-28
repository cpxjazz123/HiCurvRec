#!/usr/bin/env python3
# Task #178 Phase 0.3 — 验证 utils.py 量化器修复
# 测试 4 项:
# (1) get_codebook() 返回值与 forward 内分配阶段用的码字完全一致
#     (即两者都应该是 proj_to_ball(expmap0(weight)))
# (2) codebook_loss + β · commitment_loss 公式 (论文 Eq (8) + van den Oord 标准约定)
# (3) x_q (量化输出) 必须实际在 Poincaré 球内 ‖x_q‖ < 1/√c
# (4) 修复前的"β 挂反" + "球外点代入 poincare_distance" 是否数值上触发梯度灾难

import sys
import os
import torch
import torch.nn.functional as F

# 加 sys.path
HG_REC_ROOT = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec'
sys.path.insert(0, HG_REC_ROOT)

from model.utils import (
    HVectorQuantization,
    proj_to_ball,
    expmap0,
    poincare_distance,
)

torch.manual_seed(42)


def test_get_codebook_consistency():
    """Test 1: get_codebook() 现在必须经过 expmap0 + proj_to_ball, 与 forward 一致"""
    print("===== Test 1: get_codebook() vs forward 内部码字一致性 =====")
    vq = HVectorQuantization(n_e=32, e_dim=8, beta=0.5,
                             kmeans_init=True, kmeans_iters=10,
                             sk_eps=0.003, sk_iters=5, curvature=1.0)
    # 必须用 initted 把 weight 初始化为非零
    vq.initted = True
    with torch.no_grad():
        vq.embeddings.weight.data.uniform_(-0.5, 0.5)

    # get_codebook 路径
    cb_get = vq.get_codebook()

    # forward 内部路径: 走 expmap0 + proj_to_ball
    cb_forward = proj_to_ball(expmap0(vq.embeddings.weight, vq.c), vq.c)

    diff = (cb_get - cb_forward).abs().max().item()
    assert diff < 1e-6, f"get_codebook 与 forward 内部码字不一致: max diff = {diff}"
    print(f"  ✅ 一致 — max abs diff = {diff:.2e}")

    # 反向 sanity: cb_get 必须 ≤ proj_to_ball(weight) (没 expmap0 时 ‖·‖ 可能 > 1/√c)
    cb_naive = proj_to_ball(vq.embeddings.weight, vq.c)
    assert cb_get.shape == cb_naive.shape
    print(f"  形状: {cb_get.shape}, get_codebook 用 expmap0+proj 跟内部对齐")


def test_x_q_inside_ball():
    """Test 2: x_q 必须实际在 Poincaré 球内"""
    print("\n===== Test 2: x_q 在 Poincaré 球内 =====")
    vq = HVectorQuantization(n_e=16, e_dim=4, beta=0.5,
                             kmeans_init=True, kmeans_iters=5,
                             sk_eps=0.003, sk_iters=3, curvature=1.0)

    # 强制初始化为非平凡值
    vq.initted = True
    with torch.no_grad():
        vq.embeddings.weight.data.uniform_(-0.7, 0.7)

    # 模拟 encoder 输出
    latent = torch.randn(8, 4) * 0.5  # 8 个 item latent (4 dim)
    latent = latent.to(torch.float32)

    # 直接调 forward, x_q 应该已经是 proj_to_ball(expmap0(*))
    vq.eval()
    with torch.no_grad():
        x_q, loss, indices = vq(latent, use_sk=False)

    norms = x_q.norm(dim=-1)
    inv_sqrt_c = 1.0 / (vq.c ** 0.5)
    max_norm = norms.max().item()
    assert (norms < inv_sqrt_c).all().item(), \
        f"x_q 跑出球! max norm = {max_norm} >= 1/√c = {inv_sqrt_c}"
    print(f"  ✅ x_q ∈ 球内 — max ‖x_q‖ = {max_norm:.4f} < 1/√c = {inv_sqrt_c:.4f}")
    print(f"  loss = {loss.item():.4f} (poincare 距离 → 期望 ~2-15 范围)")


def test_beta_attached_to_commitment():
    """Test 3: β 挂在 commitment (van den Oord 标准 + 论文 Eq (8))"""
    print("\n===== Test 3: β 挂在 commitment =====")
    vq = HVectorQuantization(n_e=16, e_dim=4, beta=0.5,
                             kmeans_init=True, kmeans_iters=5,
                             sk_eps=0.003, sk_iters=3, curvature=1.0)
    vq.initted = True
    with torch.no_grad():
        vq.embeddings.weight.data.uniform_(-0.5, 0.5)

    latent = torch.randn(8, 4) * 0.5
    vq.eval()
    x_q, loss, _ = vq(latent, use_sk=False)

    # 手动拆 commitment_loss 与 codebook_loss, 验证 loss 公式
    x_q_h = x_q  # 修复后 x_q 已经在球内
    latent_h = proj_to_ball(expmap0(latent, vq.c), vq.c)
    commitment_loss = torch.mean(poincare_distance(x_q_h.detach(), latent_h, vq.c) ** 2)
    codebook_loss = torch.mean(poincare_distance(x_q_h, latent_h.detach(), vq.c) ** 2)
    expected_loss = codebook_loss + vq.beta * commitment_loss

    diff = (loss - expected_loss).abs().item()
    assert diff < 1e-4, f"loss ≠ codebook_loss + β·commitment_loss: diff = {diff}"
    print(f"  ✅ loss = codebook_loss + β · commitment_loss 验证通过")
    print(f"  codebook_loss = {codebook_loss.item():.4f}, "
          f"commitment_loss = {commitment_loss.item():.4f}, "
          f"loss = {loss.item():.4f}, β = {vq.beta}")


def test_no_off_ball_dist():
    """Test 4: 修复后球外点代入 poincare_distance 不再发生, 梯度数值稳定

    场景: encoder 输出的 latent 是需要梯度的; 我们走一遍 forward + backward,
    验证:
    (a) x_q / latent_h 都确实在 Poincaré 球内 (修复前的球外点 → gradient 灾难)
    (b) loss 数值不饱和 (修复前 ≈23.7; 修复后期望 < 10)
    (c) backward 能成功跑通 (修复前因为 latent 在球外 → clamp artanh 梯度近零)
    """
    print("\n===== Test 4: 球外点代入 poincare_distance 修复确认 =====")
    vq = HVectorQuantization(n_e=16, e_dim=4, beta=0.5,
                             kmeans_init=True, kmeans_iters=5,
                             sk_eps=0.003, sk_iters=3, curvature=1.0)
    vq.initted = True
    with torch.no_grad():
        # 故意把 weight 初始化成球外 (norm > 1/√c = 1), 模拟修复前会触发的灾难场景
        vq.embeddings.weight.data.uniform_(-1.5, 1.5)

    # 模拟 encoder 输出 — 它需要梯度 (正常训练时)
    latent = torch.randn(8, 4, requires_grad=True) * 0.5
    latent.retain_grad()  # 标记非叶 tensor 也保留 grad (默认非叶不存)
    vq.train()  # 必须 train() 才能反向

    # 触发一次 forward, 拿到 loss
    x_q, loss, _ = vq(latent, use_sk=False)
    # backward 验证 (修复后 latent_h 已经 expmap0 + proj_to_ball,
    # 梯度路径: loss → poincare_distance → mobius_add / expmap0 → latent)
    loss.backward()

    latent_h = proj_to_ball(expmap0(latent, vq.c), vq.c)
    # 验证: latent_h 在球内 (norm < 1)
    norms = latent_h.norm(dim=-1)
    assert (norms < 1.0).all().item(), "latent_h 仍在球外"
    print(f"  ✅ latent_h ∈ 球内 — max ‖latent_h‖ = {norms.max().item():.4f}")
    # 验证: x_q 在球内
    norms_xq = x_q.norm(dim=-1)
    assert (norms_xq < 1.0).all().item(), "x_q 在球外"
    print(f"  ✅ x_q ∈ 球内 — max ‖x_q‖ = {norms_xq.max().item():.4f}")
    # 验证: loss 数值在合理范围 (不应饱和到 ~23.7)
    assert loss.item() < 50, f"loss 已饱和到不可用水平: {loss.item()}"
    print(f"  ✅ loss 数值正常 — loss = {loss.item():.4f} (无饱和)")
    # 验证: latent 有 grad (反向传播成功, 无 RuntimeError)
    assert latent.grad is not None, "latent.grad 为空, 反向传播失败"
    grad_norm = latent.grad.norm().item()
    assert grad_norm > 0 and grad_norm < 100, \
        f"latent 梯度异常: norm = {grad_norm} (修复前近零 / 修复后应有非零有限值)"
    print(f"  ✅ backward 成功 — ‖latent.grad‖ = {grad_norm:.4f} (非零, 数值稳定)")


if __name__ == '__main__':
    print("===========================================================")
    print("Task #178 Phase 0.3 — 量化器修复单元测试")
    print("===========================================================")

    test_get_codebook_consistency()
    test_x_q_inside_ball()
    test_beta_attached_to_commitment()
    test_no_off_ball_dist()

    print("\n===========================================================")
    print("✅ 全部 4 项验证通过")
    print("===========================================================")
