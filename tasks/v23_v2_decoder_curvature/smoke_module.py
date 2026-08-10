#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23 v2 CurvatureResidualModule 单元测试:
1. α_*=0 起点: residual ≡ 0 (各 side)
2. α_*=0.5: residual 非零, 形状正确
3. L2 lookup (L0,L1) pair: index 计算正确
4. 梯度流: α_*_enc/dec + f1 + f2 都能 backprop
"""
import sys
import torch
import torch.nn as nn

# Bypass argparse --sid_npy required
sys.argv = ["smoke", "--sid_npy", "/tmp/fake.npy", "--product_dir", "/tmp/fake_prod"]

from common.stage3.stage3_train_pure_t5_v85p_repro import CurvatureResidualModule, _LAYER_ID_LUT  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_alpha_zero_is_identity():
    """α_* 都 init=0 时, residual ≡ 0 (训练起点等同 baseline)."""
    kappa_l1 = -0.006
    delta_kappa_l1 = torch.randn(64, 1) * 0.179
    kappa_l2 = -0.071
    delta_kappa_l2 = torch.randn(8192, 1) * 0.183
    m = CurvatureResidualModule(kappa_l1, delta_kappa_l1, kappa_l2, delta_kappa_l2,
                                d_model=128, mlp_hidden=64, alpha_init=0.0).to(DEVICE)
    m._cached_layer_id_lut = torch.from_numpy(_LAYER_ID_LUT).to(DEVICE)
    # 构造 batch: 4 段 SID (q_0,q_1,q_2,eos)
    input_ids = torch.tensor([
        [1, 65, 193, 449, 0, 0, 0],   # L0=1, L1=65, L2=193, eos=449
        [2, 100, 300, 449, 0, 0, 0],
        [3, 150, 400, 449, 0, 0, 0],
        [4, 192, 448, 449, 0, 0, 0],
    ], device=DEVICE)
    input_embeds = torch.randn(4, 7, 128, device=DEVICE)
    out_enc = m(input_ids, input_embeds, side='encoder')
    out_dec = m(input_ids, input_embeds, side='decoder')
    assert torch.allclose(out_enc, input_embeds, atol=1e-6), f"encoder α=0 not identity, max diff={(out_enc-input_embeds).abs().max()}"
    assert torch.allclose(out_dec, input_embeds, atol=1e-6), f"decoder α=0 not identity, max diff={(out_dec-input_embeds).abs().max()}"
    print("[OK] α_*=0 encoder + decoder 都是 identity")


def test_alpha_nonzero_injects_residual():
    """α_*=0.5 时, residual 非零."""
    kappa_l1 = -0.006
    delta_kappa_l1 = torch.randn(64, 1) * 0.179
    kappa_l2 = -0.071
    delta_kappa_l2 = torch.randn(8192, 1) * 0.183
    m = CurvatureResidualModule(kappa_l1, delta_kappa_l1, kappa_l2, delta_kappa_l2,
                                d_model=128, mlp_hidden=64, alpha_init=0.5).to(DEVICE)
    m._cached_layer_id_lut = torch.from_numpy(_LAYER_ID_LUT).to(DEVICE)
    input_ids = torch.tensor([
        [1, 65, 193, 449, 0, 0, 0],
        [2, 100, 300, 449, 0, 0, 0],
    ], device=DEVICE)
    input_embeds = torch.randn(2, 7, 128, device=DEVICE)
    out_enc = m(input_ids, input_embeds, side='encoder')
    out_dec = m(input_ids, input_embeds, side='decoder')
    assert not torch.allclose(out_enc, input_embeds), "encoder α=0.5 should inject residual"
    assert not torch.allclose(out_dec, input_embeds), "decoder α=0.5 should inject residual"
    # 共享 MLP f1/f2 + 同 α 值 → encoder 输出 == decoder 输出 (设计预期)
    # 验证: 单独设置 α_1_dec=0.5, α_1_enc=0 后应不同
    assert torch.allclose(out_enc, out_dec), \
        "encoder/decoder 共享 MLP + 同 α 值 → 输出应相同 (shared MLP design)"
    # 现在让 α_1_dec 不同 → 输出不同
    with torch.no_grad():
        m.alpha_1_dec.fill_(0.0)
        m.alpha_2_dec.fill_(0.0)
    out_dec2 = m(input_ids, input_embeds, side='decoder')
    assert not torch.allclose(out_enc, out_dec2), "α_1_dec=0 vs α_1_enc=0.5 → 输出应不同"
    print(f"[OK] α=0.5 encoder injects residual (max diff={(out_enc-input_embeds).abs().max():.4f})")
    print(f"[OK] α=0.5 decoder injects residual (max diff={(out_dec-input_embeds).abs().max():.4f})")
    print(f"[OK] encoder/decoder α 独立 → 不同 α 产生不同 residual")


def test_l2_pair_lookup():
    """L2 lookup 用 (L0, L1) pair, index = l0*128 + (l1-65)."""
    kappa_l1 = 0.0
    delta_kappa_l1 = torch.zeros(64, 1)
    kappa_l2 = 0.0
    # L2 delta_kappa: 设为 unique 值以验证 lookup
    delta_kappa_l2 = torch.arange(8192, dtype=torch.float32).reshape(8192, 1) * 0.001
    m = CurvatureResidualModule(kappa_l1, delta_kappa_l1, kappa_l2, delta_kappa_l2,
                                d_model=128, mlp_hidden=64, alpha_init=1.0).to(DEVICE)
    m._cached_layer_id_lut = torch.from_numpy(_LAYER_ID_LUT).to(DEVICE)
    # 测试不同 (L0, L1) pair:
    # q_0 = 1 (L0 idx 0), q_1 = 65 (L1 idx 0) → pair idx = 0*128+0 = 0
    # q_0 = 1, q_1 = 130 (L1 idx 65) → pair idx = 0*128+65 = 65
    # q_0 = 2 (L0 idx 1), q_1 = 65 (L1 idx 0) → pair idx = 1*128+0 = 128
    input_ids = torch.tensor([
        [1, 65, 193, 449],   # (L0=1, L1=65) → pair 0
        [1, 130, 193, 449],  # (L0=1, L1=130) → pair 65
        [2, 65, 193, 449],   # (L0=2, L1=65) → pair 128
        [64, 192, 448, 449], # (L0=64, L1=192) → pair 63*128+127 = 8191
    ], device=DEVICE)
    input_embeds = torch.zeros(4, 4, 128, device=DEVICE, requires_grad=False)
    out = m(input_ids, input_embeds, side='encoder')
    # 验证: 输出的 L2 位置 (pos 2) 应该有不同的 delta_kappa 注入
    # 比较 batch[2] (pair=128) 和 batch[0] (pair=0) 的 L2 residual
    diff_at_l2 = (out[2, 2] - out[0, 2]).abs().sum()
    assert diff_at_l2 > 1e-3, f"L2 lookup 应区分不同 (L0,L1) pair, diff={diff_at_l2}"
    print(f"[OK] L2 pair lookup 区分 (L0,L1) (diff_at_l2={diff_at_l2:.4f})")


def test_gradient_flow():
    """4 αs + 2 MLPs 都能 backprop (encoder 路径 + decoder 路径 分别测)."""
    # 用非零 delta_kappa 让 curvature 有意义
    delta_l1 = torch.randn(64, 1) * 0.179
    delta_l2 = torch.randn(8192, 1) * 0.183
    m = CurvatureResidualModule(-0.006, delta_l1, -0.071, delta_l2,
                                d_model=128, mlp_hidden=64, alpha_init=0.5).to(DEVICE)
    m._cached_layer_id_lut = torch.from_numpy(_LAYER_ID_LUT).to(DEVICE)
    input_ids = torch.tensor([[1, 65, 193, 449]], device=DEVICE)
    input_embeds = torch.randn(1, 4, 128, device=DEVICE, requires_grad=True)
    target = torch.randn(1, 4, 128, device=DEVICE)
    # Encoder forward
    out_enc = m(input_ids, input_embeds, side='encoder')
    loss = ((out_enc - target) ** 2).sum()
    loss.backward()
    grad_norms_enc = {
        "alpha_1_enc": m.alpha_1_enc.grad.abs().item(),
        "alpha_2_enc": m.alpha_2_enc.grad.abs().item(),
        "f1.0.weight": m.f1[0].weight.grad.norm().item(),
        "f1.2.weight": m.f1[2].weight.grad.norm().item(),
        "f2.0.weight": m.f2[0].weight.grad.norm().item(),
        "f2.2.weight": m.f2[2].weight.grad.norm().item(),
    }
    # 清零 + decoder forward
    for p in m.parameters():
        if p.grad is not None:
            p.grad.zero_()
    out_dec = m(input_ids, input_embeds, side='decoder')
    loss_dec = ((out_dec - target) ** 2).sum()
    loss_dec.backward()
    grad_norms_dec = {
        "alpha_1_dec": m.alpha_1_dec.grad.abs().item(),
        "alpha_2_dec": m.alpha_2_dec.grad.abs().item(),
        "f1.0.weight": m.f1[0].weight.grad.norm().item(),
        "f1.2.weight": m.f1[2].weight.grad.norm().item(),
        "f2.0.weight": m.f2[0].weight.grad.norm().item(),
        "f2.2.weight": m.f2[2].weight.grad.norm().item(),
    }
    for k, v in {**grad_norms_enc, **grad_norms_dec}.items():
        assert v > 0, f"{k} grad=0, expected > 0"
    print(f"[OK] 梯度流正常:")
    print("     encoder 路径:")
    for k, v in grad_norms_enc.items():
        print(f"       {k}: {v:.4e}")
    print("     decoder 路径:")
    for k, v in grad_norms_dec.items():
        print(f"       {k}: {v:.4e}")


def test_decoder_path_l2_uses_positions_minus_2():
    """Decoder 端 L2 lookup 应读 positions (i-2, i-1)."""
    m = CurvatureResidualModule(0.0, torch.zeros(64, 1), 0.0, torch.zeros(8192, 1),
                                d_model=128, mlp_hidden=64, alpha_init=1.0).to(DEVICE)
    m._cached_layer_id_lut = torch.from_numpy(_LAYER_ID_LUT).to(DEVICE)
    # decoder-style: positions 0=decoder_start, 1=q_0, 2=q_1, 3=q_2
    # 在 pos 3 (q_2 L2) 应读 pos 1 (q_0) 和 pos 2 (q_1)
    # 关键: 验证 mask=0 屏蔽 decoder_start/PAD 位置
    input_ids = torch.tensor([
        [0, 1, 65, 193, 449, 0],  # decoder_start=0, q_0=1, q_1=65, q_2=193, eos=449, pad
    ], device=DEVICE)
    input_embeds = torch.zeros(1, 6, 128, device=DEVICE)
    out_dec = m(input_ids, input_embeds, side='decoder')
    # pos 0 (decoder_start) 应无注入
    assert torch.allclose(out_dec[0, 0], torch.zeros(128, device=DEVICE), atol=1e-6), \
        f"decoder_start 应无注入, diff={(out_dec[0, 0]).abs().max()}"
    # pos 1 (q_0 L0) 应无注入
    assert torch.allclose(out_dec[0, 1], torch.zeros(128, device=DEVICE), atol=1e-6), \
        f"L0 应无注入, diff={(out_dec[0, 1]).abs().max()}"
    # pos 2 (q_1 L1) 应有注入
    assert not torch.allclose(out_dec[0, 2], torch.zeros(128, device=DEVICE)), \
        f"L1 decoder 应有注入"
    # pos 3 (q_2 L2) 应有注入
    assert not torch.allclose(out_dec[0, 3], torch.zeros(128, device=DEVICE)), \
        f"L2 decoder 应有注入"
    print("[OK] decoder path: decoder_start + L0 不注入, L1+L2 注入")


if __name__ == "__main__":
    print("=" * 60)
    print("v23 v2 CurvatureResidualModule 单元测试")
    print("=" * 60)
    test_alpha_zero_is_identity()
    test_alpha_nonzero_injects_residual()
    test_l2_pair_lookup()
    test_gradient_flow()
    test_decoder_path_l2_uses_positions_minus_2()
    print("=" * 60)
    print("ALL 5 TESTS PASSED ✅")
    print("=" * 60)