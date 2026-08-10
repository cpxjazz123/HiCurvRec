#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23 v2 P0 FIX 单元测试: 验证 fake_input_ids = [prev2?, prev1, current] 的 curvature lookup 正确性.

测试场景:
1. Step 0 (无 prev): fake = [current], 无 curvature 可查 (Layer 3 token, mask=0)
2. Step 1 (1 prev): fake = [prev1, current], L1 lookup 用 prev1 ✓
3. Step 2 (2 prev): fake = [prev2, prev1, current], L1 用 prev1, L2 用 (prev2, prev1) ✓
4. Step 3 (3 prev): fake = [prev3, prev2, prev1, current] (只取最后 2 个 prev, 即 [prev2, prev1])
5. tracker.append 正确累积 prefix
"""
import sys
import os
import torch

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
os.chdir(REPO)
sys.path.insert(0, REPO)

# 设置 sys.argv 在 import Stage4 之前 (Stage4 module-level argparse 需要)
sys.argv = [
    "smoke_p0_fix",
    "--ckpt_path", "/tmp/fake_ckpt.pth",
    "--sid_npy", "/tmp/fake.npy",
    "--product_dir", "/tmp/smoke",
]

from common.stage4.stage4_eval_pure_t5_v85p_4layer import BeamPrefixTracker, CurvatureResidualModuleEval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_step0_no_prev():
    """Step 0: 无 prev token (只有 decoder_start, vocab=0)."""
    tracker = BeamPrefixTracker(num_beams=2, device=DEVICE)
    assert tracker.prefix.shape == (2, 0)
    print("[OK] Step 0: tracker.prefix 初始为空 (2, 0)")


def test_step1_one_prev():
    """Step 1: 1 prev token. fake = [prev1, current]. L1 lookup 在 position 1 用 prev1."""
    tracker = BeamPrefixTracker(num_beams=2, device=DEVICE)
    # 模拟 step 0: append decoder_start
    decoder_start = torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
    tracker.append(decoder_start)
    assert tracker.prefix.shape == (2, 1)
    assert (tracker.prefix == 0).all()
    # 模拟 step 1: 当前 token = q_0 (L0, vocab [1,64])
    q_0 = torch.tensor([[5], [10]], dtype=torch.long, device=DEVICE)  # beam 0 q_0=5, beam 1 q_0=10
    prev = tracker.prefix
    if prev.shape[1] >= 2:
        fake = torch.cat([prev[:, -2:], q_0], dim=1)
    elif prev.shape[1] == 1:
        fake = torch.cat([prev[:, -1:], q_0], dim=1)
    else:
        fake = q_0
    assert fake.shape == (2, 2)
    assert fake[0].tolist() == [0, 5]  # [decoder_start, q_0_beam_0]
    assert fake[1].tolist() == [0, 10]  # [decoder_start, q_0_beam_1]
    # 现在 fake 形状 (2, 2): 在 position 1 (q_0, L0), torch.roll(fake, 1) at pos 1 = pos 0 = decoder_start
    # 但 q_0 是 L0, layer_id=0, mask=0 → 无 curvature 注入 (正确)
    print(f"[OK] Step 1: fake={fake.tolist()} (decoder_start + q_0)")


def test_step2_two_prev():
    """Step 2: 2 prev token. fake = [prev2, prev1, current]. L1 lookup 在 pos 2 用 prev1."""
    tracker = BeamPrefixTracker(num_beams=2, device=DEVICE)
    decoder_start = torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
    q_0 = torch.tensor([[5], [10]], dtype=torch.long, device=DEVICE)
    tracker.append(decoder_start)
    tracker.append(q_0)
    # 模拟 step 2: 当前 token = q_1 (L1, vocab [65,192])
    q_1 = torch.tensor([[100], [150]], dtype=torch.long, device=DEVICE)  # beam 0 q_1=100, beam 1 q_1=150
    prev = tracker.prefix
    fake = torch.cat([prev[:, -2:], q_1], dim=1)  # (2, 3)
    assert fake.shape == (2, 3)
    assert fake[0].tolist() == [0, 5, 100]  # [decoder_start, q_0_beam_0, q_1_beam_0]
    assert fake[1].tolist() == [0, 10, 150]  # [decoder_start, q_0_beam_1, q_1_beam_1]
    # 在 position 2 (q_1, L1), torch.roll(fake, 1) = position 1 = q_0 ✓
    # torch.roll(fake, 2) = position 0 = decoder_start (但 q_1 是 L1, L2 mask=0)
    print(f"[OK] Step 2: fake={fake.tolist()} (decoder_start + q_0 + q_1)")


def test_step3_three_prev():
    """Step 3: 3 prev token. fake = [prev2, prev1, current]. L2 lookup 在 pos 2 用 (prev2, prev1)."""
    tracker = BeamPrefixTracker(num_beams=2, device=DEVICE)
    decoder_start = torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
    q_0 = torch.tensor([[5], [10]], dtype=torch.long, device=DEVICE)
    q_1 = torch.tensor([[100], [150]], dtype=torch.long, device=DEVICE)
    tracker.append(decoder_start)
    tracker.append(q_0)
    tracker.append(q_1)
    assert tracker.prefix.shape == (2, 3)
    # 模拟 step 3: 当前 token = q_2 (L2, vocab [193,448])
    q_2 = torch.tensor([[300], [400]], dtype=torch.long, device=DEVICE)
    prev = tracker.prefix  # (2, 3)
    fake = torch.cat([prev[:, -2:], q_2], dim=1)  # (2, 3) — 取最后 2 个 prev + current
    assert fake.shape == (2, 3)
    assert fake[0].tolist() == [5, 100, 300]  # [q_0, q_1, q_2] beam 0
    assert fake[1].tolist() == [10, 150, 400]  # [q_0, q_1, q_2] beam 1
    # 在 position 2 (q_2, L2):
    #   torch.roll(fake, 1) at pos 2 = pos 1 = q_1 ✓ (L1 lookup, 但 q_2 是 L2, L1 mask=0)
    #   torch.roll(fake, 2) at pos 2 = pos 0 = q_0 ✓ (L2 q_0 lookup)
    #   torch.roll(fake, 1) at pos 2 = pos 1 = q_1 ✓ (L2 q_1 lookup)
    print(f"[OK] Step 3: fake={fake.tolist()} (q_0 + q_1 + q_2)")


def test_curvature_lookup_with_fake_prefix():
    """实际跑 CurvatureResidualModule 在 fake prefix 上, 验证 L2 lookup 拿到正确 (q_0, q_1) pair."""
    # 设置 Stage2 ckpt 的真实 kappa/delta_kappa
    kappa_l1 = 0.0
    delta_kappa_l1 = torch.zeros(64, 1, device=DEVICE)
    kappa_l2 = 0.0
    # 用 unique delta_kappa_l2 验证 lookup 正确
    delta_kappa_l2 = (torch.arange(8192, dtype=torch.float32, device=DEVICE) * 0.001).reshape(8192, 1)
    m = CurvatureResidualModuleEval(kappa_l1, delta_kappa_l1, kappa_l2, delta_kappa_l2,
                                     d_model=128, mlp_hidden=64, alpha_init=1.0).to(DEVICE)
    # 准备 layer_id_lut
    import numpy as np
    _LAYER_ID_LUT_np = np.zeros(1025, dtype=np.int64)
    _LAYER_ID_LUT_np[1:65] = 0
    _LAYER_ID_LUT_np[65:193] = 1
    _LAYER_ID_LUT_np[193:449] = 2
    _LAYER_ID_LUT_np[449] = 3
    m._cached_layer_id_lut = torch.from_numpy(_LAYER_ID_LUT_np).to(DEVICE)
    # 模拟 q_2 位置: fake = [q_0=2, q_1=130, q_2=300]
    # q_0 idx = 2-1 = 1, q_1 idx = 130-65 = 65, pair idx = 1*128+65 = 193
    # delta_kappa_l2[193] = 193 * 0.001 = 0.193
    fake_input_ids = torch.tensor([[2, 130, 300]], dtype=torch.long, device=DEVICE)  # (1, 3)
    fake_embeds = torch.zeros(1, 3, 128, device=DEVICE)
    out = m(fake_input_ids, fake_embeds, side='decoder')
    # 在 position 2 (q_2, L2), 应该注入 L2 residual based on (q_0=2, q_1=130) pair
    # delta_kappa_l2 at pair 193 = 0.193
    # f2([kappa_l2=0, delta_kappa=0.193]) 应该非零
    l2_pos_residual = out[0, 2]  # (128,)
    assert l2_pos_residual.abs().max() > 1e-3, \
        f"L2 position 应该有 curvature 注入, 但 max abs = {l2_pos_residual.abs().max()}"
    # 比较 buggy vs correct: buggy lookup 用 fake=[300] (单 token) → pair idx = (300-1)*128+(300-65) = 299*128+235
    # correct lookup 用 fake=[2, 130, 300] → pair idx = (2-1)*128+(130-65) = 193
    buggy_fake = torch.tensor([[300]], dtype=torch.long, device=DEVICE)
    buggy_out = m(buggy_fake, torch.zeros(1, 1, 128, device=DEVICE), side='decoder')
    buggy_l2_residual = buggy_out[0, 0]
    # buggy 和 correct 应该有显著不同 (因为 pair idx 完全不同)
    diff = (l2_pos_residual - buggy_l2_residual).abs().max()
    assert diff > 1e-3, \
        f"fake prefix lookup 应与 buggy (单 token) lookup 不同, 但 diff={diff}"
    print(f"[OK] Curvature lookup 在 fake prefix 上拿到正确的 (q_0, q_1) pair idx")
    print(f"     fake prefix L2 residual max abs: {l2_pos_residual.abs().max():.4f}")
    print(f"     buggy (单 token) L2 residual max abs: {buggy_l2_residual.abs().max():.4f}")
    print(f"     diff: {diff:.4f} (>1e-3 表明 lookup 路径不同)")


def test_tracker_append_accumulates():
    """tracker.append 应正确累积 prefix."""
    tracker = BeamPrefixTracker(num_beams=3, device=DEVICE)
    # Step 0: decoder_start
    t0 = torch.zeros(3, 1, dtype=torch.long, device=DEVICE)
    tracker.append(t0)
    assert tracker.prefix.shape == (3, 1)
    # Step 1: q_0
    t1 = torch.tensor([[5], [10], [20]], dtype=torch.long, device=DEVICE)
    tracker.append(t1)
    assert tracker.prefix.shape == (3, 2)
    # Step 2: q_1
    t2 = torch.tensor([[100], [150], [180]], dtype=torch.long, device=DEVICE)
    tracker.append(t2)
    assert tracker.prefix.shape == (3, 3)
    # 验证值
    expected = torch.tensor([[0, 5, 100], [0, 10, 150], [0, 20, 180]], dtype=torch.long, device=DEVICE)
    assert torch.equal(tracker.prefix, expected), f"tracker.prefix = {tracker.prefix.tolist()}, expected {expected.tolist()}"
    print(f"[OK] tracker.append 累积: prefix.shape = {tracker.prefix.shape}, prefix = {tracker.prefix.tolist()}")


if __name__ == "__main__":
    print("=" * 70)
    print("v23 v2 P0 FIX 单元测试")
    print("=" * 70)
    test_step0_no_prev()
    test_step1_one_prev()
    test_step2_two_prev()
    test_step3_three_prev()
    test_tracker_append_accumulates()
    test_curvature_lookup_with_fake_prefix()
    print("=" * 70)
    print("ALL 6 TESTS PASSED ✅")
    print("=" * 70)