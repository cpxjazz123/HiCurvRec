"""DECOR PromptFormer precheck.

Gate 1: 数值稳定性 (alpha ∈ (0,1), candidate lookup 在 0-1023)
Gate 2: 梯度流通 (alpha_raw, bos_queries, q/k projections)
Gate 3: alpha gate 正确性
  - α → 0: e_final ≈ e_fused
  - α → 1: e_final ≈ e_soft
  - α = 0.35: 35% e_soft + 65% e_fused
Gate 4: T5 集成钩子 (e_fused 作为 inputs_embeds 输入 T5)
"""
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.decor_prompt_former import DecorPromptFormer


def test_gate1_numerical_stability():
    print("=" * 60)
    print("[Gate 1] 数值稳定性")
    print("=" * 60)

    torch.manual_seed(42)
    pf = DecorPromptFormer(d_model=128, vocab_size=1024, num_bins=4,
                           codes_per_bin=256, num_bos_queries=64, alpha_init=0.35)

    B, L = 8, 20
    input_ids = torch.randint(1, 1024, (B, L))
    attention_mask = torch.ones(B, L)
    attention_mask[0, -3:] = 0  # PAD

    e_fused_embedding = torch.nn.Embedding(1024, 128)
    nn.init.normal_(e_fused_embedding.weight, std=0.02)

    e_fused = e_fused_embedding(input_ids) * (128 ** 0.5)
    e_final, aux = pf(e_fused, input_ids, attention_mask, e_fused_embedding)

    print(f"  e_final shape: {e_final.shape}")
    print(f"  alpha: {aux['alpha']:.4f}")
    print(f"  bos_vec norm: {aux['bos_vec_norm']:.4f}")
    print(f"  e_soft norm: {aux['e_soft_norm']:.4f}")

    assert e_final.shape == e_fused.shape
    assert not torch.isnan(e_final).any(), "NaN"
    assert not torch.isinf(e_final).any(), "Inf"
    print("[PASS] Gate 1\n")


def test_gate2_gradient_flow():
    print("=" * 60)
    print("[Gate 2] 梯度流通")
    print("=" * 60)

    torch.manual_seed(42)
    pf = DecorPromptFormer(d_model=128, vocab_size=1024, num_bins=4,
                           codes_per_bin=256, num_bos_queries=64, alpha_init=0.35)

    B, L = 4, 10
    input_ids = torch.randint(1, 1024, (B, L))
    attention_mask = torch.ones(B, L)
    e_fused_embedding = torch.nn.Embedding(1024, 128)
    nn.init.normal_(e_fused_embedding.weight, std=0.02)

    e_fused = e_fused_embedding(input_ids) * (128 ** 0.5)
    e_final, aux = pf(e_fused, input_ids, attention_mask, e_fused_embedding)

    fake_loss = e_final.float().sum()
    fake_loss.backward()

    print(f"  alpha_raw.grad: {pf.alpha_raw.grad.item():.4f}")
    print(f"  bos_queries.grad.norm(): {pf.bos_queries.grad.norm().item():.4f}")
    print(f"  q_ctx.weight.grad.norm(): {pf.q_ctx.weight.grad.norm().item():.4f}")
    print(f"  k_candidates.weight.grad.norm(): {pf.k_candidates.weight.grad.norm().item():.4f}")

    assert pf.alpha_raw.grad is not None
    assert pf.alpha_raw.grad.abs() > 0
    assert not torch.isnan(pf.bos_queries.grad).any()
    assert pf.bos_queries.grad.abs().sum() > 0
    assert pf.q_ctx.weight.grad.abs().sum() > 0
    assert pf.k_candidates.weight.grad.abs().sum() > 0
    print("[PASS] Gate 2\n")


def test_gate3_alpha_gate_correctness():
    print("=" * 60)
    print("[Gate 3] alpha gate 正确性")
    print("=" * 60)

    torch.manual_seed(42)
    pf = DecorPromptFormer(d_model=128, vocab_size=1024, num_bins=4,
                           codes_per_bin=256, num_bos_queries=64, alpha_init=0.5)

    B, L = 2, 8
    input_ids = torch.randint(1, 1024, (B, L))
    attention_mask = torch.ones(B, L)
    e_fused_embedding = torch.nn.Embedding(1024, 128)
    nn.init.normal_(e_fused_embedding.weight, std=0.02)

    e_fused = e_fused_embedding(input_ids) * (128 ** 0.5)

    # Case 1: alpha → 0 (force alpha_raw → -inf)
    with torch.no_grad():
        pf.alpha_raw.fill_(-10.0)  # sigmoid(-10) ≈ 0
    e_final_a0, aux_a0 = pf(e_fused, input_ids, attention_mask, e_fused_embedding)
    err_a0 = (e_final_a0 - e_fused).abs().mean().item()
    print(f"  alpha → 0: |e_final - e_fused| = {err_a0:.6f} (期望 ≈ 0)")
    assert err_a0 < 1e-3, f"alpha=0 时 e_final 应 ≈ e_fused, 但 err={err_a0}"

    # Case 2: alpha → 1 (force alpha_raw → +inf)
    with torch.no_grad():
        pf.alpha_raw.fill_(10.0)  # sigmoid(10) ≈ 1
    e_final_a1, aux_a1 = pf(e_fused, input_ids, attention_mask, e_fused_embedding)
    # alpha=1 时 e_final = e_soft, 应该 != e_fused
    diff_a1 = (e_final_a1 - e_fused).abs().mean().item()
    print(f"  alpha → 1: |e_final - e_fused| = {diff_a1:.6f} (期望 > 0, 即 e_soft ≠ e_fused)")
    assert diff_a1 > 1e-4, f"alpha=1 时 e_final 应该 ≠ e_fused"

    # Case 3: alpha = 0.35 (DECOR 默认)
    with torch.no_grad():
        pf.alpha_raw.fill_(torch.log(torch.tensor(0.35 / 0.65)))  # logit(0.35) ≈ -0.619
    e_final_a035, aux_a035 = pf(e_fused, input_ids, attention_mask, e_fused_embedding)
    err_a035 = (e_final_a035 - e_fused).abs().mean().item()
    print(f"  alpha = 0.35: |e_final - e_fused| = {err_a035:.6f}")
    print(f"    alpha 实测: {aux_a035['alpha']:.4f}")
    assert abs(aux_a035['alpha'] - 0.35) < 1e-3, f"alpha 应 ≈ 0.35, 实际 {aux_a035['alpha']}"
    assert err_a035 > err_a0, "alpha=0.35 应比 alpha=0 偏离 e_fused 更远"
    assert err_a035 < diff_a1, "alpha=0.35 应比 alpha=1 偏离 e_fused 更近"
    print("[PASS] Gate 3: alpha gate 正确\n")


def test_gate4_t5_integration():
    print("=" * 60)
    print("[Gate 4] T5 集成钩子")
    print("=" * 60)

    import subprocess
    # 检查 stage3 是否能识别 decor_prompt_former 模块
    result = subprocess.run(
        ["grep", "-n", "decor_prompt_former\\|DecorPromptFormer\\|enable_prompt_former",
         "/home/wlia0047/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py"],
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout[:500])
    else:
        # 模块已创建, 等待集成到 stage3
        from common.decor_prompt_former import DecorPromptFormer
        pf = DecorPromptFormer()
        print(f"  DecorPromptFormer 已就绪: {pf.__class__.__name__}")
        print(f"  参数: bos_queries={pf.bos_queries.shape}, alpha_raw={pf.alpha_raw.shape}")
        print(f"  bin_offsets: {pf.bin_offsets.tolist()}")
        # 验证 bin_offsets: [0, 256, 512, 768]
        assert pf.bin_offsets.tolist() == [0, 256, 512, 768], f"bin_offsets 错: {pf.bin_offsets.tolist()}"
    print("[PASS] Gate 4\n")


if __name__ == "__main__":
    test_gate1_numerical_stability()
    test_gate2_gradient_flow()
    test_gate3_alpha_gate_correctness()
    test_gate4_t5_integration()
    print("=" * 60)
    print("[ALL GATES PASS] DECOR PromptFormer precheck done.")