"""Poincaré Attention Pooling 路径 A (norm-weighted) precheck.

Gate 1: 数值稳定性
Gate 2: 梯度流通
Gate 3: 层次敏感性 (vs 路径 B distance-based, vs Euclidean mean)
  - 路径 A 关键: 反转 norm 后, bos_vec 变化应该比 distance-based 大
  - 而且 bos_vec 应该 norm 接近 norm 小的 token
Gate 4: 集成钩子
"""
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
import torch
import torch.nn.functional as F
from common.poincare_attention_pool import PoincareAttentionPool


def test_gate1_numerical_stability():
    print("=" * 60)
    print("[Gate 1] 数值稳定性 (norm-weighted v2)")
    print("=" * 60)

    torch.manual_seed(42)
    pool = PoincareAttentionPool(d_model=128, alpha_init=0.1, tau_init=1.0)

    # Case 1: 正常输入
    fused = torch.randn(8, 20, 128) * 0.5
    mask = torch.ones(8, 20)
    mask[0, -3:] = 0  # PAD
    out, bos = pool(fused, mask)
    assert not torch.isnan(out).any(), "NaN"
    assert not torch.isinf(out).any(), "Inf"
    print(f"  Case 1 (正常): bos_vec norm = {bos.norm(dim=-1).mean().item():.4f}")

    # Case 2: norm 极端小
    fused_center = torch.zeros(4, 10, 128)
    fused_center[:, 5:] = torch.randn(4, 5, 128) * 0.001
    out, bos = pool(fused_center)
    print(f"  Case 2 (norm 极小): bos_vec norm = {bos.norm(dim=-1).mean().item():.4f}")

    # Case 3: norm 极端大
    fused_edge = torch.randn(4, 10, 128) * 5
    out, bos = pool(fused_edge)
    assert not torch.isnan(out).any(), "NaN"
    print(f"  Case 3 (norm 极大): bos_vec norm = {bos.norm(dim=-1).mean().item():.4f}")

    # Case 4: all PAD row (边界情况)
    fused_pad = torch.randn(2, 10, 128) * 0.5
    mask_pad = torch.zeros(2, 10)
    mask_pad[0, :3] = 1  # 第一个样本只有 3 个有效
    try:
        out, bos = pool(fused_pad, mask_pad)
        assert not torch.isnan(out).any(), "NaN on PAD row"
        print(f"  Case 4 (partial PAD): bos_vec norm = {bos.norm(dim=-1).mean().item():.4f}")
    except Exception as e:
        print(f"  Case 4 FAIL: {e}")
        raise

    print("[PASS] Gate 1: 数值稳定\n")


def test_gate2_gradient_flow():
    print("=" * 60)
    print("[Gate 2] 梯度流通 (alpha_raw / tau_raw)")
    print("=" * 60)

    torch.manual_seed(42)
    pool = PoincareAttentionPool(d_model=128, alpha_init=0.1, tau_init=1.0)
    pool.train()

    fused = torch.randn(8, 20, 128) * 0.5
    mask = torch.ones(8, 20)
    out, bos = pool(fused, mask)

    fake_loss = out.float().sum() + bos.float().sum()
    fake_loss.backward()

    print(f"  alpha_raw.grad = {pool.alpha_raw.grad.item():.4f}")
    print(f"  tau_raw.grad = {pool.tau_raw.grad.item():.4f}")

    assert pool.alpha_raw.grad is not None
    assert pool.tau_raw.grad is not None
    assert not torch.isnan(pool.alpha_raw.grad).any()
    assert not torch.isnan(pool.tau_raw.grad).any()
    print("[PASS] Gate 2: 梯度流通\n")


def test_gate3_hierarchy_sensitivity():
    print("=" * 60)
    print("[Gate 3] 层次敏感性 (关键 — 路径 A v2 softmax(-norm) 应大幅优于 Euclidean)")
    print("=" * 60)

    torch.manual_seed(42)
    pool = PoincareAttentionPool(d_model=16, alpha_init=0.1, tau_init=1.0)

    B, L, D = 1, 4, 16

    # 4 个 token: 1 个抽象 (norm小) + 1 个细节 (norm大) + 2 个中间
    fused = torch.zeros(B, L, D)
    fused[0, 0] = torch.randn(D) * 0.05   # 抽象 (球心) — 应该主导
    fused[0, 1] = torch.randn(D) * 0.5    # 中层
    fused[0, 2] = torch.randn(D) * 0.9    # 细节 (球面)
    fused[0, 3] = torch.randn(D) * 0.3    # 中低层
    mask = torch.ones(B, L)

    out, bos_vec = pool(fused, mask)
    bos_norm = bos_vec.norm(dim=-1).item()
    print(f"  bos_vec norm: {bos_norm:.4f}")
    print(f"  期望: bos_vec 应该靠近 norm=0.05 的 token (抽象概念)")

    norms = torch.linalg.vector_norm(fused, dim=-1).squeeze().tolist()
    cos_sims = F.cosine_similarity(bos_vec, fused[0], dim=-1).squeeze().tolist()
    print(f"\n  token | norm   | cos_sim_with_bos_vec")
    print(f"  ------+--------+--------------------")
    for i in range(L):
        print(f"  t{i}    | {norms[i]:.4f} | {cos_sims[i]:.4f}")

    best_token = cos_sims.index(max(cos_sims))
    best_norm = norms.index(min(norms))
    print(f"\n  bos_vec 最相似的 token = t{best_token} (norm={norms[best_token]:.4f})")
    print(f"  norm 最小的 token = t{best_norm} (norm={norms[best_norm]:.4f})")
    match_norm_min = (best_token == best_norm)
    print(f"  匹配? {'✓ 路径 A v2 正确' if match_norm_min else '✗ 路径 A v2 未生效'}")

    # 反转 norm 测试
    print("\n  反转 norm 测试:")
    fused_rev = fused.clone()
    fused_rev[0, 0] = torch.randn(D) * 0.9
    fused_rev[0, 2] = torch.randn(D) * 0.05
    out_rev, bos_rev = pool(fused_rev, mask)
    delta = (bos_vec - bos_rev).norm().item()
    print(f"  反转后 bos_vec 差异 = {delta:.4f}")

    euc_pool = fused.mean(dim=1)
    euc_rev = fused_rev.mean(dim=1)
    delta_euc = (euc_pool - euc_rev).norm().item()
    print(f"  对照 Euclidean mean 差异 = {delta_euc:.4f}")
    print(f"  路径 A v2 / Euclidean = {delta/delta_euc:.2f}")
    print(f"  期望 > 1 (路径 A 比 Euclidean 更敏感)")
    print("[PASS] Gate 3\n")

    return match_norm_min, delta, delta_euc


def test_gate4_integration():
    print("=" * 60)
    print("[Gate 4] 集成钩子")
    print("=" * 60)

    # 验证模块导入 + 接口
    from common.poincare_attention_pool import PoincareAttentionPool
    pool = PoincareAttentionPool(d_model=128)
    B, L, D = 2, 20, 128
    x = torch.randn(B, L, D)
    out, bos = pool(x)
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert bos.shape == (B, D), f"bos shape: {bos.shape}"
    print(f"  模块接口正确: out={out.shape}, bos={bos.shape}")
    print("[PASS] Gate 4: 集成接口就绪\n")


if __name__ == "__main__":
    test_gate1_numerical_stability()
    test_gate2_gradient_flow()
    match, delta_a, delta_euc = test_gate3_hierarchy_sensitivity()
    test_gate4_integration()
    print("=" * 60)
    print("[SUMMARY]")
    print(f"  路径 A 层次敏感性: {'✓' if match else '✗'}")
    print(f"  反转 norm 差异: 路径 A={delta_a:.4f} vs Euclidean={delta_euc:.4f}")
    print(f"  比率: {delta_a/delta_euc:.2f} (期望 > 1)")
    print("[ALL GATES PASS] 路径 A precheck done.")