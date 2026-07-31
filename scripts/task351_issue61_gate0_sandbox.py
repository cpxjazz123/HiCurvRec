"""Task #351 / Issue #61 Gate 0 sandbox — hyp_c=-1.0 + Riemannian retraction sanity 5/5

本 sandbox 验证 Issue #61 Gate 0 实施基础 (zero-GPU, 不修改 upstream):
1. T1: expmap0(c=0.74) 走 Euclidean sphere 分支 (跟 #164 audit 一致)
2. T2: expmap0(c=-1.0) 走 tanh 分支 (真 hyperbolic)
3. T3: random vs hyp_c=-1.0 init 显著不同 (跟 #57 Gate 0 baseline 对比)
4. T4: Riemannian retraction 把 embedding 投影回 Poincaré ball (‖x‖ < 1/√|c|)
5. T5: wrapper 不改 upstream HG_Rec.py (HG_Rec_Issue57 继承 HG_Rec, 单点 patch)

所有测试 zero-GPU (CPU only), 跑完直接 commit.
"""
import math
import sys
import os
import numpy as np
import torch

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')
sys.path.insert(0, f'{REPO}/scripts')

from model.HG_Rec_issue57 import expmap0, build_codebook_from_sid, HG_Rec_Issue57


def riemannian_retract(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Post-step Riemannian retraction: 把 embedding 投影回 Poincaré ball.

    Poincaré ball 半径 = 1/√|c|. norm > 1/√|c| - eps 触发 projection.
    norm_new = min(norm, (1-eps)/√|c|)
    """
    if c >= 0:
        return x  # Euclidean branch: 无 retraction 需要
    ball_radius = 1.0 / math.sqrt(abs(c))
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target_norm = torch.clamp(norm, max=(1.0 - eps) * ball_radius)
    return x * (target_norm / norm)


def test_1_sphere_branch_hyp_c_074():
    """T1: expmap0(c=0.74) 走 Euclidean sphere 分支 (跟 #164 audit 一致)."""
    print("\n=== T1: expmap0(c=0.74) sphere branch ===")
    x = torch.randn(64, 32) * 0.5
    out = expmap0(x, c=0.74)
    print(f"  Input norm: mean={x.norm(dim=-1).mean():.4f}, max={x.norm(dim=-1).max():.4f}")
    print(f"  Output norm: mean={out.norm(dim=-1).mean():.4f}, max={out.norm(dim=-1).max():.4f}")
    # sphere branch: output norm < input norm (Euclidean contraction x / (1+sqrt(1+c·‖x‖²)))
    assert out.norm(dim=-1).mean() < x.norm(dim=-1).mean(), \
        "sphere branch 应该收缩 norm"
    # 1/√|0.74| ≈ 1.16, 但 sphere branch 不限制, 输出可以是任何 norm < 1
    print("  ✅ T1 PASS: expmap0(c=0.74) sphere contraction confirmed")
    return True


def test_2_hyperbolic_branch_hyp_c_neg1():
    """T2: expmap0(c=-1.0) 走 tanh 分支 (真 hyperbolic)."""
    print("\n=== T2: expmap0(c=-1.0) hyperbolic branch ===")
    x = torch.randn(64, 32) * 0.5
    out = expmap0(x, c=-1.0)
    out_norm = out.norm(dim=-1)
    print(f"  Input norm: mean={x.norm(dim=-1).mean():.4f}, max={x.norm(dim=-1).max():.4f}")
    print(f"  Output norm: mean={out_norm.mean():.4f}, max={out_norm.max():.4f}")
    # hyperbolic branch: output norm < 1/√|c| = 1.0
    ball_radius = 1.0 / math.sqrt(1.0)
    assert out_norm.max() < ball_radius, \
        f"hyperbolic branch output norm 必须 < {ball_radius}, got {out_norm.max():.4f}"
    # tanh 分支保证 norm < 1
    print(f"  ✅ T2 PASS: expmap0(c=-1.0) hyperbolic ball constraint (norm < {ball_radius})")
    return True


def test_3_random_vs_hyp_init_differs():
    """T3: random vs hyp_c=-1.0 init 显著不同 (跟 #57 Gate 0 baseline 对比)."""
    print("\n=== T3: random vs hyp_c=-1.0 init SID tokens ===")
    sid_arr = np.load(f'{REPO}/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_hyp_pre.npy')
    print(f"  SID arr shape: {sid_arr.shape}, total items: {len(sid_arr)}")

    config = {
        'num_layers': 2,
        'num_decoder_layers': 2,
        'd_model': 128,
        'd_ff': 256,
        'num_heads': 4,
        'd_kv': 32,
        'dropout_rate': 0.1,
        'vocab_size': 452,
        'pad_token_id': 0,
        'eos_token_id': 450,
        'decoder_start_token_id': 0,
        'feed_forward_proj': 'relu',
        'codebook_size': [64, 128, 256, 1],
        'sid_embedding_init': 'hyperbolic',
        'hyp_c': -1.0,  # Issue #61 fix: 真 hyperbolic branch
        'code_path': f'{REPO}/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_hyp_pre.npy',
    }

    # Random init
    config_rand = {**config, 'sid_embedding_init': 'random'}
    np.random.seed(42)
    model_rand = HG_Rec_Issue57(config_rand)
    sid_start = config['pad_token_id'] + 1
    sid_end = sid_start + sum(config['codebook_size'])
    rand_sid = model_rand.model.shared.weight[sid_start:sid_end].clone().detach()

    # Hyperbolic init (hyp_c=-1.0)
    config_hyp = {**config, 'sid_embedding_init': 'hyperbolic'}
    model_hyp = HG_Rec_Issue57(config_hyp)
    hyp_sid = model_hyp.model.shared.weight[sid_start:sid_end].clone().detach()

    diff = (rand_sid - hyp_sid).abs().mean()
    print(f"  Mean abs diff (SID tokens): {diff:.4f}")
    assert diff > 0, "hyperbolic init 应该跟 random init 不同"
    print(f"  Hyp init norm: mean={hyp_sid.norm(dim=-1).mean():.4f}, max={hyp_sid.norm(dim=-1).max():.4f}")
    # hyperbolic init norm 必须 < 1/√|c| = 1
    assert hyp_sid.norm(dim=-1).max() < 1.0, \
        f"hyperbolic init norm max 必须 < 1.0 (Poincaré ball), got {hyp_sid.norm(dim=-1).max():.4f}"
    print("  ✅ T3 PASS: random vs hyp_c=-1.0 init differs, hyp norm < 1.0")
    return True


def test_4_riemannian_retraction_projects_back():
    """T4: Riemannian retraction 把 embedding 投影回 Poincaré ball."""
    print("\n=== T4: Riemannian retraction (post-step) ===")
    # 模拟训练后 SID token embedding 飘出 ball (norm > 1/√|c| = 1)
    codebook = torch.randn(449, 32) * 2.0  # 大 norm 模拟飘出 ball
    print(f"  Pre-retract norm: mean={codebook.norm(dim=-1).mean():.4f}, max={codebook.norm(dim=-1).max():.4f}")

    retracted = riemannian_retract(codebook, c=-1.0)
    ret_norm = retracted.norm(dim=-1)
    print(f"  Post-retract norm: mean={ret_norm.mean():.4f}, max={ret_norm.max():.4f}")

    ball_radius = 1.0 - 1e-5  # eps=1e-5
    # FP tolerance: torch.clamp(..., max=0.99999) * (norm/norm) can round to 1.0 ± 1e-6
    assert ret_norm.max() <= ball_radius + 1e-5, \
        f"retract 后 norm max 必须 <= {ball_radius}, got {ret_norm.max():.6f}"
    # 跟 retracted 输入应该方向一致 (retraction 是径向 projection)
    cos_sim = torch.nn.functional.cosine_similarity(codebook, retracted, dim=-1)
    print(f"  Direction preserved: cos_sim min={cos_sim.min():.4f} (must ≈ 1.0)")
    assert cos_sim.min() > 0.999, "retraction 必须是径向 projection"
    print("  ✅ T4 PASS: retraction 投影回 Poincaré ball + 保持方向")
    return True


def test_5_wrapper_doesnt_modify_upstream():
    """T5: wrapper 不改 upstream HG_Rec.py (HG_Rec_Issue57 继承 HG_Rec, 单点 patch)."""
    print("\n=== T5: wrapper 不改 upstream HG_Rec.py ===")
    from model.HG_Rec import HG_Rec
    # HG_Rec_Issue57 继承 HG_Rec
    assert issubclass(HG_Rec_Issue57, HG_Rec), "HG_Rec_Issue57 必须继承 HG_Rec"
    print(f"  HG_Rec_Issue57.__bases__ = {HG_Rec_Issue57.__bases__}")

    # HG_Rec.__init__ source 不能改 (单点 patch 在子类的 _init_sid_embedding_hyperbolic)
    # 通过 hasattr + sig 检查 HG_Rec.__init__ 不包含 sid_embedding_init 参数
    import inspect
    hg_rec_sig = inspect.signature(HG_Rec.__init__)
    has_sid_init_param = 'sid_embedding_init' in hg_rec_sig.parameters
    print(f"  HG_Rec.__init__ has sid_embedding_init param: {has_sid_init_param}")
    assert not has_sid_init_param, "HG_Rec.__init__ 不应有 sid_embedding_init 参数 (single-point patch in subclass)"

    # 验证 wrapper 只是子类 patch, 没 monkey-patch upstream
    assert hasattr(HG_Rec_Issue57, '_init_sid_embedding_hyperbolic'), \
        "HG_Rec_Issue57 必须有 _init_sid_embedding_hyperbolic 方法"
    assert not hasattr(HG_Rec, '_init_sid_embedding_hyperbolic'), \
        "HG_Rec 不应有 _init_sid_embedding_hyperbolic (subclass-only)"
    print("  ✅ T5 PASS: wrapper 是 subclass single-point patch, 不改 upstream HG_Rec")
    return True


def main():
    print("=" * 70)
    print("Task #351 / Issue #61 Gate 0 sandbox — hyp_c=-1.0 + Riemannian retraction")
    print("=" * 70)

    results = []
    results.append(("T1 sphere branch c=0.74", test_1_sphere_branch_hyp_c_074()))
    results.append(("T2 hyperbolic branch c=-1.0", test_2_hyperbolic_branch_hyp_c_neg1()))
    results.append(("T3 random vs hyp_c=-1.0 init", test_3_random_vs_hyp_init_differs()))
    results.append(("T4 Riemannian retraction", test_4_riemannian_retraction_projects_back()))
    results.append(("T5 wrapper no upstream change", test_5_wrapper_doesnt_modify_upstream()))

    print("\n" + "=" * 70)
    print("Summary:")
    pass_count = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\nResult: {pass_count}/5 PASS")

    if pass_count == 5:
        print("\n✅ Issue #61 Gate 0 PASS — 实施基础就位 (zero-GPU sanity)")
        print("   下一步: Gate 1 Stage 3 训练 (~1.5h GPU) 等 R11.5 决策")
        return 0
    else:
        print(f"\n❌ Issue #61 Gate 0 FAIL — {5 - pass_count}/5 sanity FAIL")
        return 1


if __name__ == '__main__':
    sys.exit(main())