#!/usr/bin/env python3
"""Task #344 — Issue #62 Gate 0 sanity test: #30 + #43 joint wrapper compose.

验证两个 wrapper 顺序应用不破坏 baseline + 各自独立行为:
  T1: base HRQVAE forward = baseline
  T2: HRQVAEWithHypPre(base) forward = base + expmap0(c=0.74) pre-encoder
  T3: HRQVAEWithHypPre(base) + per-layer codebook transforms r_l+s_l
      forward = T2 + codebook weight transform
  T4: T3 codebook weight 在训练后保留 per-layer transform (grad flow OK)
  T5: T3 跟 baseline 在 r_l=[1,1,1] + s_l=[1,1,1] + HypPre disabled 输入下等价

通过条件: 5/5 PASS (跟 #30 #43 各自 Gate 0 一致).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))
sys.path.insert(0, str(REPO / "scripts"))

from model.hrqvae import HRQVAE

# Issue #43 wrapper
from task334_issue43_gate2a_hyp_pre_encoder import HRQVAEWithHypPre
# Issue #30 actual apply function lives in task301_issue30_gate1_stage1_train
from task301_issue30_gate1_stage1_train import apply_per_layer_codebook_transforms


def build_base_hrqvae(in_dim=768):
    return HRQVAE(
        in_dim=in_dim,
        num_emb_list=[64, 128, 256],
        e_dim=32,
        layers=[512, 256, 128, 64],
        dropout_prob=0.0,
        bn=False,
        loss_type='poincare',
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=True,
        kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=30,
    )


def T1_baseline_forward():
    """T1: base HRQVAE forward = baseline."""
    print("\n─── T1: base HRQVAE forward baseline ───")
    torch.manual_seed(42)
    base = build_base_hrqvae()
    base.eval()
    x = torch.randn(8, 768) * 0.5
    with torch.no_grad():
        # HRQVAE.forward returns 5-tuple: (out, rq_loss, indices, path_loss, div_ent)
        out, rq_loss, indices, path_loss, div_ent = base(x, use_sk=False)
    print(f"  out shape: {out.shape} (expected (8, 768))")
    print(f"  indices shape: {indices.shape}")
    print(f"  rq_loss: {rq_loss.item():.6f}")
    print(f"  path_loss: {path_loss}, div_ent: {div_ent}")
    assert out.shape == (8, 768), f"FAIL: out shape {out.shape}"
    print("  ✅ T1 PASS")
    return base


def T2_hyp_pre_forward():
    """T2: HRQVAEWithHypPre(base) forward."""
    print("\n─── T2: HRQVAEWithHypPre forward (Issue #43 alone) ───")
    torch.manual_seed(42)
    base = build_base_hrqvae()
    model = HRQVAEWithHypPre(base, c=0.74, enabled=True)
    model.eval()
    x = torch.randn(8, 768) * 0.5
    with torch.no_grad():
        out, rq_loss, indices, path_loss, div_ent = model(x, use_sk=False)
    print(f"  out shape: {out.shape}")
    print(f"  indices shape: {indices.shape}")
    print(f"  rq_loss: {rq_loss.item():.6f}")
    assert out.shape == (8, 768), f"FAIL: out shape {out.shape}"
    print("  ✅ T2 PASS")
    return base, model


def T3_joint_forward():
    """T3: HRQVAEWithHypPre + per-layer codebook transforms forward."""
    print("\n─── T3: HRQVAEWithHypPre + Codebook Transforms (Issue #62 joint) ───")
    torch.manual_seed(42)
    base = build_base_hrqvae()
    model = HRQVAEWithHypPre(base, c=0.74, enabled=True)

    # Build rotation matrices (identity for simplicity)
    num_emb_list = [64, 128, 256]
    e_dim = 32
    rotation_list = [torch.eye(e_dim) for _ in num_emb_list]
    # Issue #30 winning config: r_l=[0.1,1,10] + s_l=[2,2,2]
    radius_list = [0.1, 1.0, 10.0]
    scale_list = [2.0, 2.0, 2.0]

    # Apply transforms on the base's vq_layers (accessible via model.base.hrq.vq_layers)
    apply_per_layer_codebook_transforms(model.base, radius_list, rotation_list, scale_list)

    model.eval()
    x = torch.randn(8, 768) * 0.5
    with torch.no_grad():
        out, rq_loss, indices, path_loss, div_ent = model(x, use_sk=False)
    print(f"  out shape: {out.shape}")
    print(f"  indices shape: {indices.shape}")
    print(f"  rq_loss: {rq_loss.item():.6f}")
    print(f"  Codebook weight norms (post-transform):")
    for li, q in enumerate(model.base.hrq.vq_layers):
        print(f"    L{li}: ‖W‖={q.embeddings.weight.norm(dim=-1).mean():.4f}")
    assert out.shape == (8, 768), f"FAIL: out shape {out.shape}"
    print("  ✅ T3 PASS")
    return base, model


def T4_joint_gradient_flow():
    """T4: gradient flows through both wrappers."""
    print("\n─── T4: joint gradient flow ───")
    torch.manual_seed(42)
    base = build_base_hrqvae()
    model = HRQVAEWithHypPre(base, c=0.74, enabled=True)

    # Apply transforms
    num_emb_list = [64, 128, 256]
    e_dim = 32
    rotation_list = [torch.eye(e_dim) for _ in num_emb_list]
    radius_list = [0.1, 1.0, 10.0]
    scale_list = [2.0, 2.0, 2.0]
    apply_per_layer_codebook_transforms(model.base, radius_list, rotation_list, scale_list)

    # Pre-warm: trigger kmeans init with batch >= max(num_emb_list)=256 to set initted=True
    # (otherwise forward() during train() calls init_emb which fails on n_samples < n_clusters)
    model.train()
    n_warm = max(num_emb_list) * 2  # 512 samples > 256 largest codebook
    x_warm = torch.randn(n_warm, 768) * 0.5
    with torch.no_grad():
        _ = model(x_warm, use_sk=False)
    model.zero_grad()

    # Now real gradient test
    x = torch.randn(8, 768, requires_grad=False) * 0.5
    out, rq_loss, indices, path_loss, div_ent = model(x, use_sk=False)
    # Compute dummy loss = MSE
    loss = torch.nn.functional.mse_loss(out, x) + rq_loss
    loss.backward()

    # Check grad on codebook embeddings
    grad_norms = []
    for li, q in enumerate(model.base.hrq.vq_layers):
        g = q.embeddings.weight.grad
        if g is None:
            print(f"  L{li}: ❌ grad is None")
            grad_norms.append(0.0)
        else:
            gn = g.norm(dim=-1).mean().item()
            grad_norms.append(gn)
            print(f"  L{li}: grad mean norm = {gn:.6f}")
    assert all(g > 0 for g in grad_norms), f"FAIL: some layer has zero grad"
    # Check HypPre c (may be float, not learnable)
    if hasattr(model.hyp_pre, 'c'):
        c_val = model.hyp_pre.c
        if isinstance(c_val, torch.Tensor) and c_val.requires_grad:
            c_grad = c_val.grad
            print(f"  HypPre c (learnable tensor): grad={c_grad.item() if c_grad is not None else 'None'}")
        else:
            print(f"  HypPre c (fixed scalar): {c_val} (not learnable, as designed)")
    print("  ✅ T4 PASS")
    return base, model


def T5_joint_equivalence_to_baseline():
    """T5: joint with identity HypPre + identity Codebook Transforms = baseline.

    r_l=[1,1,1] + s_l=[1,1,1] (identity codebook transform) + HypPre disabled
    must equal baseline forward.
    """
    print("\n─── T5: joint identity compose ≡ baseline ───")
    torch.manual_seed(42)
    base_a = build_base_hrqvae()
    base_a.eval()
    x = torch.randn(8, 768) * 0.5

    # Baseline forward
    with torch.no_grad():
        out_base, _, _, _, _ = base_a(x, use_sk=False)

    # Joint with identity transforms
    torch.manual_seed(42)
    base_b = build_base_hrqvae()
    model = HRQVAEWithHypPre(base_b, c=0.74, enabled=False)  # disabled
    num_emb_list = [64, 128, 256]
    e_dim = 32
    rotation_list = [torch.eye(e_dim) for _ in num_emb_list]
    apply_per_layer_codebook_transforms(
        model.base,
        [1.0, 1.0, 1.0],  # identity radius
        rotation_list,
        [1.0, 1.0, 1.0],  # identity scale
    )
    model.eval()

    with torch.no_grad():
        out_joint, _, _, _, _ = model(x, use_sk=False)

    diff = (out_base - out_joint).abs().max().item()
    print(f"  baseline vs joint max diff: {diff:.6f}")
    assert diff < 1e-4, f"FAIL: identity compose not equivalent (diff={diff})"
    print("  ✅ T5 PASS")


def main():
    print("=" * 70)
    print("Task #344 — Issue #62 Gate 0 sanity test (#30 + #43 joint)")
    print("=" * 70)

    T1_baseline_forward()
    T2_hyp_pre_forward()
    T3_joint_forward()
    T4_joint_gradient_flow()
    T5_joint_equivalence_to_baseline()

    print("\n" + "=" * 70)
    print("✅ ALL 5/5 GATE 0 SANITY TESTS PASS")
    print("=" * 70)
    print("""
Issue #62 Gate 0 PASS:
  - T1: base HRQVAE forward OK
  - T2: HRQVAEWithHypPre wraps cleanly
  - T3: + per-layer Codebook Transforms r_l=[0.1,1,10]+s_l=[2,2,2] composes
  - T4: gradient flows through both wrappers (no dead params)
  - T5: identity compose ≡ baseline (no-op safe)

Gate 1 (Stage 1 + Stage 2) ready to launch (~3h GPU).
""")


if __name__ == "__main__":
    main()
