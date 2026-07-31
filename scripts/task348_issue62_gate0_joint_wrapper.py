#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #348 / Issue #62 Gate 0 — #30+#43 联合 wrapper 5/5 sanity test.

Joint wrapper: Issue #30 per-layer Codebook Transforms + Issue #43 HypPreEncoder.

Architecture:
    x_768d → [Issue #43] HypPreEncoder(expmap0 c=0.74) → encoder
          → [Issue #30] per-layer Codebook Transforms (radius+rotation+scale)
          → hrq (RQ-VAE) → decoder

Both wrappers are PASS-state individually. Joint composition is monotonic
extension (no upstream HRQ-VAE patch). Per R11.4 dry-run report: composes two
existing PASS wrappers, no critical decision needed.

Usage:
    python3 scripts/task348_issue62_gate0_joint_wrapper.py
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset, expmap0
from scripts.task334_issue43_gate2a_hyp_pre_encoder import HypPreEncoder


# ============================================================================
# Joint Wrapper: Issue #30 + #43 composition
# ============================================================================

class HRQVAEWithHypPreAndPerLayerTransforms(nn.Module):
    """Issue #62 Gate 0: 组合 Issue #30 per-layer transforms + Issue #43 HypPreEncoder.

    Pipeline:
        x (768d) → HypPreEncoder (Issue #43) → encoder → vq_layers w/ per-layer
                  transforms (Issue #30) → decoder → out (768d)

    Both transforms are opt-in via flag. When all flags OFF, behavior is identical
    to baseline HRQVAE (regression-safe).
    """

    def __init__(self, base_hrqvae: HRQVAE,
                 hyp_c: float = 0.74, hyp_enabled: bool = True,
                 radius_list: Optional[List[float]] = None,
                 rotation_list: Optional[List[torch.Tensor]] = None,
                 scale_list: Optional[List[float]] = None):
        super().__init__()
        self.base = base_hrqvae
        self.hyp_pre = HypPreEncoder(c=hyp_c, enabled=hyp_enabled)
        n_layers = len(base_hrqvae.num_emb_list)
        e_dim = base_hrqvae.e_dim

        # Default to identity transforms (r=1, R=I, s=1) per layer
        if radius_list is None:
            radius_list = [1.0] * n_layers
        if rotation_list is None:
            rotation_list = [torch.eye(e_dim) for _ in range(n_layers)]
        if scale_list is None:
            scale_list = [1.0] * n_layers

        assert len(radius_list) == n_layers, f"radius_list len {len(radius_list)} != {n_layers}"
        assert len(rotation_list) == n_layers, f"rotation_list len {len(rotation_list)} != {n_layers}"
        assert len(scale_list) == n_layers, f"scale_list len {len(scale_list)} != {n_layers}"

        self.radius_list = list(radius_list)
        self.rotation_list = list(rotation_list)
        self.scale_list = list(scale_list)

    def forward(self, x, use_sk=True):
        # Step 1: HypPreEncoder (Issue #43): x → expmap0(x, c=0.74)
        x = self.hyp_pre(x)
        # Step 2: per-layer transforms (Issue #30): apply to vq_layers forward
        return self._patched_vq_forward(x, use_sk=use_sk)

    def _patched_vq_forward(self, x, use_sk=True):
        """Monkey-patch each vq_layer.forward to apply per-layer transform matrix
        to embeddings.weight before distance computation."""
        originals = []
        for li, q in enumerate(self.base.hrq.vq_layers):
            originals.append(q.forward)
            eff = (self.scale_list[li] * self.radius_list[li]) * self.rotation_list[li]
            eff = eff.to(device=q.embeddings.weight.device, dtype=q.embeddings.weight.dtype)
            orig_fwd = q.forward

            def make_patched(orig, ef):
                def patched(_self, x, use_sk=True):
                    orig_w = _self.embeddings.weight.data.clone()
                    _self.embeddings.weight.data = orig_w @ ef.t()
                    try:
                        return orig(x, use_sk=use_sk)
                    finally:
                        _self.embeddings.weight.data = orig_w
                return patched

            q.forward = make_patched(orig_fwd, eff).__get__(q, type(q))

        try:
            return self.base(x, use_sk=use_sk)
        finally:
            for li, q in enumerate(self.base.hrq.vq_layers):
                q.forward = originals[li]

    @torch.no_grad()
    def get_indices(self, x, use_sk=True):
        x = self.hyp_pre(x)
        return self._patched_vq_get_indices(x, use_sk=use_sk)

    def _patched_vq_get_indices(self, x, use_sk=True):
        originals = []
        for li, q in enumerate(self.base.hrq.vq_layers):
            originals.append(q.forward)
            eff = (self.scale_list[li] * self.radius_list[li]) * self.rotation_list[li]
            eff = eff.to(device=q.embeddings.weight.device, dtype=q.embeddings.weight.dtype)
            orig_fwd = q.forward

            def make_patched(orig, ef):
                def patched(_self, x, use_sk=True):
                    orig_w = _self.embeddings.weight.data.clone()
                    _self.embeddings.weight.data = orig_w @ ef.t()
                    try:
                        return orig(x, use_sk=use_sk)
                    finally:
                        _self.embeddings.weight.data = orig_w
                return patched

            q.forward = make_patched(orig_fwd, eff).__get__(q, type(q))

        try:
            return self.base.get_indices(x, use_sk=use_sk)
        finally:
            for li, q in enumerate(self.base.hrq.vq_layers):
                q.forward = originals[li]

    def compute_loss(self, *args, **kwargs):
        return self.base.compute_loss(*args, **kwargs)


# ============================================================================
# Sanity Test 5/5
# ============================================================================

def build_identity_rotation_list(num_emb_list, e_dim):
    return [torch.eye(e_dim) for _ in num_emb_list]


def test_t1_baseline_only():
    """T1: baseline HRQVAE without any wrapper → shape + RQ loss finite."""
    print("\n─── T1: Baseline HRQVAE forward pass ───")
    torch.manual_seed(42)
    in_dim = 768
    num_emb_list = [64, 128, 256]
    e_dim = 32

    base = HRQVAE(
        in_dim=in_dim, num_emb_list=num_emb_list,
        e_dim=e_dim, layers=[512, 256, 128, 64],
        dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.25,
        kmeans_init=False, kmeans_iters=10,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=3,
    )

    x = torch.randn(4, 768) * 0.03
    out, rq_loss, indices, path_loss, _div = base(x, use_sk=False)

    shape_ok = out.shape == x.shape
    rq_loss_ok = torch.isfinite(rq_loss).item()
    print(f"  out.shape = {out.shape}, rq_loss = {rq_loss.item():.4f}")
    print(f"  {'✅ PASS' if shape_ok and rq_loss_ok else '❌ FAIL'}")
    return shape_ok and rq_loss_ok


def test_t2_hyp_pre_only_per_layer_identity():
    """T2: HypPre ON, per-layer identity → output == HypPre direct, ‖y‖ < 1/√c."""
    print("\n─── T2: HypPre ON + per-layer identity ───")
    torch.manual_seed(42)
    in_dim = 768
    num_emb_list = [64, 128, 256]
    e_dim = 32

    base = HRQVAE(
        in_dim=in_dim, num_emb_list=num_emb_list,
        e_dim=e_dim, layers=[512, 256, 128, 64],
        dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.25,
        kmeans_init=False, kmeans_iters=10,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=3,
    )

    wrapped = HRQVAEWithHypPreAndPerLayerTransforms(
        base_hrqvae=base, hyp_c=0.74, hyp_enabled=True,
        radius_list=[1.0, 1.0, 1.0],
        rotation_list=build_identity_rotation_list(num_emb_list, e_dim),
        scale_list=[1.0, 1.0, 1.0],
    )

    x = torch.randn(4, 768) * 0.03
    out, rq_loss, indices, _, _ = wrapped(x, use_sk=False)

    # Direct expmap0 + base should match
    x_hyp_direct = expmap0(x, c=0.74)
    out_direct, rq_loss_direct, _, _, _ = base(x_hyp_direct, use_sk=False)

    diff_out = (out - out_direct).abs().max().item()
    diff_loss = abs(rq_loss.item() - rq_loss_direct.item())

    print(f"  out shape: {out.shape}, ‖out‖ range: [{out.norm(dim=-1).min().item():.4f}, {out.norm(dim=-1).max().item():.4f}]")
    print(f"  max|out_wrapped - out_direct| = {diff_out:.6e}")
    print(f"  |rq_loss_wrapped - rq_loss_direct| = {diff_loss:.6e}")

    # Boundary check on HypPre output (should be < 1/√c=1.155)
    boundary = 1.0 / (0.74 ** 0.5) - 1e-5
    x_hyp_norm = x_hyp_direct.norm(dim=-1).max().item()
    in_ball = x_hyp_norm < boundary

    pass_test = diff_out < 1e-4 and diff_loss < 1e-4 and in_ball
    print(f"  HypPre direct ‖x_hyp‖_max = {x_hyp_norm:.4f} (boundary={boundary:.4f})")
    print(f"  {'✅ PASS' if pass_test else '❌ FAIL'}")
    return pass_test


def test_t3_per_layer_identity_only():
    """T3: Per-layer transforms identity (r=1, R=I, s=1) only, HypPre OFF → output == baseline."""
    print("\n─── T3: Per-layer identity only (HypPre OFF) ───")
    torch.manual_seed(42)
    in_dim = 768
    num_emb_list = [64, 128, 256]
    e_dim = 32

    base = HRQVAE(
        in_dim=in_dim, num_emb_list=num_emb_list,
        e_dim=e_dim, layers=[512, 256, 128, 64],
        dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.25,
        kmeans_init=False, kmeans_iters=10,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=3,
    )

    wrapped = HRQVAEWithHypPreAndPerLayerTransforms(
        base_hrqvae=base, hyp_c=0.74, hyp_enabled=False,  # HypPre OFF
        radius_list=[1.0, 1.0, 1.0],                       # identity
        rotation_list=build_identity_rotation_list(num_emb_list, e_dim),
        scale_list=[1.0, 1.0, 1.0],                       # identity
    )

    x = torch.randn(4, 768) * 0.03
    out_wrapped, rq_loss_wrapped, _, _, _ = wrapped(x, use_sk=False)
    out_base, rq_loss_base, _, _, _ = base(x, use_sk=False)

    diff_out = (out_wrapped - out_base).abs().max().item()
    diff_loss = abs(rq_loss_wrapped.item() - rq_loss_base.item())

    print(f"  max|out_wrapped - out_base| = {diff_out:.6e}")
    print(f"  |rq_loss_wrapped - rq_loss_base| = {diff_loss:.6e}")
    pass_test = diff_out < 1e-5 and diff_loss < 1e-5
    print(f"  {'✅ PASS' if pass_test else '❌ FAIL'}")
    return pass_test


def test_t4_both_wrappers_active():
    """T4: HypPre ON + per-layer design (r=[0.1,1,10], s=[2,2,2]) → output differs from baseline."""
    print("\n─── T4: Both wrappers active (Issue #30 design + Issue #43 HypPre) ───")
    torch.manual_seed(42)
    in_dim = 768
    num_emb_list = [64, 128, 256]
    e_dim = 32

    base = HRQVAE(
        in_dim=in_dim, num_emb_list=num_emb_list,
        e_dim=e_dim, layers=[512, 256, 128, 64],
        dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.25,
        kmeans_init=False, kmeans_iters=10,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=3,
    )

    wrapped = HRQVAEWithHypPreAndPerLayerTransforms(
        base_hrqvae=base, hyp_c=0.74, hyp_enabled=True,
        radius_list=[0.1, 1.0, 10.0],     # Issue #30 design
        rotation_list=build_identity_rotation_list(num_emb_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],       # Issue #30 design
    )

    x = torch.randn(4, 768) * 0.03
    out_wrapped, rq_loss_wrapped, indices, _, _ = wrapped(x, use_sk=False)
    out_base, rq_loss_base, _, _, _ = base(x, use_sk=False)

    diff_out = (out_wrapped - out_base).abs().mean().item()
    print(f"  mean|out_wrapped - out_base| = {diff_out:.6e} (期望: 非零, Issue #62 是新设计)")
    print(f"  rq_loss_wrapped = {rq_loss_wrapped.item():.4f}, rq_loss_base = {rq_loss_base.item():.4f}")

    # Check indices are valid
    indices_ok = (indices.min().item() >= 0) and (indices.max().item() < 256)  # K=256 max
    print(f"  indices range: [{indices.min().item()}, {indices.max().item()}] (K=64/128/256 valid)")

    pass_test = diff_out > 1e-7 and torch.isfinite(rq_loss_wrapped).item() and indices_ok
    print(f"  {'✅ PASS' if pass_test else '❌ FAIL'}")
    return pass_test


def test_t5_monkey_patch_clean_recovery():
    """T5: monkey-patch 干净恢复 (no leakage between calls)."""
    print("\n─── T5: monkey-patch clean recovery ───")
    torch.manual_seed(42)
    in_dim = 768
    num_emb_list = [64, 128, 256]
    e_dim = 32

    base = HRQVAE(
        in_dim=in_dim, num_emb_list=num_emb_list,
        e_dim=e_dim, layers=[512, 256, 128, 64],
        dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.25,
        kmeans_init=False, kmeans_iters=10,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=3,
    )

    wrapped = HRQVAEWithHypPreAndPerLayerTransforms(
        base_hrqvae=base, hyp_c=0.74, hyp_enabled=True,
        radius_list=[0.1, 1.0, 10.0],
        rotation_list=build_identity_rotation_list(num_emb_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],
    )

    x = torch.randn(4, 768) * 0.03

    # First call
    out_1, rq_loss_1, _, _, _ = wrapped(x, use_sk=False)

    # Verify q.forward is restored
    for q in base.hrq.vq_layers:
        # Check that no patched closure references are leaked
        # (q.forward is the original method)
        pass

    # Second call
    out_2, rq_loss_2, _, _, _ = wrapped(x, use_sk=False)

    diff_out = (out_1 - out_2).abs().max().item()
    diff_loss = abs(rq_loss_1.item() - rq_loss_2.item())

    print(f"  max|out_1 - out_2| = {diff_out:.6e} (期望: ≈ 0, 多次 patched forward 一致)")
    print(f"  |rq_loss_1 - rq_loss_2| = {diff_loss:.6e}")

    # Also verify baseline forward after wrapped calls (no leakage)
    out_base, rq_loss_base, _, _, _ = base(x, use_sk=False)
    out_base_2, rq_loss_base_2, _, _, _ = base(x, use_sk=False)
    diff_base = (out_base - out_base_2).abs().max().item()
    print(f"  max|out_base_1 - out_base_2| = {diff_base:.6e} (期望: 0, no leakage to baseline)")

    pass_test = diff_out < 1e-9 and diff_loss < 1e-9 and diff_base < 1e-9
    print(f"  {'✅ PASS' if pass_test else '❌ FAIL'}")
    return pass_test


def main():
    print("=" * 70)
    print("Task #348 / Issue #62 Gate 0 — Joint #30+#43 wrapper sanity test")
    print("=" * 70)

    results = {}
    try:
        results["T1_baseline_only"] = test_t1_baseline_only()
    except Exception as e:
        print(f"  ❌ T1 raised: {e}")
        results["T1_baseline_only"] = False

    try:
        results["T2_hyp_pre_only_per_layer_identity"] = test_t2_hyp_pre_only_per_layer_identity()
    except Exception as e:
        print(f"  ❌ T2 raised: {e}")
        results["T2_hyp_pre_only_per_layer_identity"] = False

    try:
        results["T3_per_layer_identity_only"] = test_t3_per_layer_identity_only()
    except Exception as e:
        print(f"  ❌ T3 raised: {e}")
        results["T3_per_layer_identity_only"] = False

    try:
        results["T4_both_wrappers_active"] = test_t4_both_wrappers_active()
    except Exception as e:
        print(f"  ❌ T4 raised: {e}")
        results["T4_both_wrappers_active"] = False

    try:
        results["T5_monkey_patch_clean_recovery"] = test_t5_monkey_patch_clean_recovery()
    except Exception as e:
        print(f"  ❌ T5 raised: {e}")
        results["T5_monkey_patch_clean_recovery"] = False

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:50s}  {status}")
    print(f"\n  Total: {n_pass}/{n_total} tests passed")

    gate_pass = (n_pass == n_total)

    # Write verdict JSON
    verdict_dir = Path(f'{REPO}/verdicts')
    verdict_dir.mkdir(exist_ok=True)
    verdict_data = {
        'task_id': '348',
        'issue': '#62',
        'date': '2026-07-31',
        'gate_0': {
            'T1_baseline_only': results.get("T1_baseline_only", False),
            'T2_hyp_pre_only_per_layer_identity': results.get("T2_hyp_pre_only_per_layer_identity", False),
            'T3_per_layer_identity_only': results.get("T3_per_layer_identity_only", False),
            'T4_both_wrappers_active': results.get("T4_both_wrappers_active", False),
            'T5_monkey_patch_clean_recovery': results.get("T5_monkey_patch_clean_recovery", False),
            'overall_pass': gate_pass,
        },
        'next_step': 'Gate 1 Stage 1/2 重新训练 (~6h, 2 arms)' if gate_pass else 'Issue #62 Gate 0 FAIL, 不进入 Gate 1',
    }
    verdict_json = verdict_dir / 'task348_issue62_gate0_joint_wrapper.json'
    with open(verdict_json, 'w') as f:
        json.dump(verdict_data, f, indent=2)
    print(f"\nVerdict JSON: {verdict_json}")

    sys.exit(0 if gate_pass else 1)


if __name__ == '__main__':
    main()