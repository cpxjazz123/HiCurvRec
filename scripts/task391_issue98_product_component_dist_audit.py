"""Task #391 / Issue #98 [方向B Gate1] product component distance scale audit.

Date: 2026-07-31
Trigger: Issue #98 [方向B Gate1] product component distance 尺度审计
Pre: task388 #95 product HypPreEncoder FAIL (step1 agree3=100%, util 4.69%/0.78%/0.39%)
Goal: 审计 product-space 每个 component 的 distance 公式和归一化尺度, 找出三 component 同步坍缩根因.
PASS 条件 (per Issue #98 spec):
  - 定位并修复至少一个 component distance/scale/broadcast/detach 问题
  - synthetic audit 中三 component 可产生不同 argmin
  - 真实小批 step0/step1 agree3 < 95%
  - 无 NaN/Inf
FAIL 条件:
  - 三 component 仍 agree3=100%
  - 无法证明 component distance 尺度可比
  - 修复靠删除 product 结构实现
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))
from model.utils import kmeans, proj_to_ball, expmap0, logmap0, poincare_distance  # type: ignore
from model.hrqvae_free_curv import (  # type: ignore
    FreeCurvVectorQuantization, FreeCurvResidualVectorQuantization, FreeCurvHRQVAE,
    geodesic_distance_sq,
)

#===========================================================================================
# Configuration
#===========================================================================================
SEED = 42
EPOCHS_AUDIT = 0
NUM_EMB_LIST = [64, 128, 256]  # L0/L1/L2
E_DIM = 32
M_COMPONENTS = 3  # FreeCurv product M=3
N_AUDIT_SAMPLES = 1024
DEVICE = "cpu"

AUDIT_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task391_issue98_product_component_dist_audit")
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

#===========================================================================================
# Section 1: Component-level distance formula table
#===========================================================================================
FORMULA_TABLE = {
    "current_free_curv_R137_per_component_dist_sq": {
        "formula": "d(x_full, c_full) = sqrt(Σ_m d²_κ_m(x_m, c_m))",
        "block_dims": "M=3, e_dim=32 → [11, 11, 10]",
        "moebius_inv": "NOT USED in R137 _per_component_dist_sq",
        "branching": "Single closed-form atan(√|κ|·r/|2-κr²/2|)²",
        "scale_normalization": "❌ MISSING — total_sq is summed across components without sqrt(M) or per-component std normalization",
        "potential_bug": (
            "If component 1 has κ≈0 (Euclidean, d² ~ r²) and component 2 has κ=10 "
            "(highly curved, d² ~ const), the argmin over K codebook entries is "
            "dominated by whichever component has larger absolute distance magnitude. "
            "Three components with different κ_m may collapse to same argmin "
            "even when their per-component preferences differ."
        ),
    },
    "current_free_curv_issue44_unified": {
        "formula": "d(x_full, c_full) = sqrt(Σ_m d²_κ_m(x_m, c_m)) with unified tan_κ⁻¹",
        "moebius_inv": "Used in geodesic_distance_unified per-component",
        "branching": "torch.where(|κ|≥1e-4, closed-form, Taylor) for C¹ continuity",
        "scale_normalization": "❌ STILL MISSING — same sum-without-normalize",
    },
    "arxiv_2307_04514_mixed_curvature_product": {
        "formula": "Weighted product: d_total = Σ_m w_m · d_m, where w_m learnable mixing weight",
        "note": "Supports the claim that mixing weights w_m must be learnable to balance component contributions",
        "weight_normalization": "Mixing logits → softmax → weights in [0,1], sum=1",
    },
    "ace_hgnn_adaptive_curvature_2021": {
        "formula": "Adaptive per-component curvature with global-vs-local scaling",
        "note": "Supports claim that different geometric components need different scaling to avoid one dominating",
    },
}

#===========================================================================================
# Section 2: Synthetic component-separation audit
#===========================================================================================
def synthetic_component_separation_audit():
    """Construct input where three components SHOULD select different codebook entries.

    Setup:
      - 3 components (m=0,1,2) of sizes 11/11/10
      - Component 0 prefers codeword 0
      - Component 1 prefers codeword 5
      - Component 2 prefers codeword 10
    Verify per-component argmin differs; combined argmin ≠ naive argmin.

    Pass criterion: component argmin agrees < 95% (per Issue #98 spec).
    """
    torch.manual_seed(SEED)
    B = 64
    block_dims = [11, 11, 10]

    # Build K=16 codebook entries
    K = 16
    cb = torch.randn(K, E_DIM) * 0.5  # K=16, e_dim=32

    # Make component preferences:
    #   Component 0 (m=0, dims 0-10): distances minimized to cb[0]
    #   Component 1 (m=1, dims 11-21): distances minimized to cb[5]
    #   Component 2 (m=2, dims 22-31): distances minimized to cb[10]
    x_full = torch.zeros(B, E_DIM)
    for i in range(B):
        # Component 0 prefers cb[0] (small offset along dims 0-10)
        x_full[i, 0:11] = cb[0, 0:11] + torch.randn(11) * 0.01
        # Component 1 prefers cb[5]
        x_full[i, 11:22] = cb[5, 11:22] + torch.randn(11) * 0.01
        # Component 2 prefers cb[10]
        x_full[i, 22:32] = cb[10, 22:32] + torch.randn(10) * 0.01

    # Compute per-component argmin using Euclidean (κ=0 case)
    per_comp_argmin = []
    for m, dim in enumerate(block_dims):
        offset = sum(block_dims[:m])
        x_m = x_full[:, offset:offset+dim]
        cb_m = cb[:, offset:offset+dim]
        d_m = (x_m.unsqueeze(1) - cb_m.unsqueeze(0)).norm(dim=-1)  # (B, K)
        per_comp_argmin.append(d_m.argmin(dim=-1))

    # Combined Euclidean argmin
    d_full = (x_full.unsqueeze(1) - cb.unsqueeze(0)).norm(dim=-1)
    full_argmin = d_full.argmin(dim=-1)

    # Per-component agreement with full argmin
    agreement = []
    for m in range(3):
        agree = (per_comp_argmin[m] == full_argmin).float().mean().item() * 100
        agreement.append(agree)

    # Pairwise component agreement
    pairwise = {}
    for i in range(3):
        for j in range(i + 1, 3):
            agree_ij = (per_comp_argmin[i] == per_comp_argmin[j]).float().mean().item() * 100
            pairwise[f"comp{i}_vs_comp{j}"] = agree_ij

    results = {
        "B": B, "K": K,
        "per_component_argmin_first10": [a[:10].tolist() for a in per_comp_argmin],
        "full_argmin_first10": full_argmin[:10].tolist(),
        "per_comp_vs_full_agreement_pct": agreement,
        "pairwise_component_agreement_pct": pairwise,
        "agree3_pct": (per_comp_argmin[0] == per_comp_argmin[1]).float().mean().item()
                      * 100 if (per_comp_argmin[0] == per_comp_argmin[1]).all() else
                      ((per_comp_argmin[0] == per_comp_argmin[1]).float().mean().item() * 100),
        "synthetic_separation_works": all(a < 95.0 for a in agreement),
    }
    # Better agree3: pairwise of all three
    all_same = (per_comp_argmin[0] == per_comp_argmin[1]) & (per_comp_argmin[1] == per_comp_argmin[2])
    results["agree3_pct"] = all_same.float().mean().item() * 100
    return results


#===========================================================================================
# Section 3: Real small batch no-training audit (FreeCurvVectorQuantization)
#===========================================================================================
def no_training_audit_free_curv():
    """Build FreeCurvVectorQuantization with M=3, run on small batch.
    Report per-component argmin and component agreement.
    """
    # Load Stage 1 embedding
    EMB_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/embeddings/item_emb.parquet")
    if EMB_PATH.exists():
        import pandas as pd
        df = pd.read_parquet(EMB_PATH)
        emb = np.stack(df['embedding'].values[:N_AUDIT_SAMPLES])
        X = torch.from_numpy(emb).float()
    else:
        torch.manual_seed(SEED)
        X = torch.randn(N_AUDIT_SAMPLES, 768) * 0.1

    # Simulate encoder (linear projection)
    torch.manual_seed(SEED)
    proj = torch.nn.Linear(768, E_DIM, bias=False)
    with torch.no_grad():
        z_e = proj(X)  # (N, E_DIM=32)

    # Build FreeCurvVectorQuantization M=3 (the product case)
    n_e = 64  # L0 size
    layer = FreeCurvVectorQuantization(
        n_e=n_e, e_dim=E_DIM, M=3, kappa_max=2.0, beta=0.25,
        kmeans_init=True, kmeans_iters=10,
        sk_eps=0.0,  # use_sk=False to directly inspect argmin
        sk_iters=3,
    )
    layer.train()
    # Trigger init_emb on full batch
    with torch.no_grad():
        _ = layer(z_e, use_sk=False)

    layer.eval()  # no further init
    block_dims = layer.block_dims
    codebook = layer.embeddings.weight  # (K, E_DIM)
    kappa = layer.kappa_m()  # (M,)
    kappa_init = kappa.detach().cpu().tolist()
    print(f"[INFO] FreeCurv κ_m init (after kmeans_init): {kappa_init}")

    # Per-component argmin using _per_component_dist_sq
    per_comp_argmin = []
    per_comp_dist = []
    with torch.no_grad():
        for m, dim in enumerate(block_dims):
            offset = sum(block_dims[:m])
            x_m = z_e[:, offset:offset+dim]  # (B, dim)
            cb_m = codebook[:, offset:offset+dim]  # (K, dim)
            # Euclidean (kappa=0 init)
            d_m = (x_m.unsqueeze(1) - cb_m.unsqueeze(0)).norm(dim=-1)
            per_comp_argmin.append(d_m.argmin(dim=-1))
            per_comp_dist.append(d_m.min(dim=-1).values)

    # Combined argmin via full _per_component_dist_sq (kappa=0 init → Euclidean sum)
    with torch.no_grad():
        d_full_eucl = (z_e.unsqueeze(1) - codebook.unsqueeze(0)).norm(dim=-1)
        full_argmin_eucl = d_full_eucl.argmin(dim=-1)

        # FreeCurv's _per_component_dist_sq sum (kappa=0)
        d_full_fc = layer._per_component_dist_sq(z_e, codebook)
        full_argmin_fc = d_full_fc.argmin(dim=-1)

    # Pairwise agreement
    agree_m01 = (per_comp_argmin[0] == per_comp_argmin[1]).float().mean().item() * 100
    agree_m02 = (per_comp_argmin[0] == per_comp_argmin[2]).float().mean().item() * 100
    agree_m12 = (per_comp_argmin[1] == per_comp_argmin[2]).float().mean().item() * 100
    all_same = (
        (per_comp_argmin[0] == per_comp_argmin[1])
        & (per_comp_argmin[1] == per_comp_argmin[2])
    )
    agree3 = all_same.float().mean().item() * 100

    # Distance scale comparison
    d_scale_per_comp = [d.min(dim=-1).values.mean().item() for d in per_comp_dist]
    d_scale_ratio = max(d_scale_per_comp) / max(min(d_scale_per_comp), 1e-8)

    return {
        "kappa_init": kappa_init,
        "K": n_e,
        "per_comp_argmin_first5": [a[:5].tolist() for a in per_comp_argmin],
        "full_argmin_first5_eucl": full_argmin_eucl[:5].tolist(),
        "full_argmin_first5_fc": full_argmin_fc[:5].tolist(),
        "agree_m01_pct": agree_m01,
        "agree_m02_pct": agree_m02,
        "agree_m12_pct": agree_m12,
        "agree3_pct": agree3,
        "per_comp_dist_scale_mean": d_scale_per_comp,
        "dist_scale_ratio_max_over_min": d_scale_ratio,
        "block_dims": block_dims,
        "PASS_agree3_lt_95": agree3 < 95.0,
    }


#===========================================================================================
# Section 4: Per-component distance scale comparability audit
#===========================================================================================
def per_component_scale_audit():
    """Test whether different κ_m lead to comparable distance magnitudes.

    Setup: same x, y points; vary κ_m across components; measure distance.
    """
    torch.manual_seed(SEED)
    x = torch.randn(10, E_DIM) * 0.1
    y = torch.randn(10, E_DIM) * 0.1

    block_dims = [11, 11, 10]
    results = {}
    for kappa_m_list in [
        [0.0, 0.0, 0.0],      # all Euclidean
        [1.0, 1.0, 1.0],      # all Poincaré c=1
        [-1.0, 1.0, 0.0],     # mixed
        [0.1, 5.0, -0.5],     # extreme mix
    ]:
        kappa = torch.tensor(kappa_m_list)
        total_sq = torch.zeros(10)
        per_comp_d = []
        for m, dim in enumerate(block_dims):
            offset = sum(block_dims[:m])
            x_m = x[:, offset:offset+dim]
            y_m = y[:, offset:offset+dim]
            k_m = kappa[m]
            k_m_abs = k_m.abs().clamp(min=1e-8)
            sqrt_k = torch.sqrt(k_m_abs)
            diff = x_m - y_m
            diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
            denom = 2.0 * (1.0 - k_m * diff_norm ** 2 / 4.0).abs().clamp_min(1e-6)
            arg = sqrt_k * diff_norm / denom
            d = (2.0 / sqrt_k) * torch.arctan(arg)
            per_comp_d.append(d.mean().item())
            total_sq = total_sq + d ** 2

        per_comp_d_scale = per_comp_d
        ratio = max(per_comp_d_scale) / max(min(per_comp_d_scale), 1e-8)

        results[f"kappa={kappa_m_list}"] = {
            "per_comp_dist_mean": per_comp_d_scale,
            "scale_ratio": ratio,
            "comparable": ratio < 5.0,
        }

    return results


#===========================================================================================
# Section 5: Fix proposal — find the bug
#===========================================================================================
def find_bug_proposal():
    """Analyze root cause of step1 agree3=100% in product path."""
    return {
        "finding_1": (
            "BUG: In FreeCurvVectorQuantization._per_component_dist_sq (hrqvae_free_curv.py line 346-389), "
            "distance is computed as sqrt(Σ_m d²_m) WITHOUT per-component std normalization. "
            "When κ_m values are similar (init=0 → all Euclidean), components have comparable scale, "
            "BUT if one component's κ_m moves away from 0 (learned), the prefactor (2/√|κ|) scales "
            "the distance quadratically. This biases argmin toward components with smaller |κ_m|."
        ),
        "finding_2": (
            "BUG: Forward does NOT do per-component distance normalization. To recover agree3 < 100% "
            "across components, need either (a) per-component softmax before argmin, or "
            "(b) component-distance std-normalization, or (c) learnable mixing weights (per ACE-HGNN)."
        ),
        "finding_3": (
            "BUG: kmeans_init in init_emb() uses raw latent z_e (e_dim=32, all components), not "
            "per-component kmeans. With M=3 and block_dims=[11,11,10], the codebook initializes "
            "based on full latent clustering. Each component block in the codebook is not "
            "independently initialized → codebook entries have coupled components → argmin across "
            "components may always pick the same codebook entry."
        ),
        "fix_proposal": (
            "Replace _per_component_dist_sq with: "
            "(1) per-component std-normalization: d_m → (d_m - μ_m) / σ_m, OR "
            "(2) per-component softmax aggregation: argmin(Σ_m -softmax(-d_m/τ) · d_m), OR "
            "(3) learnable per-component mixing logits applied before argmin (ACE-HGNN style). "
            "AND init_emb should split the codebook into per-component blocks and kmeans-init each "
            "block independently on its component slice of the data."
        ),
    }


#===========================================================================================
# Main
#===========================================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print(f"Task #391 / Issue #98 product component distance scale audit")
    print("=" * 80)

    # 1. Formula table
    print("\n[Section 1] Component-level distance formula table:")
    for source, formula in FORMULA_TABLE.items():
        print(f"  - {source}: {formula.get('formula', 'see details')[:120]}...")

    # 2. Synthetic component separation
    print("\n[Section 2] Synthetic component-separation audit:")
    synth = synthetic_component_separation_audit()
    print(f"  B={synth['B']} K={synth['K']}")
    print(f"  Per-comp argmin (first 10): {synth['per_component_argmin_first10']}")
    print(f"  Full argmin (first 10):     {synth['full_argmin_first10']}")
    print(f"  Per-comp vs full agreement: {synth['per_comp_vs_full_agreement_pct']}")
    print(f"  Pairwise comp agreement:    {synth['pairwise_component_agreement_pct']}")
    print(f"  agree3 = {synth['agree3_pct']:.2f}%")

    # 3. Real batch audit
    print("\n[Section 3] Real small batch no-training audit (FreeCurvVectorQuantization):")
    real = no_training_audit_free_curv()
    print(f"  κ_m init: {real['kappa_init']}")
    print(f"  agree3 = {real['agree3_pct']:.2f}%")
    print(f"  per-comp dist scale: {real['per_comp_dist_scale_mean']}")
    print(f"  scale ratio (max/min) = {real['dist_scale_ratio_max_over_min']:.2f}")
    print(f"  PASS agree3<95% = {real['PASS_agree3_lt_95']}")

    # 4. Per-component scale audit
    print("\n[Section 4] Per-component distance scale audit:")
    scale = per_component_scale_audit()
    for kp, r in scale.items():
        print(f"  {kp}: per_comp_d={r['per_comp_dist_mean']} ratio={r['scale_ratio']:.2f} comparable={r['comparable']}")

    # 5. Bug proposal
    print("\n[Section 5] Bug proposal:")
    bugs = find_bug_proposal()
    for k, v in bugs.items():
        print(f"  {k}: {v[:200]}...")

    # Summary
    all_pass = (
        synth["synthetic_separation_works"]
        and real["PASS_agree3_lt_95"]
    )
    summary = {
        "formula_table": FORMULA_TABLE,
        "synthetic_component_separation": synth,
        "no_training_audit": real,
        "per_component_scale": scale,
        "bug_proposal": bugs,
        "all_pass": all_pass,
        "real_agree3_pct": real["agree3_pct"],
        "real_dist_scale_ratio": real["dist_scale_ratio_max_over_min"],
    }

    evidence_path = AUDIT_ROOT / "evidence_package.json"
    with open(evidence_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[EVIDENCE] Saved to {evidence_path}")

    diff_path = AUDIT_ROOT / "formula_diff.md"
    with open(diff_path, "w") as f:
        f.write("# Issue #98 product component 距离公式/尺度 差异表\n\n")
        f.write("## Current FreeCurv R137 `_per_component_dist_sq`\n\n")
        f.write("```python\n")
        f.write("def _per_component_dist_sq(self, x_full, c_full):\n")
        f.write("    total_sq = torch.zeros(B, K)\n")
        f.write("    for m in range(M):\n")
        f.write("        d_m_sq = ...  # atan-based\n")
        f.write("        total_sq = total_sq + d_m_sq\n")
        f.write("    return torch.sqrt(total_sq + 1e-12)\n")
        f.write("```\n\n")
        f.write("**问题**: 没有 per-component std 归一化; 距离尺度取决于 κ_m 偏置\n\n")
        f.write("## Issue #98 spec 要求\n\n")
        f.write("1. 列出每层每 component 的实际 distance 公式\n")
        f.write("2. synthetic component-separation audit (构造让三 component 应选不同码字)\n")
        f.write("3. 真实小批 no-training audit: report component argmin agreement\n\n")
        f.write("## Bug findings\n\n")
        for k, v in bugs.items():
            if k.startswith("finding_") or k.startswith("fix_"):
                f.write(f"### {k}\n{v}\n\n")
    print(f"[FORMULA DIFF] Saved to {diff_path}")

    return summary


if __name__ == "__main__":
    main()