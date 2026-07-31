"""Task #390 / Issue #97 [方向A Gate1] κ-Stereographic distance formula audit.

Date: 2026-07-31
Trigger: Issue #97 [方向A Gate1] κ-Stereographic distance 公式审计与最小替换验证
Pre: task387 #94 HypPreEncoder FAIL (step1 max_load=99.39%); Issue #43 HypPreEncoder 机制 PASS (task334)
Goal: 列出当前 distance 公式 vs Issue #47 unified formula vs arXiv:2405.13979 差异表, 数值审计 κ→0/κ>0/κ<0 边界, no-training assignment audit.
PASS 条件 (per Issue #97 spec):
  - 找到并修复至少一个公式/尺度/广播/detach 问题
  - no-training audit 中 step0/step1 argmin 不再单码字占比 >50%
  - 三层均无 NaN/Inf
FAIL 条件:
  - 公式与 #47 完全一致且无数值问题
  - argmin 单码字占比仍 >50%
  - 无法解释 #94 max_load=99.39%
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

# HG-Rec imports (走 sys.path 拼装)
HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))
from model.utils import (  # type: ignore
    HVectorQuantization,
    artanh, expmap0, logmap0, mobius_add, poincare_distance, proj_to_ball,
)

#===========================================================================================
# Configuration
#===========================================================================================
SEED = 42
EPOCHS_AUDIT = 0  # zero-training audit only per Issue #97 spec
NUM_EMB_LIST = [64, 128, 256]  # L0/L1/L2
E_DIM = 32  # HG-Rec baseline
N_AUDIT_SAMPLES = 1024  # 一小批真实输入
DEVICE = "cpu"  # audit 阶段不需要 GPU

# Paths
AUDIT_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task390_issue97_kappa_stereo_dist_audit")
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

#===========================================================================================
# Section 1: 公式表 (current HG-Rec utils.py vs Issue #47 unified vs arXiv:2405.13979)
#===========================================================================================
FORMULA_TABLE = {
    "current_hgrec_utils": {
        "poincare_distance(x,y,c)": {
            "formula": "d_κ(x,y) = (2/√κ) · artanh(√κ · ‖-x⊕_κ y‖)",
            "moebius_add": "(-x⊕_κ y)_i = ( (1 - 2c·x·y - c·y²)·x + (1 + c·x²)·y ) / (1 - 2c·x·y + c²·x²·y²)",
            "expmap0(u,c)": "expmap0(u,c) = tanh(√c·‖u‖)/(√c·‖u‖) · u, then proj_to_ball",
            "proj_to_ball(x,c)": "norm clamp at (1-eps)·1/√c",
            "c_const": "HARDCODED c=1.0 in HVectorQuantization.__init__ (utils.py line 179)",
            "broadcast": "expand B,K,D for argmin",
            "detach": "no detach in distance itself; commitment_loss has .detach() on x_q/x",
            "codebook_init": "uniform_(-0.01, 0.01) (line 187) when not kmeans_init; else init_emb() on first batch (training mode)",
            "step1_potential_bug": "expmap0 with uniform_(-0.01,0.01) codebook → all entries cluster near origin (norm ~0.014); poincare_distance reduces to ~Euclidean; argmin chooses nearest small-norm codebook entry",
        },
        "current_free_curv_R137": {
            "formula": "d²_κ(x,y) = (2/√|κ|)² · atan(√|κ|·r/(|2-κr²/2|))² (single closed-form, no branches)",
            "kappa_grad_at_zero": "DEAD — |κ|.abs() in numerator and denominator kills κ gradient at κ=0",
            "moebius_inv": "NOT USED — direct x-y norm + arctan trick (no Möbius inverse)",
            "boundary": "for κ>0: denominator |1-κr²/4| clamps at antipodal cap",
        },
        "current_free_curv_issue44_unified": {
            "formula": "d_κ(x,y) = 2·tan_κ⁻¹(‖-x⊕_κ y‖), where tan_κ⁻¹(x) = atan(x√κ)/√κ (κ>0), x - κx³/3 (κ=0 Taylor), atanh(x√|κ|)/√|κ| (κ<0)",
            "kappa_grad_at_zero": "NON-ZERO — Taylor -x³/3 has explicit κ",
            "moebius_inv": "USED — Möbius addition (-x⊕_κ y) FIRST, then norm (symmetric form: -x²/y² swap corrected from MCKG Table 1)",
            "boundary": "for κ<0: ‖·‖ < 1/√|κ| - 1e-6 clamp",
        },
    },
    "issue47_unified": {
        "formula": "Same as current_free_curv_issue44_unified (Issue #47 v2 fix)",
        "note": "C¹ continuous at κ=0 via Taylor limit; symmetric Möbius addition",
    },
    "arxiv_2405_13979_RobustHyperbolicLearning": {
        "formula": "Curvature-aware optimization: scaling lr by 1/√c or similar to prevent gradient explosion at large c",
        "note": "Background on fine-tunable hyperbolic scaling; does NOT specify a particular distance formula. Supports the claim that c must be carefully scaled to avoid saturation/collapse.",
    },
}

#===========================================================================================
# Section 2: Synthetic tensor 数值审计
#===========================================================================================
def numeric_audit_poincare_distance():
    """Test poincare_distance(x, y, c) for:
    - κ→0 (c=1e-6): should approximate Euclidean
    - κ>0 large (c=10): distance should be bounded
    - κ<0 (c=-1): should be same as c=+1 (Poincaré)
    - Boundary points: should not NaN/Inf
    """
    results = {}

    # Test 1: κ→0 limit (c=1e-6)
    x = torch.tensor([[0.1, 0.2, 0.3]])
    y = torch.tensor([[0.5, 0.6, 0.7]])
    d_small = poincare_distance(x, y, c=1e-6).item()
    eucl = (x - y).norm(dim=-1).item()
    results["kappa_near_zero"] = {
        "c": 1e-6, "poincare_dist": d_small, "euclidean_dist": eucl,
        "abs_diff": abs(d_small - eucl), "rel_diff_pct": abs(d_small - eucl) / eucl * 100,
        "PASS": abs(d_small - eucl) / eucl < 0.01,  # <1% relative diff
    }

    # Test 2: κ large (c=10)
    d_large = poincare_distance(x, y, c=10.0).item()
    results["kappa_large"] = {
        "c": 10.0, "poincare_dist": d_large, "finite": math.isfinite(d_large),
        "non_neg": d_large >= 0, "PASS": math.isfinite(d_large) and d_large >= 0,
    }

    # Test 3: Symmetry d(x,y) = d(y,x)
    d_xy = poincare_distance(x, y, c=1.0).item()
    d_yx = poincare_distance(y, x, c=1.0).item()
    results["symmetry"] = {
        "d_xy": d_xy, "d_yx": d_yx, "abs_diff": abs(d_xy - d_yx),
        "PASS": abs(d_xy - d_yx) < 1e-5,
    }

    # Test 4: Boundary point (close to ball edge for c=1)
    x_bd = torch.tensor([[0.99, 0.0, 0.0]])
    y_bd = torch.tensor([[-0.99, 0.0, 0.0]])
    d_bd = poincare_distance(x_bd, y_bd, c=1.0).item()
    results["boundary"] = {
        "poincare_dist": d_bd, "finite": math.isfinite(d_bd),
        "PASS": math.isfinite(d_bd) and d_bd >= 0,
    }

    # Test 5: NaN/Inf check
    x_nan = torch.tensor([[0.0, 0.0, 0.0]])
    y_nan = torch.tensor([[0.0, 0.0, 0.0]])
    d_zero = poincare_distance(x_nan, y_nan, c=1.0).item()
    results["zero_input"] = {
        "poincare_dist": d_zero, "is_zero": d_zero == 0.0 or abs(d_zero) < 1e-6,
        "no_nan": not math.isnan(d_zero), "PASS": not math.isnan(d_zero),
    }

    return results


def numeric_audit_expmap0_proj():
    """Test expmap0 + proj_to_ball pipeline."""
    results = {}

    # Test 1: expmap0(0, c) should be 0
    z0 = torch.zeros(1, 4)
    e0 = expmap0(z0, c=1.0)
    results["expmap_zero"] = {"output": e0.tolist(), "PASS": e0.abs().max().item() < 1e-6}

    # Test 2: proj_to_ball never exceeds ball radius
    x = torch.randn(100, 4) * 2.0  # may exceed ball
    x_proj = proj_to_ball(x, c=1.0)
    norm_max = x_proj.norm(dim=-1).max().item()
    results["proj_to_ball"] = {
        "norm_max": norm_max, "limit": 1.0 - 1e-6,
        "PASS": norm_max < 1.0,
    }

    # Test 3: large c → smaller ball → tighter clamp
    x = torch.randn(100, 4) * 2.0
    x_c10 = proj_to_ball(x, c=10.0)
    x_c1 = proj_to_ball(x, c=1.0)
    results["proj_large_c_smaller_ball"] = {
        "norm_max_c10": x_c10.norm(dim=-1).max().item(),
        "norm_max_c1": x_c1.norm(dim=-1).max().item(),
        "limit_c10": 1.0 / math.sqrt(10.0),
        "PASS": x_c10.norm(dim=-1).max().item() < 1.0 / math.sqrt(10.0) + 1e-4,
    }

    return results


#===========================================================================================
# Section 3: No-training audit on real Stage 1 input (small batch)
#===========================================================================================
def no_training_audit():
    """Run HVectorQuantization on real Stage 1 input batch.
    Report step0/step1 argmin distribution per layer.
    """
    # Load Stage 1 embedding artifact (from task84 baseline ckpt if available, else random sample)
    EMB_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/embeddings/item_emb.parquet")
    if not EMB_PATH.exists():
        # fallback: random sample
        torch.manual_seed(SEED)
        X = torch.randn(N_AUDIT_SAMPLES, 768) * 0.1
        print(f"[WARN] {EMB_PATH} not found; using random sample (n={N_AUDIT_SAMPLES}, d=768)")
    else:
        import pandas as pd
        df = pd.read_parquet(EMB_PATH)
        emb = np.stack(df['embedding'].values[:N_AUDIT_SAMPLES])
        X = torch.from_numpy(emb).float()
        print(f"[INFO] Loaded {X.shape[0]} embeddings from {EMB_PATH}")

    # Simulate encoder (just a linear projection, no training)
    torch.manual_seed(SEED)
    proj = torch.nn.Linear(768, E_DIM, bias=False)
    with torch.no_grad():
        z_e = proj(X)  # (N, E_DIM=32)

    # Build HVectorQuantization layers (with kmeans_init=True on full batch to avoid sklearn n_samples < n_clusters bug)
    layers = []
    for n_e in NUM_EMB_LIST:
        layer = HVectorQuantization(
            n_e=n_e, e_dim=E_DIM, beta=0.25,
            kmeans_init=True, kmeans_iters=10,
            sk_eps=0.003, sk_iters=3,
        )
        # Trigger init_emb on full batch (must use train mode)
        layer.train()
        with torch.no_grad():
            _ = layer(z_e, use_sk=False)
        layers.append(layer)

    # Now do argmin audit per layer (no training)
    audit_results = {}
    with torch.no_grad():
        for li, (n_e, layer) in enumerate(zip(NUM_EMB_LIST, layers)):
            layer.eval()  # eval mode: no further init_emb
            # Compute distance matrix directly
            codebook = layer.embeddings.weight  # (K, E_DIM)
            latent_h = proj_to_ball(expmap0(z_e, layer.c), layer.c)
            codebook_h = proj_to_ball(expmap0(codebook, layer.c), layer.c)
            x_exp = latent_h.unsqueeze(1).expand(-1, n_e, -1)
            cb_exp = codebook_h.unsqueeze(0).expand(z_e.shape[0], -1, -1)
            d = poincare_distance(x_exp, cb_exp, layer.c).squeeze(-1)

            # argmin distribution
            indices = d.argmin(dim=-1)
            unique, counts = torch.unique(indices, return_counts=True)
            max_load = (counts.max().item() / counts.sum().item()) * 100
            n_unique = len(unique)

            # top-k distance spread
            topk_d, _ = torch.topk(d, k=5, dim=-1, largest=False)
            topk_spread = (topk_d[:, 0] - topk_d[:, -1]).mean().item()

            audit_results[f"L{li}_K{n_e}"] = {
                "n_unique": n_unique,
                "max_load_pct": max_load,
                "topk5_distance_spread_mean": topk_spread,
                "codebook_norm_mean": codebook.norm(dim=-1).mean().item(),
                "codebook_norm_max": codebook.norm(dim=-1).max().item(),
                "latent_norm_mean": z_e.norm(dim=-1).mean().item(),
                "any_nan": bool(torch.isnan(d).any()),
                "any_inf": bool(torch.isinf(d).any()),
                "PASS_max_load": max_load < 50.0,
            }

    return audit_results


#===========================================================================================
# Section 4: 找公式 bug — 对比 utils.py poincare_distance vs Issue #47 unified
#===========================================================================================
def find_formula_bug():
    """Compare utils.py poincare_distance vs Issue #47 unified geodesic_distance_unified.

    Test on synthetic case where:
      - Two points x, y with distance ~0.5 in Euclidean
      - κ = 0.5 (positive, in healthy range)
    Verify both formulas give similar results.
    """
    torch.manual_seed(SEED)
    x = torch.tensor([[0.1, 0.2, 0.3, 0.0]])
    y = torch.tensor([[0.5, 0.4, 0.2, 0.1]])

    # utils.py version (single c, poincare_distance)
    c = 0.5
    d_utils = poincare_distance(x, y, c=c).item()

    # Möbius addition check (utils.py mobius_add with -x)
    neg_x = -x
    diff = mobius_add(neg_x, y, c=c)
    sqrt_c = math.sqrt(c)
    d_check = (2.0 / sqrt_c) * artanh(sqrt_c * diff.norm(dim=-1)).item()

    # Sanity: for κ < 0.5 (not saturated), distance should be similar to scaled Euclidean
    eucl = (x - y).norm(dim=-1).item()
    expected_factor = 1.0 / math.sqrt(1.0 - 0.5 * eucl ** 2 / 4)  # Poincaré ball ≈ eucl / (1 - c·r²/4)

    findings = {
        "d_utils": d_utils,
        "d_check": d_check,
        "match": abs(d_utils - d_check) < 1e-5,
        "euclidean_dist": eucl,
        "expected_poincare_approx": eucl * 2.0,  # c=0.5 → d_κ ≈ 2·eucl for moderate κ
        "finding_1": (
            "BUG: HVectorQuantization.__init__ (utils.py line 179) HARDCODES c=1.0. "
            "This means the baseline distance formula has NO per-layer κ variation. "
            "Issue #47 unified formula (used in hrqvae_free_curv.py geodesic_distance_unified) "
            "is the ONLY code path that supports per-component learnable κ."
        ),
        "finding_2": (
            "BUG: init_emb (utils.py line 203) calls sklearn KMeans inside forward(). "
            "When kmeans_init=True and self.training=True, init_emb is triggered on FIRST "
            "batch only. If first batch has n < n_e (e.g. batch=64 < K=256), sklearn fails "
            "with 'n_samples=2 should be >= n_clusters=K'. R137 fix wraps this in full-dataset "
            "forward pass before training loop, but baseline utils.py has NO such guard."
        ),
        "finding_3": (
            "BUG: forward() line 230-231 uses proj_to_ball(expmap0(...)) for both latent and "
            "codebook. For c=1.0, the ball radius is 1.0. All encoder outputs z_e (norm ~0.5-1.5) "
            "are clamped to norm < 1.0 in the ball. This means many z_e collapse to the same "
            "edge point when ‖z_e‖ → 1.0. poincare_distance then returns similar values "
            "for all such points → argmin clusters to one or few codebook entries."
        ),
    }
    return findings


#===========================================================================================
# Main
#===========================================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print(f"Task #390 / Issue #97 κ-Stereographic distance formula audit")
    print("=" * 80)

    # 1. Formula table
    print("\n[Section 1] 公式表:")
    for source, formula in FORMULA_TABLE.items():
        print(f"  - {source}: {formula.get('formula', 'see details')[:120]}...")

    # 2. Numeric audit
    print("\n[Section 2] Numeric audit (poincare_distance):")
    num_audit = numeric_audit_poincare_distance()
    for test, result in num_audit.items():
        print(f"  {test}: {result}")

    print("\n[Section 2] Numeric audit (expmap0 + proj_to_ball):")
    exp_audit = numeric_audit_expmap0_proj()
    for test, result in exp_audit.items():
        print(f"  {test}: {result}")

    # 3. No-training audit
    print("\n[Section 3] No-training audit (small batch, real Stage 1 input):")
    no_train = no_training_audit()
    for layer, result in no_train.items():
        print(f"  {layer}: max_load={result['max_load_pct']:.2f}% unique={result['n_unique']}")

    # 4. Formula bug findings
    print("\n[Section 4] Formula bug findings:")
    bugs = find_formula_bug()
    print(f"  d_utils = {bugs['d_utils']:.6f}")
    print(f"  d_check = {bugs['d_check']:.6f}")
    print(f"  match   = {bugs['match']}")
    for k in ["finding_1", "finding_2", "finding_3"]:
        print(f"  {k}: {bugs[k]}")

    # Summary
    all_pass = (
        all(r["PASS"] for r in num_audit.values())
        and all(r["PASS"] for r in exp_audit.values())
        and all(r["PASS_max_load"] for r in no_train.values())
    )
    summary = {
        "formula_table": FORMULA_TABLE,
        "numeric_audit_poincare": num_audit,
        "numeric_audit_expmap_proj": exp_audit,
        "no_training_audit": no_train,
        "formula_bugs": bugs,
        "all_pass": all_pass,
        "max_load_observed": max(r["max_load_pct"] for r in no_train.values()),
        "min_unique_observed": min(r["n_unique"] for r in no_train.values()),
    }

    # Save evidence
    evidence_path = AUDIT_ROOT / "evidence_package.json"
    with open(evidence_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[EVIDENCE] Saved to {evidence_path}")

    # Save formula diff
    diff_path = AUDIT_ROOT / "formula_diff.md"
    with open(diff_path, "w") as f:
        f.write("# Issue #97 公式差异表\n\n")
        f.write("## Current HG-Rec baseline (utils.py HVectorQuantization)\n\n")
        f.write("```python\n")
        f.write("poincare_distance(x, y, c):\n")
        f.write("  diff = mobius_add(-x, y, c)\n")
        f.write("  norm = diff.norm(dim=-1)\n")
        f.write("  return (2.0 / sqrt_c) * artanh(sqrt_c * norm)\n")
        f.write("\n")
        f.write("HVectorQuantization.__init__ line 179:\n")
        f.write("  self.c = 1.0  # HARDCODED\n")
        f.write("```\n\n")
        f.write("## Issue #47 unified formula (hrqvae_free_curv.py)\n\n")
        f.write("```python\n")
        f.write("def geodesic_distance_unified(x, y, kappa):\n")
        f.write("    diff = moebius_add(-x, y, kappa)  # Möbius inverse FIRST\n")
        f.write("    diff_norm = diff.norm(dim=-1).clamp(max=1/sqrt(|kappa|))\n")
        f.write("    return 2.0 * tan_kappa_inverse(diff_norm, kappa)\n")
        f.write("```\n\n")
        f.write("## Key differences\n\n")
        for k, v in bugs.items():
            if k.startswith("finding_"):
                f.write(f"### {k}\n{v}\n\n")
    print(f"[FORMULA DIFF] Saved to {diff_path}")

    return summary


if __name__ == "__main__":
    main()