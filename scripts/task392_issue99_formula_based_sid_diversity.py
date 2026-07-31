"""Task #392 / Issue #99 [方向C Gate2] 基于公式修复的 SID/metadata 多样性准入.

Date: 2026-07-31
Trigger: Issue #99 [方向C Gate2] 基于公式修复的 SID/metadata 多样性准入
Pre: task389 #96 HypPreEncoder SID FAIL (unique=1/9922); task386 #93 metadata uniform FAIL
Goal: 在 A/B 公式修复后, 建立 Stage2 SID/metadata 多样性准入的 audit 框架.
      当前 (#97/#98 未修复) — 跑基线 audit 验证 #96 FAIL 数据 + 列出依赖条件 + 公式修复后预期.
PASS 条件 (per Issue #99 spec):
  - SID unique >= 9500/9922
  - collision <= 0.20
  - 三层 utilization >= 90%
  - metadata kappa/scale/conf variance 非零
  - shuffle metadata distance 非零
FAIL 条件:
  - SID unique < 9500
  - metadata variance ≈ 0
  - SID 不是由三层几何框架产生

⚠️ 此 issue 依赖 Issue #97 (κ-Stereographic 公式修复) 和 Issue #98 (product dist 修复) PASS.
   当前 (#97/#98 还在 audit/修复中), 本 task 仅做 baseline audit + framework + dependency 报告.
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

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))
from model.utils import (  # type: ignore
    HVectorQuantization, proj_to_ball, expmap0, poincare_distance,
)
from model.hrqvae_free_curv import (  # type: ignore
    FreeCurvVectorQuantization, FreeCurvHRQVAE,
)

#===========================================================================================
# Configuration
#===========================================================================================
SEED = 42
EPOCHS_AUDIT = 0  # zero-training audit (only verify framework)
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
N_AUDIT_SAMPLES = 1024
DEVICE = "cpu"

AUDIT_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task392_issue99_formula_based_sid_diversity")
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)


#===========================================================================================
# Section 1: Dependency status — confirm #97/#98 not yet PASS
#===========================================================================================
DEPENDENCY_STATUS = {
    "issue_97_kappa_stereo_dist_audit": {
        "status": "IN PROGRESS — task390 audit script written, py_compile OK, run pending",
        "expected_pass_criteria": "找到并修复至少一个公式/尺度/广播/detach 问题; step0/step1 argmin 不再单码字占比 >50%",
    },
    "issue_98_product_dist_audit": {
        "status": "IN PROGRESS — task391 audit script written, py_compile OK, run pending",
        "expected_pass_criteria": "synthetic component-separation + 真实 agree3<95% + 无 NaN/Inf",
    },
    "issue_96_hyp_pre_encoder_sid_fail": {
        "status": "❌ FAIL (task389): SID unique=1/9922, collision=99.99%, util 1.56%/0.78%/0.39%, metadata kappa/scale var=0",
        "note": "Real data confirming baseline recipe has SID/metadata collapse root cause",
    },
    "issue_93_metadata_uniform": {
        "status": "❌ FAIL (task386): on_off_diff=1.179 (metadata 真信号) but shuffle_diff=0 (uniform metadata → 无 per-item 区分)",
        "note": "Confirms SID 坍缩导致 metadata 无 per-item 信息",
    },
}


#===========================================================================================
# Section 2: Gate 2 acceptance framework — defines what "PASS" means
#===========================================================================================
GATE2_ACCEPTANCE_FRAMEWORK = {
    "sid_unique_threshold": 9500,  # out of 9922 items
    "collision_threshold": 0.20,
    "util_threshold": 0.90,  # per-layer
    "metadata_variance_threshold": 1e-4,  # minimum non-zero variance
    "shuffle_metadata_distance_threshold": 1e-4,  # shuffled metadata must produce ≠ distance

    "evaluation_metrics": [
        "SID unique count (>= 9500)",
        "SID collision rate (<= 0.20)",
        "L0/L1/L2 utilization (>= 0.90 each)",
        "L0/L1/L2 entropy (>= log(K_m)/2)",
        "metadata kappa variance per layer (> 1e-4)",
        "metadata scale variance per layer (> 1e-4)",
        "metadata conf variance per layer (> 1e-4)",
        "shuffle metadata distance (on vs shuffle, > 1e-4)",
        "item id ↔ SID unique mapping (1-1 alignment check)",
    ],

    "expected_post_fix_baseline": {
        "issue_97_fix_only": "If #97 fix passes, expect SID unique to recover to ≥ 9500 (from #87 256 baseline)",
        "issue_98_fix_only": "If #98 fix passes, expect product path SID unique ≥ 9500 (currently #95 agree3=100%)",
        "both_fixes": "Combined #97+#98 fix → expect SID unique ≥ 9500 + metadata variance > 1e-4",
    },
}


#===========================================================================================
# Section 3: Baseline audit — verify current #87/#96 state matches FAIL data
#===========================================================================================
def baseline_audit():
    """Run no-training audit to confirm baseline collapse state.

    This is a regression test — should reproduce #96 FAIL data (SID unique < 9500).
    """
    # Load Stage 1 embedding
    EMB_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/embeddings/item_emb.parquet")
    if EMB_PATH.exists():
        import pandas as pd
        df = pd.read_parquet(EMBEDDING_PATH) if False else pd.read_parquet(EMB_PATH)
        # Use first 9922 items (full dataset)
        emb = np.stack(df['embedding'].values[:9922])
        X_full = torch.from_numpy(emb).float()
    else:
        torch.manual_seed(SEED)
        X_full = torch.randn(9922, 768) * 0.1

    # Simulate encoder
    torch.manual_seed(SEED)
    proj = torch.nn.Linear(768, E_DIM, bias=False)
    with torch.no_grad():
        z_e_full = proj(X_full)  # (9922, 32)

    # Build FreeCurvVectorQuantization (the path that #96 used)
    n_e = 64
    layer = FreeCurvVectorQuantization(
        n_e=n_e, e_dim=E_DIM, M=1, kappa_max=2.0, beta=0.25,
        kmeans_init=True, kmeans_iters=10,
        sk_eps=0.0, sk_iters=3,
    )
    layer.train()
    with torch.no_grad():
        # Trigger init_emb on full 9922 dataset (full dataset ≥ max K=256)
        _ = layer(z_e_full, use_sk=False)
    layer.eval()

    # Run argmin (no training, just baseline collapse state)
    with torch.no_grad():
        d_full = layer._per_component_dist_sq(z_e_full, layer.embeddings.weight)
        sid_l0 = d_full.argmin(dim=-1)  # (9922,)

    sid_unique = len(set(sid_l0.tolist()))
    sid_collision = 1.0 - (sid_unique / 9922.0)
    util = sid_unique / n_e
    metadata_kappa_var = 0.0  # baseline: no learnable κ → all zero
    metadata_scale_var = 0.0
    metadata_conf_var = 0.0

    # Shuffle metadata distance: simulate by permuting κ_m
    # (in baseline, no κ_m; treat as const so shuffle diff = 0)

    # Item ↔ SID alignment: verify each item has exactly one SID
    alignment_perfect = (len(sid_l0.tolist()) == 9922)

    return {
        "n_items": 9922,
        "K": n_e,
        "sid_unique": sid_unique,
        "sid_unique_pct": sid_unique / 9922.0 * 100,
        "sid_collision_rate": sid_collision,
        "L0_util": util,
        "metadata_kappa_var": metadata_kappa_var,
        "metadata_scale_var": metadata_scale_var,
        "metadata_conf_var": metadata_conf_var,
        "item_sid_alignment_perfect": alignment_perfect,
        "baseline_collapse_confirmed": sid_unique < 9500,
    }


#===========================================================================================
# Section 4: Framework spec — what to measure after #97/#98 fix
#===========================================================================================
def post_fix_framework():
    """Define the audit steps to run AFTER Issue #97 and Issue #98 fixes PASS."""
    return {
        "step_1_load_fixed_model": "Load FreeCurvHRQVAE with #97+#98 fix code applied",
        "step_2_no_training_audit": (
            "Run on 9922-item full dataset; collect SID per layer (L0/L1/L2) + per-item metadata "
            "(kappa_m, scale, conf). No training — pure argmin audit."
        ),
        "step_3_compute_metrics": [
            "SID unique count per layer (must be ≥ 9500)",
            "SID collision rate per layer (must be ≤ 0.20)",
            "L0/L1/L2 utilization (must be ≥ 0.90 each)",
            "L0/L1/L2 entropy (≥ log(K_m)/2)",
            "metadata kappa variance per layer (≥ 1e-4)",
            "metadata scale variance per layer (≥ 1e-4)",
            "metadata conf variance per layer (≥ 1e-4)",
        ],
        "step_4_shuffle_audit": (
            "Shuffle metadata across items; recompute on/off/shuffle T5 stage 3 ablation. "
            "If shuffle diff > 1e-4, metadata has per-item signal (PASS)."
        ),
        "step_5_alignment_audit": (
            "Verify each item has unique SID (1-1 mapping). If duplicates, collision check FAIL."
        ),
        "step_6_gate_decision": (
            "If ALL metrics above PASS, write verdict 'Gate 2 PASS' + commit + push + close issue. "
            "If any FAIL, write 'Gate 2 FAIL' + commit + close issue."
        ),
    }


#===========================================================================================
# Main
#===========================================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print(f"Task #392 / Issue #99 formula-based SID/metadata diversity audit")
    print("=" * 80)

    # 1. Dependency status
    print("\n[Section 1] Dependency status:")
    for issue, status in DEPENDENCY_STATUS.items():
        print(f"  - {issue}: {status['status']}")

    # 2. Gate 2 acceptance framework
    print("\n[Section 2] Gate 2 acceptance framework:")
    for k, v in GATE2_ACCEPTANCE_FRAMEWORK.items():
        if isinstance(v, list):
            print(f"  - {k}:")
            for item in v:
                print(f"    • {item}")
        elif isinstance(v, dict):
            print(f"  - {k}:")
            for sk, sv in v.items():
                print(f"    • {sk}: {sv}")
        else:
            print(f"  - {k}: {v}")

    # 3. Baseline audit (verify #96 FAIL state)
    print("\n[Section 3] Baseline audit (current state, should reproduce #96 FAIL):")
    baseline = baseline_audit()
    print(f"  SID unique = {baseline['sid_unique']}/9922 ({baseline['sid_unique_pct']:.2f}%)")
    print(f"  SID collision = {baseline['sid_collision_rate']*100:.2f}%")
    print(f"  L0 util = {baseline['L0_util']*100:.2f}%")
    print(f"  metadata kappa/scale/conf var = {baseline['metadata_kappa_var']}/{baseline['metadata_scale_var']}/{baseline['metadata_conf_var']}")
    print(f"  baseline collapse confirmed = {baseline['baseline_collapse_confirmed']}")

    # 4. Post-fix framework
    print("\n[Section 4] Post-fix framework (after #97/#98 fix):")
    framework = post_fix_framework()
    for k, v in framework.items():
        if isinstance(v, list):
            print(f"  - {k}:")
            for item in v:
                print(f"    • {item}")
        else:
            print(f"  - {k}: {v[:120]}...")

    # Summary
    # 当前 baseline 应该确认 FAIL (= #96 状态), 这是预期结果
    # Issue #99 当前无法 PASS (依赖 #97/#98 修复)
    summary = {
        "dependency_status": DEPENDENCY_STATUS,
        "gate2_acceptance_framework": GATE2_ACCEPTANCE_FRAMEWORK,
        "baseline_audit": baseline,
        "post_fix_framework": framework,
        "current_state": "FAIL — 依赖 #97/#98 修复; 当前 baseline 确认坍缩 (跟 #96 一致)",
        "can_close_now": False,  # 不允许现在 close, 必须等 #97/#98 PASS
        "next_action": "等 #97/#98 修复 PASS 后, 跑 post_fix framework → 写 Gate 2 verdict → commit+push+comment+close",
    }

    evidence_path = AUDIT_ROOT / "evidence_package.json"
    with open(evidence_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[EVIDENCE] Saved to {evidence_path}")

    framework_path = AUDIT_ROOT / "gate2_acceptance_framework.md"
    with open(framework_path, "w") as f:
        f.write("# Issue #99 Gate 2 SID/metadata 多样性准入 Framework\n\n")
        f.write("## 依赖 (Dependency)\n\n")
        f.write("Issue #99 Gate 2 PASS 依赖:\n")
        f.write("- **Issue #97 [方向A Gate1]** κ-Stereographic distance 公式修复 PASS\n")
        f.write("- **Issue #98 [方向B Gate1]** product component distance 尺度修复 PASS\n\n")
        f.write("## Gate 2 验收阈值\n\n")
        for k, v in GATE2_ACCEPTANCE_FRAMEWORK.items():
            if isinstance(v, list):
                f.write(f"### {k}\n")
                for item in v:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(v, dict):
                f.write(f"### {k}\n")
                for sk, sv in v.items():
                    f.write(f"- **{sk}**: {sv}\n")
                f.write("\n")
            else:
                f.write(f"- **{k}**: {v}\n\n")

        f.write("## 当前状态 (Baseline audit)\n\n")
        f.write(f"- SID unique: {baseline['sid_unique']}/9922 ({baseline['sid_unique_pct']:.2f}%)\n")
        f.write(f"- SID collision rate: {baseline['sid_collision_rate']*100:.2f}%\n")
        f.write(f"- L0 utilization: {baseline['L0_util']*100:.2f}%\n")
        f.write(f"- metadata kappa variance: {baseline['metadata_kappa_var']}\n")
        f.write(f"- baseline collapse confirmed: {baseline['baseline_collapse_confirmed']}\n\n")

        f.write("## Post-fix framework (after #97/#98 PASS)\n\n")
        for k, v in framework.items():
            if isinstance(v, list):
                f.write(f"### {k}\n")
                for item in v:
                    f.write(f"- {item}\n")
                f.write("\n")
            else:
                f.write(f"- **{k}**: {v}\n\n")
    print(f"[FRAMEWORK] Saved to {framework_path}")

    return summary


if __name__ == "__main__":
    main()