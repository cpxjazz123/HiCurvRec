#!/usr/bin/env python3
"""task469_bulk_mfsr.py — Phase 2 扩展: 批量对 7 variant 跑 brand MFSR

任务: 验证 task56 v3 + task57 7 variant SID tensor 的 brand MFSR，
扩展 task458 (P0.2 vs P0.3 二对照) 到 7 variant 全对照。

Variant 与 H 配置:
  - P0.3_v3:  h_dims=(32,0,0) text  baseline
  - A_v3:     h_dims=(0,0,0)   反事实 (无 H)
  - B_v3:     h_dims=(32,0,0)  brand
  - C_v3:     h_dims=(96,0,0)  taxonomy
  - D_v3:     h_dims=(32,0,0)  behavior
  - HHEE:     h_dims=(32,32,0) brand (L1+L2)
  - HHHH:     h_dims=(32,32,32) brand (全层)

对比 P0.2 baseline brand MFSR: L1=0.252, L2=0.254, L3=0.267
"""
import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch

MULTI_FACTOR_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/multi_factor.pt"

LOG_INF = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs"
TASK437_BASE = f"{LOG_INF}/task56"
TASK438_BASE = f"{LOG_INF}/task57"

OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp469")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CODEWORD = 30
SEED = 42

VARIANTS = [
    ("P0.3_v3", f"{TASK437_BASE}_p03_v3_s22/pickle/merged_predictions_tensor.pt"),
    ("A_v3",    f"{TASK437_BASE}_counterfactual_A_v3_s22/pickle/merged_predictions_tensor.pt"),
    ("B_v3",    f"{TASK437_BASE}_counterfactual_B_v3_s22/pickle/merged_predictions_tensor.pt"),
    ("C_v3",    f"{TASK437_BASE}_counterfactual_C_v3_s22/pickle/merged_predictions_tensor.pt"),
    ("D_v3",    f"{TASK437_BASE}_counterfactual_D_v3_s22/pickle/merged_predictions_tensor.pt"),
    ("HHEE",    f"{TASK438_BASE}_HHEE_s22/pickle/merged_predictions_tensor.pt"),
    ("HHHH",    f"{TASK438_BASE}_HHHH_s22/pickle/merged_predictions_tensor.pt"),
]


def load_mf():
    mf = torch.load(MULTI_FACTOR_PATH, map_location="cpu", weights_only=False)
    return mf["F_text"].float(), mf["F_brand"].float(), \
           mf["F_taxonomy"].float(), mf["F_behavior"].float(), mf["item_ids"].long()


def compute_brand_mfsr(sid: torch.Tensor, mf_brand: torch.Tensor, item_ids: torch.Tensor):
    """Compute brand MFSR for each layer.

    sid: (4, N) or (N, 3) int64
    mf_brand: (n_mf, 32)
    item_ids: (n_mf,) — maps mf idx → catalog item idx
    """
    # Auto-detect shape
    if sid.dim() == 2 and sid.shape[0] == 4:
        is_fixed = True
        n_items = sid.shape[1]
    else:
        is_fixed = False
        n_items = sid.shape[0]

    in_bounds = item_ids < n_items
    item_ids_valid = item_ids[in_bounds]
    mf_brand_valid = mf_brand[in_bounds]

    results = {}
    for layer in (1, 2, 3):
        if is_fixed:
            code_per_item = sid[layer - 1, item_ids_valid]
        else:
            code_per_item = sid[item_ids_valid, layer - 1]
        unique_codes, inverse = torch.unique(code_per_item, return_inverse=True)
        mfsr_values = []
        for i in range(len(unique_codes)):
            member_mask = (inverse == i)
            n_members = member_mask.sum().item()
            if n_members < 2:
                continue
            member_idx = member_mask.nonzero(as_tuple=True)[0]
            if n_members > MAX_PER_CODEWORD:
                rng = random.Random(SEED)
                sel = rng.sample(range(n_members), MAX_PER_CODEWORD)
                member_idx = member_idx[sel]
            emb_subset = mf_brand_valid[member_idx]
            normed = torch.nn.functional.normalize(emb_subset, dim=1)
            centroid = normed.mean(dim=0, keepdim=True)
            centroid = torch.nn.functional.normalize(centroid, dim=1)
            cosines = (normed @ centroid.T).squeeze(-1)
            mfsr_values.append(float(cosines.mean().item()))
        mfsr_arr = np.array(mfsr_values)
        results[layer] = {
            "mean": float(mfsr_arr.mean()) if len(mfsr_arr) > 0 else float("nan"),
            "std": float(mfsr_arr.std()) if len(mfsr_arr) > 0 else float("nan"),
            "n_codewords": len(mfsr_arr),
        }
    return results


def main():
    print(f"[MFSR-7] loading multi_factor...", flush=True)
    F_text, F_brand, F_tax, F_beh, item_ids = load_mf()
    print(f"[MFSR-7] item_ids: {len(item_ids)} (max {item_ids.max().item()})", flush=True)

    all_results = {}
    for vname, vpath in VARIANTS:
        if not os.path.exists(vpath):
            print(f"[MFSR-7] SKIP {vname}: {vpath} not found", flush=True)
            continue
        print(f"\n[MFSR-7] === {vname} ===", flush=True)
        print(f"[MFSR-7]   loading {vpath}", flush=True)
        sid = torch.load(vpath, map_location="cpu", weights_only=False).long()
        print(f"[MFSR-7]   sid shape: {tuple(sid.shape)}", flush=True)
        res = compute_brand_mfsr(sid, F_brand, item_ids)
        for layer in (1, 2, 3):
            r = res[layer]
            print(f"   L{layer} brand MFSR = {r['mean']:.4f} ± {r['std']:.4f} (n_cw={r['n_codewords']})", flush=True)
        all_results[vname] = res

    # P0.2 baseline from task458
    p0_2_baseline = {1: 0.2520039558711678, 2: 0.2540622064261697, 3: 0.26671193481660355}

    # Compute diff vs P0.2
    diff_table = {}
    for vname, res in all_results.items():
        diff_table[vname] = {}
        for layer in (1, 2, 3):
            diff_table[vname][layer] = res[layer]["mean"] - p0_2_baseline[layer]

    out = {
        "max_per_codeword": MAX_PER_CODEWORD,
        "seed": SEED,
        "p0_2_baseline_brand_mfsr": p0_2_baseline,
        "variant_results": all_results,
        "diff_vs_p0_2": diff_table,
    }
    out_path = OUT_DIR / "task469_bulk_brand_mfsr.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[MFSR-7] Saved to {out_path}", flush=True)

    # Print summary table
    print(f"\n[MFSR-7] === SUMMARY (brand MFSR diff vs P0.2 baseline) ===")
    print(f"{'Variant':<10} {'L1':>8} {'L2':>8} {'L3':>8}")
    for vname in [v[0] for v in VARIANTS]:
        if vname in diff_table:
            row = diff_table[vname]
            print(f"{vname:<10} {row[1]:+.4f} {row[2]:+.4f} {row[3]:+.4f}")


if __name__ == "__main__":
    main()