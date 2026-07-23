#!/usr/bin/env python3
"""task475_crossseed_posindep_mfsr.py — Task #475: 跨种子 position independence MFSR

对 B/C/D seed=43 跑 brand MFSR, 与 seed=42 baseline 对照 (task469 报告),
判定 H 位置独立性是否在 seed=43 下仍然成立 (within 5%)。

依据: task472 发现 seed=43 会让 P0.3 L3 坍缩, 必须验证 B/C/D 在 seed=43
是不是也 L3 一致坍缩 (position-independence holds), 还是说 "B/C/D 比 P0.3
更稳" 或 "更不稳" —— 即 H 位置在跨种子条件下提供差异化贡献。
"""
import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch

# Re-use task469 MFSR function
SCRIPT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts"
sys.path.insert(0, SCRIPT_DIR)
from task469_bulk_mfsr import compute_brand_mfsr, load_mf  # type: ignore

LOG_INF = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs"
TASK472_BASE = f"{LOG_INF}/task472"

OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp475")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CODEWORD = 30
SEED = 42

# Variants with seed=43
VARIANTS_S43 = [
    ("P0.3_v3", "task472_p03_v3_s43/pickle/merged_predictions_tensor.pt"),
    ("B_v3",    "task472_B_v3_s43/pickle/merged_predictions_tensor.pt"),
    ("C_v3",    "task472_C_v3_s43/pickle/merged_predictions_tensor.pt"),
    ("D_v3",    "task472_D_v3_s43/pickle/merged_predictions_tensor.pt"),
]

# Seed=42 reference (from task469)
P0_2_BASELINE = {1: 0.2520039558711678, 2: 0.2540622064261697, 3: 0.26671193481660355}
SEED42_REFERENCE_DELTA = {
    # from task469_bulk_brand_mfsr.json: variant_results[variant][layer]['mean'] - P0_2
    "P0.3_v3": {1: 0.0624, 2: 0.0711, 3: 0.0666},
    "B_v3":    {1: 0.0671, 2: 0.0718, 3: 0.0630},
    "C_v3":    {1: 0.0518, 2: 0.0584, 3: 0.0461},
    "D_v3":    {1: 0.0742, 2: 0.0581, 3: 0.0571},
}


def main():
    print(f"[task475] loading multi_factor...", flush=True)
    _, F_brand, _, _, item_ids = load_mf()
    print(f"[task475] item_ids: {len(item_ids)} (max {item_ids.max().item()})", flush=True)

    all_results = {}
    all_diff = {}
    for vname, vpath in VARIANTS_S43:
        full = os.path.join(LOG_INF, vpath)
        if not os.path.exists(full):
            print(f"[task475] SKIP {vname}: {full} not found", flush=True)
            continue
        print(f"\n[task475] === {vname} (seed=43) ===", flush=True)
        sid = torch.load(full, map_location="cpu", weights_only=False).long()
        print(f"[task475]   sid shape: {tuple(sid.shape)}", flush=True)
        res = compute_brand_mfsr(sid, F_brand, item_ids)
        for layer in (1, 2, 3):
            r = res[layer]
            print(f"   L{layer} MFSR = {r['mean']:.4f} ± {r['std']:.4f} (n_cw={r['n_codewords']})", flush=True)
        all_results[vname] = res
        all_diff[vname] = {layer: res[layer]["mean"] - P0_2_BASELINE[layer] for layer in (1,2,3)}

    out = {
        "max_per_codeword": MAX_PER_CODEWORD,
        "seed": 43,
        "p0_2_baseline_brand_mfsr": P0_2_BASELINE,
        "variant_results_s43": all_results,
        "diff_vs_p0_2_s43": all_diff,
        "seed42_reference_delta": SEED42_REFERENCE_DELTA,
    }
    out_path = OUT_DIR / "task475_crossseed_brand_mfsr.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[task475] saved {out_path}", flush=True)

    # Print comparison: seed=42 vs seed=43, delta-diff
    print(f"\n[task475] === Δ Brand MFSR (vs P0.2): seed=42 vs seed=43 ===")
    print(f"{'Variant':<10} {'L1 s42':>8} {'L1 s43':>8} {'ΔL1':>8} {'L2 s42':>8} {'L2 s43':>8} {'ΔL2':>8} {'L3 s42':>8} {'L3 s43':>8} {'ΔL3':>8}")
    for vname in [v[0] for v in VARIANTS_S43]:
        if vname not in all_diff:
            continue
        s42 = SEED42_REFERENCE_DELTA[vname]
        s43 = all_diff[vname]
        dl1 = s43[1] - s42[1]
        dl2 = s43[2] - s42[2]
        dl3 = s43[3] - s42[3]
        print(f"{vname:<10} {s42[1]:+.4f} {s43[1]:+.4f} {dl1:+.4f} {s42[2]:+.4f} {s43[2]:+.4f} {dl2:+.4f} {s42[3]:+.4f} {s43[3]:+.4f} {dl3:+.4f}")


if __name__ == "__main__":
    main()
