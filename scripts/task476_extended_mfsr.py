#!/usr/bin/env python3
"""task476_extended_mfsr.py — Task #476: HHHH 6-seed MFSR + P0.3 5-seed MFSR 扩展

将 task472 的 4-seed HHHH/P0.3/A 跨种子稳定性扩展到 6-seed HHHH / 5-seed P0.3。
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts"
sys.path.insert(0, SCRIPT_DIR)
from task469_bulk_mfsr import compute_brand_mfsr, load_mf

LOG_INF = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs"

OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp476")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seeds to compute MFSR for
TARGETS = [
    ("HHHH",  "task57_HHHH_s22", 42),  # original task57 seed=22 (=seed=42 in this codebase)
    ("HHHH",  "task472_HHHH_s43", 43),
    ("HHHH",  "task472_HHHH_s44", 44),
    ("HHHH",  "task472_HHHH_s45", 45),
    ("HHHH",  "task472_HHHH_s46", 46),
    ("HHHH",  "task472_HHHH_s47", 47),
    ("P0.3_v3", "task472_p03_v3_s42", 42),
    ("P0.3_v3", "task472_p03_v3_s43", 43),
    ("P0.3_v3", "task472_p03_v3_s44", 44),
    ("P0.3_v3", "task472_p03_v3_s45", 45),
    ("A_v3",  "task472_A_v3_s42", 42),
    ("A_v3",  "task472_A_v3_s43", 43),
    ("A_v3",  "task472_A_v3_s44", 44),
    ("A_v3",  "task472_A_v3_s45", 45),
]

P0_2_BASELINE = {1: 0.2520039558711678, 2: 0.2540622064261697, 3: 0.26671193481660355}


def main():
    print(f"[task476] loading multi_factor...", flush=True)
    _, F_brand, _, _, item_ids = load_mf()
    print(f"[task476] item_ids: {len(item_ids)}", flush=True)

    by_variant = {"HHHH": {}, "P0.3_v3": {}, "A_v3": {}}
    diff_by_variant = {"HHHH": {}, "P0.3_v3": {}, "A_v3": {}}

    for vname, sub, seed in TARGETS:
        sid_path = f"{LOG_INF}/{sub}/pickle/merged_predictions_tensor.pt"
        if not os.path.exists(sid_path):
            print(f"[task476] SKIP {vname} s{seed}: not found", flush=True)
            continue
        sid = torch.load(sid_path, map_location="cpu", weights_only=False).long()
        res = compute_brand_mfsr(sid, F_brand, item_ids)
        by_variant[vname][seed] = res
        diff_by_variant[vname][seed] = {l: res[l]["mean"] - P0_2_BASELINE[l] for l in (1, 2, 3)}
        print(f"[task476] {vname} s{seed}: L1={res[1]['mean']:.4f} L2={res[2]['mean']:.4f} L3={res[3]['mean']:.4f}", flush=True)

    # Compute μ ± σ per variant per layer (cross seeds)
    stats = {}
    for vname in by_variant:
        seeds_avail = sorted(by_variant[vname].keys())
        if len(seeds_avail) < 2:
            continue
        stats[vname] = {"n_seeds": len(seeds_avail), "seeds": seeds_avail}
        for l in (1, 2, 3):
            ms = np.array([by_variant[vname][s][l]["mean"] for s in seeds_avail])
            stats[vname][f"L{l}"] = {
                "mu": float(ms.mean()),
                "sigma": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
                "cv": float(ms.std(ddof=1) / ms.mean()) if (len(ms) > 1 and ms.mean() > 0) else 0.0,
                "values": {int(s): float(by_variant[vname][s][l]["mean"]) for s in seeds_avail},
                "n_codewords": {int(s): int(by_variant[vname][s][l]["n_codewords"]) for s in seeds_avail},
            }

    out = {
        "p0_2_baseline_brand_mfsr": P0_2_BASELINE,
        "raw_per_variant_per_seed": by_variant,
        "diff_vs_p0_2_per_variant_per_seed": diff_by_variant,
        "cross_seed_stats": stats,
    }
    out_path = OUT_DIR / "task476_extended_mfsr.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[task476] saved {out_path}", flush=True)

    # Print summary
    print(f"\n[task476] === Cross-seed (μ ± σ cv) Brand MFSR ===")
    print(f"{'Variant':<10} {'n':>3} {'L1 μ±σ (cv)':>26} {'L2 μ±σ (cv)':>26} {'L3 μ±σ (cv)':>26}")
    for vname in ("HHHH", "P0.3_v3", "A_v3"):
        if vname not in stats:
            continue
        s = stats[vname]
        n = s["n_seeds"]
        l1, l2, l3 = s["L1"], s["L2"], s["L3"]
        print(f"{vname:<10} {n:>3} "
              f"{l1['mu']:.4f}±{l1['sigma']:.4f} ({l1['cv']*100:.1f}%)  "
              f"{l2['mu']:.4f}±{l2['sigma']:.4f} ({l2['cv']*100:.1f}%)  "
              f"{l3['mu']:.4f}±{l3['sigma']:.4f} ({l3['cv']*100:.1f}%)")

    # Also L_unique stats
    print(f"\n[task476] === Cross-seed (μ ± σ cv) L_unique/256 ===")
    for vname in ("HHHH", "P0.3_v3", "A_v3"):
        if vname not in stats:
            continue
        s = stats[vname]
        seeds = s["seeds"]
        for l in (1, 2, 3):
            n_cw = np.array([s[f"L{l}"]["n_codewords"][seed] for seed in seeds])
            if len(n_cw) < 2:
                continue
            mu = n_cw.mean(); sd = n_cw.std(ddof=1); cv = sd/mu if mu > 0 else 0
            print(f"  {vname} L{l} (n={len(seeds)}): {mu:.0f}±{sd:.1f} (cv={cv*100:.1f}%) values={n_cw.tolist()}")


if __name__ == "__main__":
    main()
