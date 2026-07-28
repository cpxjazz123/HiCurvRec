#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 (J-plan, 用户 2026-07-27): 类别路径深度审计.

决策闸门:
  A. 路径深度分布 (std ≥ 0.5 层) ← 最关键
  B. 各层类别数 (一级 ≥ 5, 三级 ≥ 50)
  C. 覆盖率 ≥ 80% (有完整类别路径的 item 占比)
  D. baseline 半径 vs 路径深度 Spearman ≈ 0 (有改进空间)

不通过 → 停止, 换回 F/H 方案.
"""
import sys, os, json
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict

DATA_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"

def load_items():
    with open(os.path.join(DATA_DIR, "Instruments.item.json"), "r") as f:
        return json.load(f)

def load_item_emb():
    df = pd.read_parquet(os.path.join(DATA_DIR, "item_emb.parquet"))
    emb = np.stack(df['embedding'].values, axis=0)
    return torch.from_numpy(emb).float()

def compute_depth(items):
    depths, has_cat = [], []
    for iid, item in items.items():
        cats = item.get("categories", "") or ""
        if not cats or not cats.strip():
            depths.append(0); has_cat.append(False); continue
        parts = [p.strip() for p in cats.split(",") if p.strip()]
        depths.append(len(parts)); has_cat.append(True)
    return np.array(depths), np.array(has_cat)

def per_layer_category_count(items):
    layer_cats = defaultdict(Counter)
    for iid, item in items.items():
        cats = item.get("categories", "") or ""
        if not cats: continue
        parts = [p.strip() for p in cats.split(",") if p.strip()]
        for li, c in enumerate(parts):
            layer_cats[li][c] += 1
    summary = {}
    for li in sorted(layer_cats.keys()):
        summary[li] = {"n_unique": len(layer_cats[li]),
                       "top5": layer_cats[li].most_common(5)}
    return summary

def spearman_baseline(depths, has_cat, emb):
    from scipy.stats import spearmanr
    mask = has_cat & (depths > 0)
    sub = depths[mask]
    radius = emb[mask].norm(dim=-1).numpy()
    rho_val, p_val = spearmanr(sub, radius)
    return {"n": int(mask.sum()),
            "spearman_rho": float(rho_val), "p_value": float(p_val),
            "depth_mean": float(sub.mean()), "depth_std": float(sub.std()),
            "radius_mean": float(radius.mean()), "radius_std": float(radius.std())}

def main():
    print("="*70)
    print("Phase 0 (J-plan): 类别路径深度数据审计")
    print("="*70)
    items = load_items()
    emb = load_item_emb()
    print(f"items loaded: {len(items)}, emb shape: {tuple(emb.shape)}")
    depths, has_cat = compute_depth(items)
    valid = has_cat & (depths > 0)
    print(f"\n--- A. 深度分布 ---")
    cnt = Counter(depths.tolist())
    for d in sorted(cnt.keys()):
        print(f"  depth={d}: {cnt[d]} ({100*cnt[d]/len(items):.2f}%)")
    if valid.any():
        vd = depths[valid]
        print(f"  valid: n={valid.sum()} ({100*valid.sum()/len(items):.2f}%)")
        print(f"  mean={vd.mean():.4f}, std={vd.std():.4f}, min={vd.min()}, max={vd.max()}")
        print(f"  quartiles [25/50/75] = {np.percentile(vd, [25,50,75]).round(2).tolist()}")

    print(f"\n--- B. 各层类别数 ---")
    layer_summary = per_layer_category_count(items)
    for li in sorted(layer_summary.keys()):
        s = layer_summary[li]
        print(f"  L{li+1}: {s['n_unique']} distinct, top5={s['top5']}")

    coverage = 100.0 * valid.sum() / len(items)
    print(f"\n--- C. 覆盖率 ---")
    print(f"  {valid.sum()}/{len(items)} = {coverage:.2f}%")

    print(f"\n--- D. baseline (item_emb L2 norm vs depth) Spearman ---")
    sp = spearman_baseline(depths, has_cat, emb)
    for k, v in sp.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*70}\n闸门判定\n{'='*70}")
    L1_n = layer_summary.get(0, {}).get("n_unique", 0)
    L3_n = layer_summary.get(2, {}).get("n_unique", 0)
    gates = {
        "A. std ≥ 0.5": (sp["depth_std"] >= 0.5, f"std={sp['depth_std']:.4f}"),
        "C. coverage ≥ 80%": (coverage >= 80.0, f"cov={coverage:.2f}%"),
        "B1. L1 ≥ 5": (L1_n >= 5, f"L1={L1_n}"),
        "B3. L3 ≥ 50": (L3_n >= 50, f"L3={L3_n}"),
        "D. |Spearman| ≤ 0.3": (abs(sp["spearman_rho"]) <= 0.3, f"|ρ|={abs(sp['spearman_rho']):.4f}"),
    }
    for name, (passed, msg) in gates.items():
        print(f"  {'✅' if passed else '❌'} {name}: {msg}")
    all_pass = all(p for p, _ in gates.values())
    print(f"\n  整体: {'✅ 闸门通过, 进 Phase 1' if all_pass else '❌ 闸门不通过, 停 (换 F/H)'}")

    out = {
        "n_items": int(len(items)),
        "coverage_pct": float(coverage),
        "depth_stats_valid": {"mean": sp["depth_mean"], "std": sp["depth_std"],
                              "min": int(depths[valid].min()) if valid.any() else 0,
                              "max": int(depths[valid].max()) if valid.any() else 0,
                              "depth_histogram": {str(k): int(v) for k, v in cnt.items()}},
        "layer_summary": {str(li): {"n_unique": s["n_unique"], "top5": s["top5"]}
                          for li, s in layer_summary.items()},
        "spearman_baseline": sp,
        "gates": {k: {"passed": bool(p), "msg": m} for k, (p, m) in gates.items()},
        "all_pass": bool(all_pass),
    }
    out_path = "/home/wlia0047/ar57/wenyu/GeneRec/verdicts/_phase0j_data_audit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {out_path}")

if __name__ == "__main__":
    main()
