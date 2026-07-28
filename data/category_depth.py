#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户 2026-07-27 Phase 1 (J-plan): 从 Instruments.item.json 抽类别路径,
按 item_emb.parquet 顺序生成 category_depth.npy (Float32).

输出:
  /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/category_depth.npy
    shape (N,), dtype float32
    depth_target[i] = rho_min + (rho_max - rho_min) * (depth[i] - 2) / (max_depth - 2)
    (depth 起点 = 2, 因为 L1 = 1 是 Amazon Music 商业分类根, 全部 item 共享, 没区分度)
  /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/category_depth_meta.json
    统计信息 + 默认 rho_min/rho_max

用法: 单独跑一次, 写完后训练直接 load.
"""
import json, os
import numpy as np
import pandas as pd

DATA_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"

RHO_MIN = 1.5
RHO_MAX = 2.9  # √c·ρ ∈ [1.5, 2.9], 跨越激活阈值 2.

def main():
    # 1. 加载 item_emb.parquet (给定 ItemID 顺序)
    df = pd.read_parquet(os.path.join(DATA_DIR, "item_emb.parquet"))
    item_ids = df["ItemID"].tolist()
    N = len(item_ids)
    print(f"items: {N}")
    assert N == 9922, f"expected 9922 items, got {N}"

    # 2. 加载 item.json, 用 ItemID 索引
    with open(os.path.join(DATA_DIR, "Instruments.item.json"), "r") as f:
        items = json.load(f)

    # 3. 计算每个 item 的路径深度
    depths = np.zeros(N, dtype=np.int32)
    n_missing = 0
    for i, iid in enumerate(item_ids):
        entry = items.get(str(iid))
        if entry is None:
            n_missing += 1
            continue
        cats = entry.get("categories", "") or ""
        if not cats:
            continue
        parts = [p.strip() for p in cats.split(",") if p.strip()]
        depths[i] = len(parts)
    print(f"n_missing/no-cats: {n_missing}")
    print(f"depth histogram: {dict(zip(*np.unique(depths, return_counts=True)))}")

    # 4. 有效深度 = depth ≥ 2 (因为 L1 = 1 是根, 没区分度)
    valid_mask = depths >= 2
    valid_depths = depths[valid_mask]
    min_depth = int(valid_depths.min())
    max_depth = int(valid_depths.max())
    print(f"valid range: depth ∈ [{min_depth}, {max_depth}] (n_valid={valid_mask.sum()})")

    # 5. 算 rho_target (depth=2 → rho_min, depth=max_depth → rho_max, 线性)
    rho_target = np.zeros(N, dtype=np.float32)
    rho_target[valid_mask] = RHO_MIN + (RHO_MAX - RHO_MIN) * \
                             (depths[valid_mask] - min_depth) / (max_depth - min_depth)
    # depth < 2 (缺失) → 默认 rho = (rho_min + rho_max) / 2.
    default_rho = 0.5 * (RHO_MIN + RHO_MAX)
    rho_target[~valid_mask] = default_rho
    print(f"rho_target stats: mean={rho_target.mean():.4f}, "
          f"std={rho_target.std():.4f}, "
          f"min={rho_target.min():.4f}, max={rho_target.max():.4f}")

    # 6. 落盘
    out_npy = os.path.join(DATA_DIR, "category_depth.npy")
    np.save(out_npy, rho_target)
    print(f"saved: {out_npy} (shape {rho_target.shape})")

    meta = {
        "N": int(N),
        "n_valid": int(valid_mask.sum()),
        "n_invalid": int((~valid_mask).sum()),
        "depth_min": min_depth, "depth_max": max_depth,
        "rho_min": RHO_MIN, "rho_max": RHO_MAX,
        "default_rho_for_invalid": default_rho,
        "depth_histogram": {int(d): int(c) for d, c in
                            zip(*np.unique(depths, return_counts=True))},
    }
    out_json = os.path.join(DATA_DIR, "category_depth_meta.json")
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved: {out_json}")

if __name__ == "__main__":
    main()
