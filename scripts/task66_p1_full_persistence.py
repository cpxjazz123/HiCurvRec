#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task66_p1_full_persistence.py — Task #66 (G3) P1: 全数据拓扑保真度诊断

对全 embedding 矩阵 {flan-t5 2048d, sentence-t5 768d, hybrid 2816d} × 重构方法 {vanilla L0, sinkhorn L0, sinkhorn cascade}
计算 H_0 bottleneck distance d_B(single-linkage heights) — 用 scipy single-linkage, 100 子样 × 5 bootstrap.
分析:
  (1) d_B 是否在不同 embedding 源/重构方法之间有可分辨差异
  (2) Sinkhorn-balanced 是否比 vanilla 更好地保留拓扑 (G3 与 G1 联动)
  (3) d_B 是否随 cascade 层数增加 (L0 → L0+L1 → L0+L1+L2)

GO 条件 (vs D0): signal/noise 仍 > 50×, 且 d_B 在 embedding/recon 组合之间有可分辨差异.

执行:
  python3 scripts/task66_p1_full_persistence.py \
      --src_embeddings task59=...pt task60=...pt task61=...pt \
      --recon_methods vanilla_l0 sinkhorn_l0 sinkhorn_cascade \
      --output_dir logs/task66_p1
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist


# === 已有产物路径 ===
STAGE1_PATHS = {
    "flan-t5_2048d": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt",
    "sentence-t5_768d": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task60_s1/runs/2026-07-20/05-53-06/pickle/merged_predictions_tensor.pt",
    "hybrid_2816d": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task61_s1/merged_predictions_2816d.pt",
}
RECON_PATHS = {
    # sinkhorn L0 + L1 + L2 centers from task63 P1 (flan-t5 only)
    "sinkhorn_l0": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_centers_l0_sinkhorn_balanced.npy",
    "sinkhorn_l1": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_centers_l1_sinkhorn_balanced.npy",
    "sinkhorn_l2": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_centers_l2_sinkhorn_balanced.npy",
    "sinkhorn_idx_l0": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_idx_l0_sinkhorn_balanced.npy",
    "sinkhorn_idx_l1": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_idx_l1_sinkhorn_balanced.npy",
    "sinkhorn_idx_l2": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_idx_l2_sinkhorn_balanced.npy",
    "vanilla_l0": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_centers_l0_vanilla.npy",
    "vanilla_idx_l0": "/home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle/cluster_idx_l0_vanilla.npy",
}


def single_linkage_heights(X: np.ndarray, n_samples: int = 200, seed: int = 42) -> np.ndarray:
    """scipy single-linkage 返回 N-1 个高度, 我们用全部 N-1 个 (不限制 n_samples)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > n_samples:
        idx = rng.choice(n, size=n_samples, replace=False)
        X = X[idx]
    if X.shape[0] < 100:
        # 处理过小子样
        return None
    try:
        Z = single(pdist(X))
        return Z[:, 2]  # N-1 heights
    except Exception:
        return None


def bottleneck_distance_h0(heights_X: np.ndarray, heights_Y: np.ndarray) -> float:
    """H_0 bottleneck: scikit-tda 直接算, 这里用 sorted heights + sup norm (D0 同方法)."""
    if heights_X is None or heights_Y is None:
        return float("nan")
    n_min = min(len(heights_X), len(heights_Y))
    return float(np.max(np.abs(np.sort(heights_X)[:n_min] - np.sort(heights_Y)[:n_min])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_bootstrap", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============ 加载所有原始 embedding ============
    embeddings = {}
    for name, path in STAGE1_PATHS.items():
        emb = torch.load(path, weights_only=False, map_location="cpu").float().numpy()
        embeddings[name] = emb
        print(f"[G3 P1] {name}: shape={emb.shape}, mean_norm={float(np.linalg.norm(emb, axis=1).mean()):.2f}")

    # ============ 加载 sinkhorn centers 并构造重构 ============
    print("\n[G3 P1] building reconstructions for flan-t5 (sinkhorn cascade)")
    flan_t5 = embeddings["flan-t5_2048d"]
    s_idx_l0 = np.load(RECON_PATHS["sinkhorn_idx_l0"])
    s_idx_l1 = np.load(RECON_PATHS["sinkhorn_idx_l1"])
    s_idx_l2 = np.load(RECON_PATHS["sinkhorn_idx_l2"])
    s_c_l0 = np.load(RECON_PATHS["sinkhorn_l0"])
    s_c_l1 = np.load(RECON_PATHS["sinkhorn_l1"])
    s_c_l2 = np.load(RECON_PATHS["sinkhorn_l2"])
    # 重构 = L0 + L1 + L2 centers 累加 (无 dedup, 因为我们看的是 raw 失真)
    recon_sinkhorn_l0 = s_c_l0[s_idx_l0]
    recon_sinkhorn_cascade = s_c_l0[s_idx_l0] + s_c_l1[s_idx_l1] + s_c_l2[s_idx_l2]
    print(f"  sinkhorn L0 recon: shape={recon_sinkhorn_l0.shape}, mean_norm={float(np.linalg.norm(recon_sinkhorn_l0, axis=1).mean()):.2f}")
    print(f"  sinkhorn cascade recon: shape={recon_sinkhorn_cascade.shape}, mean_norm={float(np.linalg.norm(recon_sinkhorn_cascade, axis=1).mean()):.2f}")

    v_idx_l0 = np.load(RECON_PATHS["vanilla_idx_l0"])
    v_c_l0 = np.load(RECON_PATHS["vanilla_l0"])
    recon_vanilla_l0 = v_c_l0[v_idx_l0]
    print(f"  vanilla L0 recon: shape={recon_vanilla_l0.shape}, mean_norm={float(np.linalg.norm(recon_vanilla_l0, axis=1).mean()):.2f}")

    # ============ 主分析: 对每个组合算 d_B(H_0) ============
    print("\n[G3 P1] computing bottleneck distances...")
    conditions = {
        "flan-t5_raw_vs_sinkhorn_l0": (flan_t5, recon_sinkhorn_l0),
        "flan-t5_raw_vs_sinkhorn_cascade": (flan_t5, recon_sinkhorn_cascade),
        "flan-t5_raw_vs_vanilla_l0": (flan_t5, recon_vanilla_l0),
        "flan-t5_sinkhorn_l0_vs_vanilla_l0": (recon_sinkhorn_l0, recon_vanilla_l0),
    }

    results = {}
    for cond_name, (X, Y) in conditions.items():
        print(f"\n[{cond_name}] bootstrap × {args.n_bootstrap}")
        t0 = time.time()
        dBs = []
        noise_band = []
        for b in range(args.n_bootstrap):
            h_X = single_linkage_heights(X, n_samples=args.n_samples, seed=42 + b * 100)
            h_Y = single_linkage_heights(Y, n_samples=args.n_samples, seed=42 + b * 100 + 50)
            dB = bottleneck_distance_h0(h_X, h_Y)
            dBs.append(dB)
            # 自举噪声: 同一组子样内两个随机半 + 比
            h_X1 = single_linkage_heights(X, n_samples=args.n_samples, seed=42 + b * 100 + 200)
            h_X2 = single_linkage_heights(X, n_samples=args.n_samples, seed=42 + b * 100 + 250)
            noise_band.append(bottleneck_distance_h0(h_X1, h_X2))
        dBs_arr = np.array(dBs)
        noise_arr = np.array(noise_band)
        snr = float(dBs_arr.mean() / (noise_arr.mean() + 1e-12))
        results[cond_name] = {
            "d_B_mean": float(dBs_arr.mean()),
            "d_B_std": float(dBs_arr.std()),
            "noise_mean": float(noise_arr.mean()),
            "noise_std": float(noise_arr.std()),
            "signal_noise_ratio": snr,
            "bootstrap_dB": np.array(dBs, dtype=float).tolist(),
            "bootstrap_noise": noise_band,
            "time_sec": time.time() - t0,
        }
        print(f"  d_B = {dBs_arr.mean():.4f} ± {dBs_arr.std():.4f}")
        print(f"  noise_band = {noise_arr.mean():.4f} ± {noise_arr.std():.4f}")
        print(f"  signal/noise = {snr:.1f}×  time = {time.time()-t0:.1f}s")

    # ============ G1 vs G3 联动 (核心) ============
    # 比较 vanilla L0 vs sinkhorn L0 的拓扑保留程度
    db_vanilla = results["flan-t5_raw_vs_vanilla_l0"]["d_B_mean"]
    db_sinkhorn_l0 = results["flan-t5_raw_vs_sinkhorn_l0"]["d_B_mean"]
    db_sinkhorn_cascade = results["flan-t5_raw_vs_sinkhorn_cascade"]["d_B_mean"]
    db_l0_vs_l_l = results["flan-t5_sinkhorn_l0_vs_vanilla_l0"]["d_B_mean"]

    print(f"\n=== G1 ↔ G3 联动: 量化器选择 vs 拓扑保留 ===")
    print(f"  vanilla L0:    d_B = {db_vanilla:.4f}")
    print(f"  sinkhorn L0:   d_B = {db_sinkhorn_l0:.4f}  Δ_vs_vanilla = {db_sinkhorn_l0 - db_vanilla:+.4f}")
    print(f"  sinkhorn cas:  d_B = {db_sinkhorn_cascade:.4f}")
    print(f"  sinkhorn_L0 vs vanilla_L0 (vs each other): d_B = {db_l0_vs_l_l:.4f}")

    # ============ 跨 embedding 诊断 (raw 维度统计) ============
    print(f"\n=== 跨 embedding raw 几何统计 (raw vs raw between sources) ===")
    cross_emb_dB = {}
    for name, emb in embeddings.items():
        if name == "flan-t5_2048d":
            continue
        X = embeddings["flan-t5_2048d"]
        # 维度不同 → 距离空间不同, d_B 仅在等维度可比
        # 退而求其次: 比较 top-50 single-linkage heights 的 L2 距离
        h_flant5 = single_linkage_heights(X, n_samples=args.n_samples, seed=42)
        h_other = single_linkage_heights(emb, n_samples=args.n_samples, seed=42)
        n_min = min(len(h_flant5) if h_flant5 is not None else 0, len(h_other) if h_other is not None else 0)
        n_min = min(n_min, 50)
        if h_flant5 is not None and h_other is not None:
            l2 = float(np.linalg.norm(np.sort(h_flant5)[:n_min] - np.sort(h_other)[:n_min]))
            cross_emb_dB[name] = {"top50_l2_heights": l2, "n_heights": n_min}
            print(f"  flan-t5 vs {name}: top-50 heights L2 = {l2:.4f}")

    # ============ 汇总 ============
    summary = {
        "input_embeddings": list(STAGE1_PATHS.keys()),
        "n_samples": args.n_samples,
        "n_bootstrap": args.n_bootstrap,
        "bottleneck_results": results,
        "cross_embedding_heights_l2": cross_emb_dB,
        "verdict": {
            "G3_main": "GO" if results["flan-t5_raw_vs_sinkhorn_l0"]["signal_noise_ratio"] > 50 else (
                "PARTIAL" if results["flan-t5_raw_vs_sinkhorn_l0"]["signal_noise_ratio"] > 10 else "NO-GO"
            ),
            "G1_G3_interaction": {
                "sinkhorn_L0_better_than_vanilla_topology": (db_sinkhorn_l0 < db_vanilla),
                "sinkhorn_L0_dB": db_sinkhorn_l0,
                "vanilla_L0_dB": db_vanilla,
            }
        }
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[G3 P1] saved → {out_dir / 'summary.json'}")
    print(f"\n[G3 P1 main verdict]: {summary['verdict']['G3_main']}")


if __name__ == "__main__":
    main()
