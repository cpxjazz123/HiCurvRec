"""Density-aware radial preconditioning for Stage 1 → Stage 2 RQ-VAE input.

iter28 (R36m / Stage 0 Pre-RQ Geometry Adapter, 2026-09-19 用户设计):
- Stage 1 embedding 完全冻结, 不引入用户序列/额外监督
- 在 embedding 进入 RQ-VAE 之前做一个纯几何变换 (Curvature-aware radial preconditioning)
- 公式:
    e_i = r_i · u_i,  u_i = e_i / ||e_i||
    ρ_i = (1/k) Σ_{j ∈ N_k(i)} d(e_i, e_j)        # Euclidean 距离, kNN 用 cosine 选
    tilde_e_i = (ρ_i / ρ_bar)^α · e_i              # α ∈ {-0.5,-0.25,0,0.25,0.5}, α=0 = iter11 baseline
- 单变量 vs iter11: 只改输入 embedding 的几何形状 (radius scaled by local density),
  RQ-VAE 主体 (curriculum / midpoint / commit / Sinkhorn) 完全不动

论文支撑: 几何预调节 / local-density-aware metric scaling 在 metric learning (Kar & Jain 2011)
和 density-aware manifold learning (Goldberger et al. 2005) 中有先例; 在 RQ-VAE 之前
预处理输入嵌入是新方向, 没有直接先例, 但与 v337/v361 等曲率机制互补.

硬编码路径 (Project Rules §1):
- 输入: /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy
- 输出: /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/dataset/Instruments/item_emb_transformed_alpha{α}.npy

参数 (硬编码, 0 CLI flag):
- ALPHA = 0.25 (第一版)
- K_NEIGHBORS = 50
- 用 cosine 相似度选 kNN, 用 Euclidean 距离算 ρ_i (cosine 更稳定地表达"语义邻居"概念,
  Euclidean 距离对预处理后的 radial scaling 更直观)
- ρ_bar = mean(ρ_i) over all items (全局平均, 不是局部归一化)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INPUT_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments"
OUTPUT_NPY = f"{OUTPUT_DIR}/item_emb_transformed_alpha-0.5.npy"
ALPHA = -0.5
K_NEIGHBORS = 50
N_ITEMS = 24587
EMB_DIM = 768
CHUNK_SIZE = 512


def compute_topk_neighbors(emb: np.ndarray, k: int) -> np.ndarray:
    """chunked cosine similarity, return (N, k) top-k neighbor indices (excluding self).

    用 cosine sim 找语义近邻, 排除自身.
    """
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    emb_n = (emb / norm).astype(np.float32)
    N = emb_n.shape[0]
    top_k = np.zeros((N, k), dtype=np.int32)
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        sims = emb_n[start:end] @ emb_n.T  # (chunk, N)
        # 排除自身: 把对角线设为 -inf
        for li, gi in enumerate(range(start, end)):
            sims[li, gi] = -np.inf
        top_k[start:end] = np.argpartition(-sims, k, axis=1)[:, :k]
    return top_k


def compute_local_density(emb: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """对每个 item, ρ_i = mean Euclidean distance to its k nearest neighbors."""
    N = emb.shape[0]
    rho = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nbr_idx = neighbors[i]
        diffs = emb[nbr_idx] - emb[i]  # (k, D)
        dists = np.linalg.norm(diffs, axis=1)  # (k,)
        rho[i] = float(dists.mean())
    return rho


def main() -> None:
    print("=== iter28 density-aware radial preconditioning ===")
    print(f"  ALPHA = {ALPHA}")
    print(f"  K_NEIGHBORS = {K_NEIGHBORS}")
    print(f"  INPUT  = {INPUT_NPY}")
    print(f"  OUTPUT = {OUTPUT_NPY}")

    emb = np.load(INPUT_NPY).astype(np.float32)
    if emb.shape != (N_ITEMS, EMB_DIM):
        raise ValueError(f"input shape {emb.shape} ≠ ({N_ITEMS}, {EMB_DIM})")
    print(f"  loaded: shape={emb.shape}, mean_norm={float(np.linalg.norm(emb, axis=1).mean()):.4f}")

    # 1) 找 kNN (cosine sim)
    print("  [1/3] computing top-K cosine neighbors...")
    neighbors = compute_topk_neighbors(emb, K_NEIGHBORS)
    print(f"  neighbors shape: {neighbors.shape}")

    # 2) 计算 ρ_i = mean Euclidean dist to kNN
    print("  [2/3] computing local density ρ_i (Euclidean distance)...")
    rho = compute_local_density(emb, neighbors)
    print(
        f"  ρ stats: min={float(rho.min()):.4f} max={float(rho.max()):.4f} "
        f"mean={float(rho.mean()):.4f} std={float(rho.std()):.4f}"
    )
    rho_bar = float(rho.mean())
    if rho_bar <= 0:
        raise ValueError(f"ρ_bar={rho_bar} ≤ 0, density 计算异常")

    # 3) 应用 radial scaling: tilde_e_i = (ρ_i / ρ_bar)^α · e_i
    print(f"  [3/3] applying radial scaling α={ALPHA}...")
    scale = np.power(rho / rho_bar, ALPHA).astype(np.float32)  # (N,)
    print(
        f"  scale stats: min={float(scale.min()):.4f} max={float(scale.max()):.4f} "
        f"mean={float(scale.mean()):.4f}"
    )
    emb_transformed = emb * scale[:, None]  # (N, D)

    # 验证 invariant:
    # - α=0 时 tilde_e = e (严格等于 baseline, 验证 isolation 干净)
    # - α≠0 时 mean_norm 应该变化 (radial scaling 起作用)
    norm_orig = float(np.linalg.norm(emb, axis=1).mean())
    norm_trans = float(np.linalg.norm(emb_transformed, axis=1).mean())
    print(f"  mean_norm: original={norm_orig:.4f} transformed={norm_trans:.4f} ratio={norm_trans/norm_orig:.4f}")

    # 验证方向不变 (direction preservation): cos(e_i, tilde_e_i) 应该 = 1 (alpha=0) 或 close to 1 (alpha≠0)
    norm_orig_vec = np.linalg.norm(emb, axis=1, keepdims=True)
    norm_trans_vec = np.linalg.norm(emb_transformed, axis=1, keepdims=True)
    cos_sim = (emb * emb_transformed).sum(axis=1) / (norm_orig_vec.squeeze() * norm_trans_vec.squeeze() + 1e-12)
    print(
        f"  direction preservation: cos(e, tilde_e) "
        f"min={float(cos_sim.min()):.4f} mean={float(cos_sim.mean()):.4f} max={float(cos_sim.max()):.4f}"
    )
    # 余弦接近 1 表示方向保留 (radial scaling 只改 radius 不改 direction)
    if float(cos_sim.min()) < 0.5:
        raise ValueError(f"direction preservation 严重破坏: min cos_sim={float(cos_sim.min()):.4f}")

    # 写盘
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, emb_transformed.astype(np.float32))
    size_mb = Path(OUTPUT_NPY).stat().st_size / 1024 / 1024
    print(f"  written: {OUTPUT_NPY} ({size_mb:.1f} MB)")

    # 写 transform metadata (记录 α/k/ρ_bar/scale_stats 用于复现)
    meta = {
        "alpha": ALPHA,
        "k_neighbors": K_NEIGHBORS,
        "rho_bar": rho_bar,
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_mean": float(rho.mean()),
        "rho_std": float(rho.std()),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "scale_mean": float(scale.mean()),
        "norm_ratio": norm_trans / norm_orig,
        "direction_cos_min": float(cos_sim.min()),
        "direction_cos_mean": float(cos_sim.mean()),
        "input_npy": INPUT_NPY,
        "output_npy": OUTPUT_NPY,
    }
    meta_path = OUTPUT_NPY.replace(".npy", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta: {meta_path}")
    print(f"=== DONE ===")


if __name__ == "__main__":
    main()