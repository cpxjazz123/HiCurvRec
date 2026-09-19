"""Sphere→Hyperbolic 显式映射 (Stage 0 Pre-RQ Manifold Adapter).

iter30 (R36m-S2 / Stage 0 Pre-RQ Manifold Adapter, 2026-09-19 用户设计):
- Stage 1 embedding 完全冻结, 不引入用户序列/额外监督
- 在 embedding 进入 RQ-VAE 之前做一个 manifold 显式映射:
    h_i = exp_0^c(e_i) = tanh(√c · ||e_i||) · e_i / (√c · ||e_i||)
- exp_0^c 把 T_0 H^{c,d} (origin 的切空间 = R^d Euclidean) 的点映射到 B^{c,d} (Poincaré ball)
- 性质: ||h_i|| = tanh(√c · ||e_i||) / √c < 1/√c (永远在 Poincaré ball 内)
- 单变量 vs iter11 baseline: RQ-VAE 主体一字不动, 只改输入 embedding 几何
  (tanh 软截断 + manifold 显式, **非线性** magnitude 变换, 与 R36m radial (单调 pow) **完全不同**)

设计动机:
- iter28 α=+0.25 oracle 0.0970 (-63.7%)
- iter29 α=-0.5 oracle 0.0983 (-63.2%)
- 两者退化值几乎完全相同 → R36m radial 框架证伪 (单调 radius scaling 必然失败)
- 跳出 radial 框架: tanh 软截断让 norm 大的 items 被**饱和压缩**到接近 1/√c, norm 小的 items 保留
- 这是 Poincaré 流形上原点指数映射的标准公式 (Ungar 2008 hyperbolic geometry)

参数 (硬编码, 0 CLI flag, Project Rules §1):
- C = 1.0 (Poincaré ball 曲率, 第一版)
- 单变量 sweep 计划: iter30 = c=1.0, iter31 = c=0.5 (低曲率更接近 Euclidean), iter32 = c=2.0 (高曲率更 hyperbolic)

硬编码路径 (Project Rules §1):
- 输入: /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy
- 输出: /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/dataset/Instruments/item_emb_hyperbolic_c1.0.npy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INPUT_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/dataset/Instruments"
OUTPUT_NPY = f"{OUTPUT_DIR}/item_emb_hyperbolic_c1.0.npy"
C = 1.0  # Poincaré ball 曲率, 第一版
N_ITEMS = 24587
EMB_DIM = 768


def exp_0_c(emb: np.ndarray, c: float) -> np.ndarray:
    """exp_0^c: R^d → B^{c,d}. 标准 Poincaré ball 原点指数映射.

    公式: exp_0^c(v) = tanh(√c · ||v||) · v / (√c · ||v||)
    性质: ||exp_0^c(v)|| = tanh(√c · ||v||) / √c < 1/√c
    边界: ||v|| → 0 时 exp_0^c(v) ≈ v (Euclidean limit); ||v|| → ∞ 时 ||exp|| → 1/√c (饱和)
    """
    sqrt_c = np.sqrt(c)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)  # (N, 1)
    norm_safe = np.where(norm < 1e-12, 1e-12, norm)
    tanh_term = np.tanh(sqrt_c * norm_safe)  # (N, 1)
    # exp = tanh(√c·||v||) · v / (√c·||v||)
    return (tanh_term / (sqrt_c * norm_safe)) * emb


def main() -> None:
    print("=== iter30 Sphere→Hyperbolic 显式映射 ===")
    print(f"  C (curvature) = {C}")
    print(f"  formula: exp_0^c(v) = tanh(√c · ||v||) · v / (√c · ||v||)")
    print(f"  INPUT  = {INPUT_NPY}")
    print(f"  OUTPUT = {OUTPUT_NPY}")

    emb = np.load(INPUT_NPY).astype(np.float32)
    if emb.shape != (N_ITEMS, EMB_DIM):
        raise ValueError(f"input shape {emb.shape} ≠ ({N_ITEMS}, {EMB_DIM})")
    print(f"  loaded: shape={emb.shape}, mean_norm={float(np.linalg.norm(emb, axis=1).mean()):.4f}")

    norm_orig_vec = np.linalg.norm(emb, axis=1, keepdims=True)
    print(f"  original norm stats: min={float(norm_orig_vec.min()):.4f} max={float(norm_orig_vec.max()):.4f} mean={float(norm_orig_vec.mean()):.4f}")

    # 应用 exp_0^c
    print(f"  [1/2] applying exp_0^{C} ...")
    emb_hyp = exp_0_c(emb, C)

    # 验证 Poincaré ball 内: ||h_i|| < 1/√C
    bound = 1.0 / np.sqrt(C)
    norm_hyp_vec = np.linalg.norm(emb_hyp, axis=1, keepdims=True)
    max_norm = float(norm_hyp_vec.max())
    print(f"  hyperbolic norm stats: min={float(norm_hyp_vec.min()):.4f} max={max_norm:.4f} mean={float(norm_hyp_vec.mean()):.4f}")
    print(f"  Poincaré bound (1/√C={bound:.4f}): max_norm < bound? {max_norm < bound}")
    if max_norm >= bound:
        raise ValueError(f"Poincaré ball 约束违反: max_norm {max_norm} ≥ 1/√C {bound}")

    # 验证 direction preservation (与 iter28/29 一致): radial+manifold mapping 保持 direction
    # 注意: exp_0^c(v) 沿 v 方向, 所以 direction 完美保留
    norm_orig_safe = np.where(norm_orig_vec < 1e-12, 1e-12, norm_orig_vec)
    norm_hyp_safe = np.where(norm_hyp_vec < 1e-12, 1e-12, norm_hyp_vec)
    cos_sim = (emb * emb_hyp).sum(axis=1) / (norm_orig_safe.squeeze() * norm_hyp_safe.squeeze() + 1e-12)
    print(
        f"  direction preservation cos(e, h): "
        f"min={float(cos_sim.min()):.4f} mean={float(cos_sim.mean()):.4f} max={float(cos_sim.max()):.4f}"
    )
    if float(cos_sim.min()) < 0.99:
        raise ValueError(f"direction preservation 严重破坏: min cos_sim={float(cos_sim.min()):.4f}")

    # tanh 软截断的 norm 分布对比: 原本 norm 大的 items 被饱和压缩
    # norm 分布压缩比: tanh(√C · ||e||) / √C / ||e||
    # 应当 tanh(√C · ||e||) / (√C · ||e||) ≤ 1 (单调下降)
    norm_ratio = norm_hyp_safe.squeeze() / norm_orig_safe.squeeze()
    print(
        f"  norm_ratio (||h||/||e||): "
        f"min={float(norm_ratio.min()):.4f} mean={float(norm_ratio.mean()):.4f} max={float(norm_ratio.max()):.4f}"
    )

    # 写盘
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, emb_hyp.astype(np.float32))
    size_mb = Path(OUTPUT_NPY).stat().st_size / 1024 / 1024
    print(f"  written: {OUTPUT_NPY} ({size_mb:.1f} MB)")

    # 写 transform metadata
    meta = {
        "transform": "exp_0_c (Poincaré ball origin exponential map)",
        "formula": "tanh(√c · ||v||) · v / (√c · ||v||)",
        "c": C,
        "poincare_bound": bound,
        "max_norm": max_norm,
        "original_mean_norm": float(norm_orig_vec.mean()),
        "hyperbolic_mean_norm": float(norm_hyp_vec.mean()),
        "norm_ratio_min": float(norm_ratio.min()),
        "norm_ratio_mean": float(norm_ratio.mean()),
        "norm_ratio_max": float(norm_ratio.max()),
        "direction_cos_min": float(cos_sim.min()),
        "direction_cos_mean": float(cos_sim.mean()),
        "direction_cos_max": float(cos_sim.max()),
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