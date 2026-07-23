#!/usr/bin/env python
"""Task #22 PM-RQ Phase 0 (D0) 几何诊断 — MCKG embedding 版

输入: MCKG Toys embedding (products/task19/mckg_M3_c0.5_dim32_toys/entity_embedding.pt)
      - subspace_item: shape (3, 11924, 32) — 3 κ 子空间
      - fused_item: shape (11924, 32) — 跨 κ 融合
      - kappas = [0.8446, -0.1741, -1.0586]

目标: 验证 3 个 κ 子空间是否确实学到不同几何 (sphere / euclid / hyperbolic)
     — 若 3 个子空间的 δ/anisotropy/norm 分布几乎相同 → PM-RQ 无意义
     — 若 3 个子空间的几何特征显著不同 → PM-RQ 有希望

D0 决策:
  ✓ 3 个 κ 子空间的几何特征显著不同 → proceed Phase 1
  ✗ 3 个子空间几何特征几乎相同 → flat, 任务提前终止
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

# Task #99: 支持 CLI 切换 MCKG embedding 路径 + 自定义报告路径
DEFAULT_MCKG_PATH = REPO_ROOT / "products/task19/mckg_M3_c0.5_dim32_toys/entity_embedding.pt"
DEFAULT_REPORT_PATH = REPO_ROOT / "verdicts/task22_phase0_d0_report.md"
DEFAULT_FIG_DIR = REPO_ROOT / "products/task22_pm_rq/d0_figs"

_ap = argparse.ArgumentParser()
_ap.add_argument('--emb-path', type=str, default=str(DEFAULT_MCKG_PATH),
                 help='MCKG entity_embedding.pt path')
_ap.add_argument('--report-path', type=str, default=str(DEFAULT_REPORT_PATH),
                 help='Output verdict report path')
_ap.add_argument('--fig-dir', type=str, default=str(DEFAULT_FIG_DIR),
                 help='Output figures dir')
_args = _ap.parse_args()

MCKG_PATH = Path(_args.emb_path)
REPORT_PATH = Path(_args.report_path)
FIG_DIR = Path(_args.fig_dir)
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "products/task22_pm_rq/d0_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SAMPLE = 2000       # smaller than T5 case (MCKG only 11924 items, dim=32)
N_QUAD = 5000         # 4-tuple samples for δ
N_PCA_LOCAL = 500     # anchors for local PCA
N_TSNE = 1000         # t-SNE subsample


def load_mckg():
    d = torch.load(MCKG_PATH, map_location="cpu", weights_only=False)
    print(f"[info] MCKG kappas: {d['kappas']}")
    print(f"[info] MCKG M={d['M']}, dim_per_subspace={d['dim_per_subspace']}")
    subspace_item = d["subspace_item"].float()  # (3, 11924, 32)
    fused_item = d["fused_item"].float()        # (11924, 32)
    kappas = d["kappas"]
    print(f"[info] subspace_item: {tuple(subspace_item.shape)}, fused_item: {tuple(fused_item.shape)}")
    return subspace_item, fused_item, kappas


def compute_delta_hyperbolicity(emb_np: np.ndarray, n_quad: int = N_QUAD, seed: int = SEED) -> dict:
    """Gromov δ-hyperbolicity via sampled 4-tuples."""
    rng = np.random.default_rng(seed)
    n = emb_np.shape[0]
    idx = rng.choice(n, size=min(N_SAMPLE, n), replace=False)
    sub = emb_np[idx]
    print(f"  [δ] pairwise distances on {sub.shape[0]} subsample...")
    D = squareform(pdist(sub, metric="euclidean")).astype(np.float32)
    diameter = D.max()
    print(f"  [δ] diameter={diameter:.4f}, mean dist={D[D > 0].mean():.4f}")

    quads = rng.integers(0, sub.shape[0], size=(n_quad, 4))
    deltas = np.empty(n_quad, dtype=np.float32)
    for i, (a, b, c, d) in enumerate(quads):
        dab = D[a, b]; dcd = D[c, d]
        dac = D[a, c]; dbd = D[b, d]
        dad = D[a, d]; dbc = D[b, c]
        sums = np.array([dab + dcd, dac + dbd, dad + dbc])
        deltas[i] = (np.sort(sums)[-1] - np.sort(sums)[-2]) / 2.0
    return {
        "delta_mean": float(deltas.mean()),
        "delta_max": float(deltas.max()),
        "delta_p50": float(np.percentile(deltas, 50)),
        "delta_p95": float(np.percentile(deltas, 95)),
        "delta_normalized_mean": float(deltas.mean() / diameter) if diameter > 0 else 0.0,
        "delta_normalized_max": float(deltas.max() / diameter) if diameter > 0 else 0.0,
        "diameter": float(diameter),
    }


def compute_local_anisotropy(emb_np: np.ndarray, k: int = 10, n_sample: int = N_PCA_LOCAL, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = emb_np.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    sub = emb_np[idx]

    # Global PCA
    pca = PCA(n_components=min(sub.shape[1], sub.shape[0] - 1))
    pca.fit(sub)
    ev = pca.explained_variance_ratio_

    # Local top-1 eig ratio (k-NN subspace anisotropy)
    local_anisotropy = []
    for i in range(min(300, sub.shape[0])):
        anchor = sub[i]
        d = np.linalg.norm(sub - anchor, axis=1)
        knn_idx = np.argsort(d)[:k + 1][1:]
        if len(knn_idx) < 3:
            continue
        local_pts = sub[knn_idx] - anchor
        try:
            _, s, _ = np.linalg.svd(local_pts, full_matrices=False)
            s = s / (s.sum() + 1e-9)
            local_anisotropy.append(float(s[0]))
        except Exception:
            continue

    return {
        "global_pca_ev": ev.tolist(),
        "global_pca_top1": float(ev[0]),
        "global_eig_ratio_2_1": float(ev[1] / ev[0]) if ev[0] > 0 else 0.0,
        "local_anisotropy_mean": float(np.mean(local_anisotropy)),
        "local_anisotropy_p50": float(np.percentile(local_anisotropy, 50)),
        "local_anisotropy_p95": float(np.percentile(local_anisotropy, 95)),
    }


def compute_norm_stats(emb_t: torch.Tensor) -> dict:
    norms = emb_t.norm(dim=1).numpy()
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_p05": float(np.percentile(norms, 5)),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p95": float(np.percentile(norms, 95)),
        "norm_cv": float(norms.std() / norms.mean()) if norms.mean() > 0 else 0.0,
    }


def compute_pairwise_subspace_independence(subspaces: torch.Tensor) -> dict:
    """测试 3 个 κ 子空间是否学到独立信息
    1. Cosine sim 矩阵 (mean vectors across items)
    2. Per-item 范数序列的 Pearson correlation
    """
    M, n, d = subspaces.shape  # (3, 11924, 32)
    means = np.zeros((M, d), dtype=np.float32)
    for i in range(M):
        means[i] = subspaces[i].mean(dim=0).numpy()
    norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
    cos_sim = (means / norms) @ (means / norms).T

    # Per-item norms
    norms_seq = np.zeros((M, n), dtype=np.float32)
    for i in range(M):
        norms_seq[i] = subspaces[i].norm(dim=1).numpy()
    pearson = np.corrcoef(norms_seq)

    # Per-item vector cosine (average pairwise item)
    # 更精细: 随机抽 500 items, 算 mean pairwise cos sim
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=500, replace=False)
    sample_mean_cos = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            a = subspaces[i][idx].numpy()
            b = subspaces[j][idx].numpy()
            a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            sample_mean_cos[i, j] = float((a_n * b_n).sum(axis=1).mean())
    return {
        "mean_vector_cos_sim": cos_sim.tolist(),
        "norm_seq_pearson": pearson.tolist(),
        "per_item_cos_sim_mean": sample_mean_cos.tolist(),
    }


def make_decision(per_subspace: dict, indep: dict) -> dict:
    """D0 决策: 3 个 κ 子空间的几何特征是否显著不同

    信号:
      - δ_normalized 在 3 个子空间间 spread > 0.05
      - anisotropy 在 3 个子空间间 spread > 0.1
      - norm_cv 在 3 个子空间间 spread > 0.05
      - 子空间 mean vector cos sim 矩阵的 off-diag 平均 < 0.7 (独立性)
    """
    deltas = [per_subspace[f"subspace_{i}"]["delta_normalized_mean"] for i in range(3)]
    anisos = [per_subspace[f"subspace_{i}"]["local_anisotropy_p95"] for i in range(3)]
    cvs = [per_subspace[f"subspace_{i}"]["norm_stats"]["norm_cv"] for i in range(3)]

    delta_spread = max(deltas) - min(deltas)
    aniso_spread = max(anisos) - min(anisos)
    cv_spread = max(cvs) - min(cvs)

    cos_mat = np.array(indep["mean_vector_cos_sim"])
    off_diag = []
    for i in range(3):
        for j in range(3):
            if i != j:
                off_diag.append(cos_mat[i, j])
    mean_off_diag_cos = float(np.mean(off_diag))

    reasons = []
    hyperbolic_signal = delta_spread > 0.05
    spherical_signal = aniso_spread > 0.1
    euclidean_signal = cv_spread > 0.05
    independence_signal = mean_off_diag_cos < 0.7

    reasons.append(f"δ_normalized spread = {delta_spread:.4f} (3 sub: {deltas[0]:.4f}, {deltas[1]:.4f}, {deltas[2]:.4f})")
    reasons.append(f"anisotropy_p95 spread = {aniso_spread:.4f} (3 sub: {anisos[0]:.4f}, {anisos[1]:.4f}, {anisos[2]:.4f})")
    reasons.append(f"norm_cv spread = {cv_spread:.4f} (3 sub: {cvs[0]:.4f}, {cvs[1]:.4f}, {cvs[2]:.4f})")
    reasons.append(f"subspace mean cos sim off-diag = {mean_off_diag_cos:.4f} (independence: <0.7)")

    if hyperbolic_signal:
        reasons.append("✓ hyperbolic signal: δ spread across κ subspaces > 0.05")
    else:
        reasons.append(f"✗ weak hyperbolic: δ spread = {delta_spread:.4f} ≤ 0.05")

    if spherical_signal:
        reasons.append(f"✓ spherical signal: aniso spread = {aniso_spread:.4f} > 0.1")
    else:
        reasons.append(f"△ moderate: aniso spread = {aniso_spread:.4f}")

    if independence_signal:
        reasons.append(f"✓ independence: subspace mean cos = {mean_off_diag_cos:.4f} < 0.7")
    else:
        reasons.append(f"✗ redundancy: subspace mean cos = {mean_off_diag_cos:.4f} ≥ 0.7")

    # 决策: 至少 2 个几何 spread 信号 + 独立性信号 → 通过
    # 更严格: 至少 3 个信号 (包括 euclidean CV spread, 这是最强信号)
    passed = sum([hyperbolic_signal, spherical_signal, euclidean_signal, independence_signal])
    # 至少 2 个信号通过且独立性必须通过 (避免学到冗余)
    go = (passed >= 2) and independence_signal and (passed >= 3 or euclidean_signal)
    # 简化为: ≥2 通过且独立性 + 至少一个几何 spread

    return {
        "deltas": deltas,
        "anisotropies": anisos,
        "norm_cvs": cvs,
        "delta_spread": delta_spread,
        "aniso_spread": aniso_spread,
        "cv_spread": cv_spread,
        "mean_off_diag_cos": mean_off_diag_cos,
        "reasons": reasons,
        "passed_signals": int(passed),
        "go_decision": "PROCEED" if go else "STOP",
    }


def plot_subspace_comparison(per_subspace: dict, kappas: list) -> None:
    """柱状图比较 3 个 κ 子空间的 δ / aniso / norm_cv"""
    labels = [f"κ={k:+.2f}" for k in kappas]
    deltas = [per_subspace[f"subspace_{i}"]["delta_normalized_mean"] for i in range(3)]
    anisos = [per_subspace[f"subspace_{i}"]["local_anisotropy_p95"] for i in range(3)]
    cvs = [per_subspace[f"subspace_{i}"]["norm_stats"]["norm_cv"] for i in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]  # 红/蓝/绿对应 sphere/euclid/hyperbolic

    for ax, vals, ylabel, title in zip(
        axes, [deltas, anisos, cvs],
        ["δ_normalized", "anisotropy_p95", "norm_cv"],
        ["Hyperbolicity (higher = more tree-like)", "Anisotropy (sphere indicator)", "Norm CV (spread indicator)"],
    ):
        bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Task #22 D0: MCKG 3 κ-subspace geometry comparison", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "subspace_comparison.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[info] saved: {out}")


def plot_tsne_subspaces(subspaces: torch.Tensor, kappas: list) -> None:
    """3 个 κ 子空间并排 t-SNE"""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(subspaces.shape[1], size=N_TSNE, replace=False)
    sub0 = subspaces[0][idx].numpy()
    sub1 = subspaces[1][idx].numpy()
    sub2 = subspaces[2][idx].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, k in zip(axes, [sub0, sub1, sub2], kappas):
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=500)
        proj = tsne.fit_transform(data)
        ax.scatter(proj[:, 0], proj[:, 1], s=2, alpha=0.6, c="steelblue")
        ax.set_title(f"κ = {k:+.3f} ({'sphere' if k > 0 else 'hyperbolic'})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Task #22 D0: MCKG 3 κ-subspace t-SNE (1000 items)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "tsne_3_subspaces.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[info] saved: {out}")


def plot_fused_tsne(fused: torch.Tensor) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(fused.shape[0], size=N_TSNE, replace=False)
    sub = fused[idx].numpy()
    pca = PCA(n_components=min(20, sub.shape[1])).fit_transform(sub)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=500)
    proj = tsne.fit_transform(pca)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(proj[:, 0], proj[:, 1], s=2, alpha=0.6, c="steelblue")
    ax.set_title("MCKG fused_item t-SNE")
    ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "tsne_fused.png"
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[info] saved: {out}")


def write_report(subspaces, fused, kappas, per_subspace, fused_metrics, indep, decision) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task #22 PM-RQ Phase 0 (D0) — MCKG 几何诊断报告",
        "",
        "> **数据源**: `products/task19/mckg_M3_c0.5_dim32_toys/entity_embedding.pt`",
        f"> **MCKG kappas**: {[f'{k:+.4f}' for k in kappas]}",
        f"> **subspace_item shape**: {tuple(subspaces.shape)}, **fused_item shape**: {tuple(fused.shape)}",
        "> **环境**: grid_toys (Python 3.10, torch, sklearn 1.7.2)",
        "> **时间**: 2026-07-19",
        "",
        "---",
        "",
        "## 1. D0 决策结论",
        "",
        f"**结论**: `{decision['go_decision']}` — {decision['passed_signals']}/4 信号通过",
        "",
        "**关键信号汇总**:",
        "",
        "| 信号 | 阈值 | 实测 | 判定 |",
        "|------|------|------|------|",
        f"| δ-hyperbolicity spread | > 0.05 | {decision['delta_spread']:.4f} | "
        f"{'✓' if decision['delta_spread'] > 0.05 else '✗'} |",
        f"| anisotropy_p95 spread | > 0.1 | {decision['aniso_spread']:.4f} | "
        f"{'✓' if decision['aniso_spread'] > 0.1 else '✗'} |",
        f"| norm_cv spread | > 0.05 | {decision['cv_spread']:.4f} | "
        f"{'✓' if decision['cv_spread'] > 0.05 else '✗'} |",
        f"| subspace mean cos sim | < 0.7 | {decision['mean_off_diag_cos']:.4f} | "
        f"{'✓' if decision['mean_off_diag_cos'] < 0.7 else '✗'} |",
        "",
        "理由:",
    ]
    for r in decision["reasons"]:
        lines.append(f"- {r}")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-κ 子空间几何诊断",
        "",
        f"MCKG 用 3 个 κ 训练: κ₀ = {kappas[0]:+.4f} (sphere), "
        f"κ₁ = {kappas[1]:+.4f} (mild hyperbolic), κ₂ = {kappas[2]:+.4f} (strong hyperbolic)",
        "",
        "### 2.1 δ-Hyperbolicity (Gromov 4-point)",
        "",
        "| κ | diameter | δ_mean | δ_normalized_mean | 解读 |",
        "|---|----------|--------|--------------------|------|",
    ])
    for i, k in enumerate(kappas):
        m = per_subspace[f"subspace_{i}"]
        interpretation = "强双曲" if m["delta_normalized_mean"] < 0.05 else ("弱双曲" if m["delta_normalized_mean"] < 0.3 else "非双曲")
        lines.append(
            f"| κ={k:+.4f} | {m['diameter']:.4f} | {m['delta_mean']:.4f} | "
            f"**{m['delta_normalized_mean']:.4f}** | {interpretation} |"
        )

    lines.extend([
        "",
        "### 2.2 局部各向异性 (PCA neighborhood)",
        "",
        "| κ | global_pca_top1 | local_aniso_mean | local_aniso_p95 | 解读 |",
        "|---|-----------------|-------------------|-----------------|------|",
    ])
    for i, k in enumerate(kappas):
        m = per_subspace[f"subspace_{i}"]
        sphere_score = m["local_anisotropy_p95"]
        interp = "球面 (anisotropy 高)" if sphere_score > 0.5 else ("中等" if sphere_score > 0.3 else "各向同性 (欧氏主导)")
        lines.append(
            f"| κ={k:+.4f} | {m['global_pca_top1']:.4f} | {m['local_anisotropy_mean']:.4f} | "
            f"**{m['local_anisotropy_p95']:.4f}** | {interp} |"
        )

    lines.extend([
        "",
        "### 2.3 范数分布 (球面 vs 欧氏 vs 双曲 指示器)",
        "",
        "| κ | mean | std | min | max | norm_cv | 解读 |",
        "|---|------|-----|-----|-----|---------|------|",
    ])
    for i, k in enumerate(kappas):
        m = per_subspace[f"subspace_{i}"]["norm_stats"]
        cv = m["norm_cv"]
        interp = "球面 (cv<0.05)" if cv < 0.05 else ("欧氏 (cv>0.1)" if cv > 0.1 else "中间")
        lines.append(
            f"| κ={k:+.4f} | {m['norm_mean']:.3f} | {m['norm_std']:.3f} | "
            f"{m['norm_min']:.3f} | {m['norm_max']:.3f} | **{m['norm_cv']:.4f}** | {interp} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. 子空间独立性 (3 κ 子空间是否学到不同信号)",
        "",
        "### 3.1 Mean vector cosine similarity (3×3)",
        "",
        "| | κ₀ | κ₁ | κ₂ |",
        "|---|----|----|----|",
    ])
    cos_mat = indep["mean_vector_cos_sim"]
    for i in range(3):
        row = " | ".join(f"{cos_mat[i][j]:+.4f}" for j in range(3))
        lines.append(f"| κ={kappas[i]:+.4f} | {row} |")

    lines.extend([
        "",
        f"**off-diagonal mean = {decision['mean_off_diag_cos']:.4f}**",
        "",
        "- < 0.3: 三个 κ 高度独立, 学的是不同信号",
        "- 0.3-0.7: 中等相关性, 部分冗余",
        "- > 0.7: 强相关, 三个 κ 学到几乎相同信号 → PM-RQ 无效",
        "",
        "### 3.2 Per-item norm Pearson correlation",
        "",
        "| | κ₀ | κ₁ | κ₂ |",
        "|---|----|----|----|",
    ])
    pear = indep["norm_seq_pearson"]
    for i in range(3):
        row = " | ".join(f"{pear[i][j]:+.4f}" for j in range(3))
        lines.append(f"| κ={kappas[i]:+.4f} | {row} |")

    lines.extend([
        "",
        "### 3.3 Per-item vector cos sim (随机 500 items)",
        "",
        "| | κ₀ | κ₁ | κ₂ |",
        "|---|----|----|----|",
    ])
    pcos = indep["per_item_cos_sim_mean"]
    for i in range(3):
        row = " | ".join(f"{pcos[i][j]:+.4f}" for j in range(3))
        lines.append(f"| κ={kappas[i]:+.4f} | {row} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Fused Item Embedding",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| shape | {tuple(fused.shape)} |",
        f"| δ_normalized_mean | {fused_metrics['delta_normalized_mean']:.4f} |",
        f"| local_anisotropy_p95 | {fused_metrics['local_anisotropy_p95']:.4f} |",
        f"| norm_cv | {fused_metrics['norm_stats']['norm_cv']:.4f} |",
        "",
        "---",
        "",
        "## 5. 产物",
        "",
        "- `products/task22_pm_rq/d0_figs/subspace_comparison.png` — 3 κ 子空间 δ/aniso/cv 对比",
        "- `products/task22_pm_rq/d0_figs/tsne_3_subspaces.png` — 3 κ 子空间 t-SNE",
        "- `products/task22_pm_rq/d0_figs/tsne_fused.png` — fused t-SNE",
        f"- `products/task22_pm_rq/d0_metrics.json` — 完整指标 JSON",
        "",
        "---",
        "",
        "## 6. Go/No-Go 决策",
        "",
        f"### 决策: **{decision['go_decision']}**",
        "",
        f"- {decision['passed_signals']}/4 信号通过",
        "",
        "**下一步**:",
    ])
    if decision["go_decision"] == "PROCEED":
        lines.extend([
            "- ✅ Phase 0 通过 → 进入 Phase 1 Toy Implementation (K=64, 10K items)",
            "- 三个 κ 子空间几何特征显著不同, PM-RQ 有合理动机",
            "- 启动命令见 `descriptions/task22_pm_rq_product_manifold.md` §2 Phase 1",
        ])
    else:
        lines.extend([
            "- ❌ Phase 0 否证 → MCKG 3 个 κ 子空间学到几乎相同的信号",
            "- 任务提前终止, 写 `verdicts/task22_result.md` 终局结论",
            "- 终局结论: MCKG 已经学到了 3 κ 几何表示, 但 3 κ 之间冗余度高 → PM-RQ 不会比 MCKG 单独 κ 表现更好",
        ])

    REPORT_PATH.write_text("\n".join(lines))
    print(f"[info] saved: {REPORT_PATH}")


def main() -> None:
    print("=" * 60)
    print("Task #22 PM-RQ Phase 0 (D0) — MCKG embedding 几何诊断")
    print("=" * 60)

    subspaces, fused, kappas = load_mckg()

    per_subspace = {}
    for i in range(3):
        print(f"\n--- κ subspace {i} (κ={kappas[i]:+.4f}) ---")
        sub = subspaces[i].numpy()
        print(f"  shape: {sub.shape}")
        delta = compute_delta_hyperbolicity(sub)
        aniso = compute_local_anisotropy(sub)
        norm = compute_norm_stats(subspaces[i])
        per_subspace[f"subspace_{i}"] = {**delta, **aniso, "norm_stats": norm}
        print(f"  δ_norm={delta['delta_normalized_mean']:.4f}, "
              f"aniso_p95={aniso['local_anisotropy_p95']:.4f}, "
              f"norm_cv={norm['norm_cv']:.4f}")

    print("\n--- Fused item ---")
    fused_delta = compute_delta_hyperbolicity(fused.numpy())
    fused_aniso = compute_local_anisotropy(fused.numpy())
    fused_norm = compute_norm_stats(fused)
    fused_metrics = {**fused_delta, **fused_aniso, "norm_stats": fused_norm}
    print(f"  δ_norm={fused_delta['delta_normalized_mean']:.4f}, "
          f"aniso_p95={fused_aniso['local_anisotropy_p95']:.4f}, "
          f"norm_cv={fused_norm['norm_cv']:.4f}")

    print("\n--- Subspace independence ---")
    indep = compute_pairwise_subspace_independence(subspaces)
    cos_mat = np.array(indep["mean_vector_cos_sim"])
    off_diag = []
    for i in range(3):
        for j in range(3):
            if i != j:
                off_diag.append(cos_mat[i, j])
    print(f"  off-diag mean cos sim = {float(np.mean(off_diag)):.4f}")

    decision = make_decision(per_subspace, indep)
    print("\n--- D0 Decision ---")
    print(f"  {decision['go_decision']} ({decision['passed_signals']}/4 signals)")
    for r in decision["reasons"]:
        print(f"  - {r}")

    print("\n--- Plots ---")
    plot_subspace_comparison(per_subspace, kappas)
    plot_tsne_subspaces(subspaces, kappas)
    plot_fused_tsne(fused)

    print("\n--- Report ---")
    write_report(subspaces, fused, kappas, per_subspace, fused_metrics, indep, decision)

    out_json = FIG_DIR.parent / "d0_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "kappas": kappas,
        "per_subspace": per_subspace,
        "fused_metrics": fused_metrics,
        "subspace_independence": indep,
        "decision": decision,
    }, indent=2))
    print(f"[info] saved: {out_json}")

    print("\n[done] Task #22 D0 (MCKG 版) 诊断完成")


if __name__ == "__main__":
    main()