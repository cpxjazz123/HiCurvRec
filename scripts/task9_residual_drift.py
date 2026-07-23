"""
task1_residual_drift.py — Online RQ 前置检验：残差训练非平稳性

输入：多个 RKMeans ckpt (step 1000/2000/.../5000) + LLM embedding
输出：
  - per-layer 残差 norm/var 在 step 上的序列
  - 相邻 step 残差分布的漂移（Wasserstein / cosine similarity）
  - 时序图 PNG + JSON 报告

判断：drift > 10% → 残差训练非平稳 → Online RQ 有动机
      drift < 5%  → 残差训练平稳   → 静态 RQ 足够
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

# 修复 GRID import
GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
from src.utils.custom_hydra_resolvers import (  # noqa: F401
    remove_chars_from_string,
    conditional_expression,
    extract_fields_from_list_of_dicts,
    create_map_from_list_of_dicts,
    math_eval,
    remove_item_from_list,
)


def quantize_one_layer(x: torch.Tensor, centroids: torch.Tensor):
    """找最近 centroid，返回 quantize 后的向量"""
    dist = torch.cdist(x.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # (N, W)
    idx = dist.argmin(dim=1)
    return centroids[idx], idx


def compute_layer_residuals(embedding: torch.Tensor, codebooks: list, normalize_residuals: bool = True) -> List[np.ndarray]:
    """逐层量化，返回每层残差 (N, D) np.ndarray 列表"""
    residuals = []
    r = embedding.float().clone()
    for cb in codebooks:
        if normalize_residuals:
            r = torch.nn.functional.normalize(r, dim=-1)
        q, _ = quantize_one_layer(r, cb)
        r = r - q
        residuals.append(r.detach().cpu().numpy())
    return residuals


def load_rkmeans_ckpt(ckpt_path: str):
    """加载 RKMeans LightningModule ckpt，返回 (codebooks list, hyper_params dict)"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    codebooks = []
    i = 0
    while f"quantization_layer_list.{i}.centroids" in state:
        codebooks.append(state[f"quantization_layer_list.{i}.centroids"])
        i += 1
    return codebooks, hp


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein 距离（两个分布的 L1-排序距离）"""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    # 长度不一致时插值
    if len(a_sorted) != len(b_sorted):
        n = max(len(a_sorted), len(b_sorted))
        a_sorted = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
        b_sorted = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def cosine_sim_per_sample(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """逐样本计算 r1 vs r2 的余弦相似度"""
    n1 = np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12
    n2 = np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12
    return (r1 * r2 / (n1 * n2)).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Paths to RKMeans ckpts (in training-time order, e.g. step1000 step2000 step3000)",
    )
    parser.add_argument("--steps", nargs="+", type=int, required=True, help="Step numbers matching ckpts order")
    parser.add_argument("--embedding", required=True, help="Path to LLM embedding pt")
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task11_results")
    parser.add_argument("--max_items", type=int, default=2000, help="Subsample N items to speed up")
    args = parser.parse_args()

    assert len(args.ckpts) == len(args.steps), "ckpts and steps length mismatch"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding
    print(f"[load] embedding: {args.embedding}")
    embedding = torch.load(args.embedding, weights_only=True).float()  # (N, D)
    if args.max_items and embedding.shape[0] > args.max_items:
        # 固定种子保证多次实验一致
        torch.manual_seed(42)
        idx = torch.randperm(embedding.shape[0])[:args.max_items]
        embedding = embedding[idx]
        print(f"  subsample to {embedding.shape}")
    N, D = embedding.shape
    print(f"  shape={tuple(embedding.shape)}")

    # 加载所有 ckpt，逐个计算残差
    per_step_residuals = []  # list of [list of (N, D) per layer]
    per_step_meta = []
    for ckpt_path, step in zip(args.ckpts, args.steps):
        print(f"\n[load] step {step}: {ckpt_path}")
        codebooks, hp = load_rkmeans_ckpt(ckpt_path)
        n_layers = len(codebooks)
        normalize = hp.get("normalize_residuals", False)
        print(f"  n_layers={n_layers}, normalize_residuals={normalize}, input_dim={hp.get('input_dim', D)}")
        residuals = compute_layer_residuals(embedding, codebooks, normalize_residuals=normalize)
        per_step_residuals.append(residuals)
        per_step_meta.append({"step": step, "n_layers": n_layers, "normalize_residuals": normalize})

    # 收集 per-step per-layer 残差的 norm stats
    n_layers = per_step_meta[0]["n_layers"]
    layers_stats = []  # layers_stats[layer_idx] = list of dict per step
    for l in range(n_layers):
        per_step = []
        for s_idx, residuals in enumerate(per_step_residuals):
            r = residuals[l]  # (N, D)
            norm = np.linalg.norm(r, axis=1)
            per_step.append({
                "step": per_step_meta[s_idx]["step"],
                "norm_mean": float(norm.mean()),
                "norm_std": float(norm.std()),
                "abs_mean": float(np.abs(r).mean()),
                "abs_var": float(np.abs(r).var()),
            })
        layers_stats.append(per_step)

    # 计算相邻 step 的残差分布漂移
    print("\n[compute] drift between consecutive steps ...")
    drift_report = []  # list of dict per (layer, step_pair)
    for l in range(n_layers):
        for s_idx in range(len(per_step_residuals) - 1):
            r1 = per_step_residuals[s_idx][l]
            r2 = per_step_residuals[s_idx + 1][l]

            # Norm 分布的 Wasserstein 距离
            n1 = np.linalg.norm(r1, axis=1)
            n2 = np.linalg.norm(r2, axis=1)
            w_norm = wasserstein_1d(n1, n2)

            # |residual| 全元素的 Wasserstein
            w_abs = wasserstein_1d(np.abs(r1).flatten(), np.abs(r2).flatten())

            # 逐样本余弦相似度
            cos = cosine_sim_per_sample(r1, r2)
            cos_mean = float(cos.mean())
            cos_std = float(cos.std())

            # Norm 均值相对变化
            mean_norm_1 = float(n1.mean())
            mean_norm_2 = float(n2.mean())
            mean_shift = abs(mean_norm_2 - mean_norm_1) / (mean_norm_1 + 1e-12)

            drift_report.append({
                "layer": l,
                "step_from": per_step_meta[s_idx]["step"],
                "step_to": per_step_meta[s_idx + 1]["step"],
                "wasserstein_norm": w_norm,
                "wasserstein_abs": w_abs,
                "cosine_sim_mean": cos_mean,
                "cosine_sim_std": cos_std,
                "mean_norm_shift": mean_shift,
            })
            print(
                f"  layer {l}, step {per_step_meta[s_idx]['step']} -> {per_step_meta[s_idx+1]['step']}: "
                f"||r||_mean Δ={mean_shift*100:.2f}%, W_norm={w_norm:.4f}, cos_mean={cos_mean:.4f}"
            )

    # 计算整体漂移幅度
    #    max(per-step-pair relative norm shift) across all (layer, step_pair)
    max_mean_shift = max(d["mean_norm_shift"] for d in drift_report)
    avg_cos = np.mean([d["cosine_sim_mean"] for d in drift_report])

    if max_mean_shift > 0.10:
        judgment = (
            f"max norm shift = {max_mean_shift*100:.1f}% > 10% → **残差训练高度非平稳** → "
            f"Online RQ 边训边更新 codebook 有强动机"
        )
    elif max_mean_shift > 0.05:
        judgment = (
            f"max norm shift = {max_mean_shift*100:.1f}% ∈ [5%, 10%] → **残差训练中度非平稳** → "
            f"Online RQ 有边际收益"
        )
    else:
        judgment = (
            f"max norm shift = {max_mean_shift*100:.1f}% < 5% → **残差训练平稳** → "
            f"静态 RQ 已足够，Online RQ 收益有限"
        )

    report = {
        "ckpt_paths": args.ckpts,
        "step_numbers": args.steps,
        "embedding_path": args.embedding,
        "embedding_shape": [N, D],
        "n_layers": n_layers,
        "per_step_meta": per_step_meta,
        "per_layer_per_step_stats": layers_stats,
        "drift_per_step_pair": drift_report,
        "summary": {
            "max_mean_norm_shift": float(max_mean_shift),
            "avg_cosine_sim_mean": float(avg_cos),
            "judgment": judgment,
        },
    }

    # Save JSON
    with open(out_dir / "task1_drift_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[report] saved {out_dir}/task1_drift_report.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1) ||r||_mean vs step per layer
        for l in range(n_layers):
            steps = [s["step"] for s in layers_stats[l]]
            means = [s["norm_mean"] for s in layers_stats[l]]
            axes[0, 0].plot(steps, means, "o-", label=f"layer {l}")
        axes[0, 0].set_xlabel("training step")
        axes[0, 0].set_ylabel("||residual||_mean")
        axes[0, 0].set_title("Residual norm mean vs training step")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2) cos sim vs step pair per layer
        for l in range(n_layers):
            xs = [f"{d['step_from']}->{d['step_to']}" for d in drift_report if d["layer"] == l]
            ys = [d["cosine_sim_mean"] for d in drift_report if d["layer"] == l]
            axes[0, 1].plot(range(len(ys)), ys, "o-", label=f"layer {l}")
        axes[0, 1].set_xticks(range(len(xs)))
        axes[0, 1].set_xticklabels(xs, rotation=30)
        axes[0, 1].set_ylabel("cosine sim (mean)")
        axes[0, 1].set_title("Per-sample residual cosine similarity (closer to 1 = less drift)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(0.95, color="red", linestyle="--", alpha=0.5, label="0.95 threshold")

        # 3) mean norm shift per step pair per layer
        for l in range(n_layers):
            xs = [f"{d['step_from']}->{d['step_to']}" for d in drift_report if d["layer"] == l]
            ys = [d["mean_norm_shift"] * 100 for d in drift_report if d["layer"] == l]
            axes[1, 0].bar(range(len(ys)), ys, label=f"layer {l}")
        axes[1, 0].set_xticks(range(len(xs)))
        axes[1, 0].set_xticklabels(xs, rotation=30)
        axes[1, 0].set_ylabel("relative ||r||_mean shift (%)")
        axes[1, 0].set_title(f"Residual norm mean shift per step pair\n{judgment}")
        axes[1, 0].axhline(5, color="orange", linestyle="--", alpha=0.5, label="5% threshold")
        axes[1, 0].axhline(10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4) wasserstein norm distance
        for l in range(n_layers):
            xs = [f"{d['step_from']}->{d['step_to']}" for d in drift_report if d["layer"] == l]
            ys = [d["wasserstein_norm"] for d in drift_report if d["layer"] == l]
            axes[1, 1].plot(range(len(ys)), ys, "o-", label=f"layer {l}")
        axes[1, 1].set_xticks(range(len(xs)))
        axes[1, 1].set_xticklabels(xs, rotation=30)
        axes[1, 1].set_ylabel("Wasserstein distance (||r|| distribution)")
        axes[1, 1].set_title("Wasserstein distance of residual norm distribution")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "task11_drift.png", dpi=120)
        print(f"[plot] saved {out_dir}/task11_drift.png")
    except Exception as e:
        print(f"[plot] failed: {e}")

    # Print summary
    print("\n=== Summary ===")
    print(f"  embedding shape: {N} x {D}")
    print(f"  n_layers: {n_layers}")
    print(f"  step range: {min(args.steps)} to {max(args.steps)}")
    print(f"  max mean norm shift: {max_mean_shift*100:.2f}%")
    print(f"  avg cosine sim mean: {avg_cos:.4f}")
    print(f"  judgment: {judgment}")


if __name__ == "__main__":
    main()