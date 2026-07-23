"""
task10_dimension_decay.py — BD-RFSQ 前置检验：残差逐维度衰减均匀性

输入：RKMeans ckpt (含 5 个 codebooks) + LLM embedding (11924, 2048)
输出：
  - decay vector (D=2048,)
  - 直方图 PNG: task_artifacts/results/task10_decay_histogram.png
  - JSON 报告: task_artifacts/results/task10_decay_report.json

判断：CV (变异系数 = std/mean) > 0.5 → 衰减不均匀 → BD-RFSQ 假设成立
"""

import sys
import json
import argparse
from pathlib import Path

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


def quantize_one_layer(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """找到每个 x 的最近 centroid，返回该 centroid"""
    # x: (N, D), centroids: (W, D)
    # dist: (N, W)
    dist = torch.cdist(x.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # (N, W)
    idx = dist.argmin(dim=1)  # (N,)
    return centroids[idx], idx  # (N, D), (N,)


def compute_residuals(embedding: torch.Tensor, codebooks: list) -> list:
    """逐层量化，返回每层残差 list of (N, D) tensors"""
    residuals = []
    r = embedding.clone()
    for cb in codebooks:
        # 如果 normalize_residuals=True，先 normalize
        # 这里直接从训练产物推断
        q, _ = quantize_one_layer(r, cb)
        r = r - q
        residuals.append(r.detach().cpu())
    return residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to RKMeans LightningModule ckpt")
    parser.add_argument("--embedding", required=True, help="Path to LLM embedding pt")
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task10_results")
    parser.add_argument("--normalize", action="store_true", help="Match ckpt normalize_residuals=True")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load RKMeans ckpt
    print(f"[load] ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    # extract codebooks (key: quantization_layer_list.{i}.centroids)
    codebooks = []
    layer_idx = 0
    while f"quantization_layer_list.{layer_idx}.centroids" in state:
        cb = state[f"quantization_layer_list.{layer_idx}.centroids"]
        codebooks.append(cb)
        layer_idx += 1
    n_layers = len(codebooks)
    normalize_residuals = hp.get("normalize_residuals", False)
    input_dim = hp.get("input_dim", codebooks[0].shape[1])
    print(f"  n_layers={n_layers}, normalize_residuals={normalize_residuals}, input_dim={input_dim}")

    # 2. Load embedding
    print(f"[load] embedding: {args.embedding}")
    embedding = torch.load(args.embedding, weights_only=True)  # (N, D)
    if embedding.dim() != 2:
        raise ValueError(f"embedding must be 2D, got {embedding.shape}")
    N, D = embedding.shape
    print(f"  shape={tuple(embedding.shape)} dtype={embedding.dtype}")
    if D != input_dim:
        print(f"  [warn] embedding dim {D} != ckpt input_dim {input_dim}, will truncate/use first")

    # 3. Compute residuals per layer
    print("[compute] per-layer residuals ...")
    residuals = []
    r = embedding.float().clone()
    for l, cb in enumerate(codebooks):
        if normalize_residuals:
            r = torch.nn.functional.normalize(r, dim=-1)
        q, _ = quantize_one_layer(r, cb)
        r = r - q
        residuals.append(r.detach().cpu().numpy())  # (N, D)
        # Report
        rnorm = np.linalg.norm(r.numpy(), axis=1).mean()
        print(f"  layer {l}: residual mean norm = {rnorm:.6f}")

    # 4. Compute per-dim mean of |residual|
    print("[compute] per-dimension mean of |residual| ...")
    means = np.zeros((n_layers, D))
    for l in range(n_layers):
        means[l] = np.abs(residuals[l]).mean(axis=0)  # (D,)

    # 5. Compute decay_i = m_last / m_first  (注意：用户要求 m3/m1，但 RKMeans 是 5 层)
    #    用 m_n_layers / m_0
    m1 = means[0]
    mL = means[-1]
    decay = mL / (m1 + 1e-12)  # (D,)

    # 6. Summary stats
    report = {
        "ckpt_path": str(args.ckpt),
        "embedding_path": str(args.embedding),
        "n_layers": n_layers,
        "normalize_residuals": normalize_residuals,
        "embedding_shape": [N, D],
        "decay_summary": {
            "mean": float(decay.mean()),
            "std": float(decay.std()),
            "min": float(decay.min()),
            "max": float(decay.max()),
            "p10": float(np.percentile(decay, 10)),
            "p25": float(np.percentile(decay, 25)),
            "p50": float(np.percentile(decay, 50)),
            "p75": float(np.percentile(decay, 75)),
            "p90": float(np.percentile(decay, 90)),
            "cv": float(decay.std() / (decay.mean() + 1e-12)),  # 变异系数
        },
        "judgment": "",
    }
    cv = report["decay_summary"]["cv"]
    if cv > 0.5:
        report["judgment"] = (
            f"CV={cv:.3f} > 0.5 → **衰减高度不均匀** → BD-RFSQ 分块处理有动机"
        )
    elif cv > 0.2:
        report["judgment"] = (
            f"CV={cv:.3f} ∈ [0.2, 0.5] → **衰减中等不均匀** → BD-RFSQ 有边际收益"
        )
    else:
        report["judgment"] = (
            f"CV={cv:.3f} < 0.2 → **衰减均匀** → BD-RFSQ 分块处理收益有限"
        )

    # 7. Save decay vector as numpy
    np.save(out_dir / "task10_decay.npy", decay)
    np.save(out_dir / "task10_per_layer_means.npy", means)  # (n_layers, D)

    # 8. Plot histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        axes[0].hist(decay, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].axvline(decay.mean(), color="red", linestyle="--", label=f"mean={decay.mean():.4f}")
        axes[0].axvline(np.median(decay), color="green", linestyle="--", label=f"median={np.median(decay):.4f}")
        axes[0].set_xlabel("decay ratio (m_last / m_first)")
        axes[0].set_ylabel("number of dimensions")
        axes[0].set_title(
            f"Per-dimension decay distribution (CV={cv:.3f})\n"
            f"{report['judgment']}"
        )
        axes[0].legend()

        # Sorted decay (log scale y)
        sorted_decay = np.sort(decay)
        axes[1].plot(sorted_decay, lw=1)
        axes[1].set_xlabel("dimension rank (sorted)")
        axes[1].set_ylabel("decay ratio")
        axes[1].set_yscale("log")
        axes[1].set_title("Sorted decay (log scale) — tail dimensions converge faster")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "task10_decay_histogram.png", dpi=120)
        print(f"[plot] saved {out_dir}/task10_decay_histogram.png")
    except Exception as e:
        print(f"[plot] failed: {e}")

    # 9. Save JSON report
    with open(out_dir / "task10_decay_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[report] saved {out_dir}/task10_decay_report.json")
    print("\n=== Summary ===")
    print(f"  n_layers: {n_layers}")
    print(f"  normalize_residuals: {normalize_residuals}")
    print(f"  decay mean={decay.mean():.4f}, std={decay.std():.4f}, CV={cv:.3f}")
    print(f"  judgment: {report['judgment']}")


if __name__ == "__main__":
    main()