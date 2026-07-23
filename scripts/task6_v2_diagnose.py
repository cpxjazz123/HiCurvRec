"""
task10_v2_diagnose.py — Task 14 v2：normalize_residuals=False 对照实验

输入：normalize_residuals=False 训练的 RKMeans ckpt + LLM embedding
输出：decay 向量 + CV
对照：v1 (normalize=True) CV=0.051 vs v2 (normalize=False) CV=?
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


def quantize_one_layer(x: torch.Tensor, centroids: torch.Tensor):
    dist = torch.cdist(x.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    idx = dist.argmin(dim=1)
    return centroids[idx], idx


def compute_residuals(embedding: torch.Tensor, codebooks: list, normalize: bool):
    residuals = []
    r = embedding.float().clone()
    for cb in codebooks:
        if normalize:
            r = torch.nn.functional.normalize(r, dim=-1)
        q, _ = quantize_one_layer(r, cb)
        r = r - q
        residuals.append(r.detach().cpu().numpy())
    return residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task10_v2_results")
    parser.add_argument("--v1_report", default="/home/wlia0047/ar57/wenyu/GeneRec/task10_results/task10_decay_report.json",
                        help="v1 (normalize=True) report for comparison")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ckpt
    print(f"[load] ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    codebooks = []
    i = 0
    while f"quantization_layer_list.{i}.centroids" in state:
        codebooks.append(state[f"quantization_layer_list.{i}.centroids"])
        i += 1
    n_layers = len(codebooks)
    normalize_residuals = hp.get("normalize_residuals", False)
    print(f"  n_layers={n_layers}, normalize_residuals={normalize_residuals}")

    # Centroids norm 统计
    cb_norms = [cb.norm(dim=1).mean().item() for cb in codebooks]
    print(f"  centroid mean norm per layer: {cb_norms}")

    # Load embedding
    embedding = torch.load(args.embedding, weights_only=True).float()
    N, D = embedding.shape
    print(f"  embedding shape={tuple(embedding.shape)}")

    # Compute residuals
    residuals = compute_residuals(embedding, codebooks, normalize=normalize_residuals)

    # Per-dim mean of |residual|
    means = np.zeros((n_layers, D))
    for l in range(n_layers):
        means[l] = np.abs(residuals[l]).mean(axis=0)
        rnorm = np.linalg.norm(residuals[l], axis=1).mean()
        print(f"  layer {l}: residual mean norm = {rnorm:.6f}")

    # decay_i = m_last / m_first
    m1 = means[0]
    mL = means[-1]
    decay = mL / (m1 + 1e-12)

    cv = float(decay.std() / (decay.mean() + 1e-12))
    summary = {
        "mean": float(decay.mean()),
        "std": float(decay.std()),
        "min": float(decay.min()),
        "max": float(decay.max()),
        "p10": float(np.percentile(decay, 10)),
        "p25": float(np.percentile(decay, 25)),
        "p50": float(np.percentile(decay, 50)),
        "p75": float(np.percentile(decay, 75)),
        "p90": float(np.percentile(decay, 90)),
        "cv": cv,
    }

    # 加载 v1 对照
    v1_summary = None
    v1_path = Path(args.v1_report)
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        v1_summary = v1["decay_summary"]
        print(f"\n[v1 对照] normalize_residuals=True CV = {v1_summary['cv']:.4f}, decay mean = {v1_summary['mean']:.4f}")
    print(f"[v2  本次] normalize_residuals=False CV = {cv:.4f}, decay mean = {summary['mean']:.4f}")

    # 残差范数序列（v1 vs v2）
    rnorms_v2 = [float(np.linalg.norm(residuals[l], axis=1).mean()) for l in range(n_layers)]
    rnorms_v1 = [0.2351, 0.8702, 0.9212, 0.9404, 0.9490]  # 已知 v1 数据

    if cv > 0.5:
        judgment = f"CV={cv:.3f} > 0.5 → **衰减高度不均匀** → BD-RFSQ 分块处理有强动机"
    elif cv > 0.2:
        judgment = f"CV={cv:.3f} ∈ [0.2, 0.5] → **衰减中度不均匀** → BD-RFSQ 有边际收益"
    else:
        judgment = f"CV={cv:.3f} < 0.2 → **衰减均匀** → BD-RFSQ 分块处理收益有限"

    # 判断 v1 vs v2 差异
    if v1_summary is not None:
        if cv > v1_summary["cv"] * 2:
            comparison = (
                f"\n[v1 vs v2 解读] 去掉 normalize 后 CV 从 {v1_summary['cv']:.3f} 暴涨到 {cv:.3f} "
                f"({cv/v1_summary['cv']:.1f}x) → **normalize_residuals 是 task12 不均匀性的主要压制因素**"
            )
        elif cv < v1_summary["cv"] * 0.5:
            comparison = (
                f"\n[v1 vs v2 解读] 去掉 normalize 后 CV 从 {v1_summary['cv']:.3f} 降到 {cv:.3f} "
                f"→ 残差在 normalize 之前反而不均匀"
            )
        else:
            comparison = (
                f"\n[v1 vs v2 解读] CV 变化不显著（{v1_summary['cv']:.3f} → {cv:.3f}）"
                f"→ 衰减模式对 normalize 不敏感"
            )
    else:
        comparison = ""
    judgment_full = judgment + comparison

    # Save
    report = {
        "version": "v2",
        "normalize_residuals": normalize_residuals,
        "ckpt_path": str(args.ckpt),
        "embedding_path": str(args.embedding),
        "n_layers": n_layers,
        "embedding_shape": [N, D],
        "centroid_mean_norms": cb_norms,
        "residual_mean_norms_per_layer": rnorms_v2,
        "decay_summary": summary,
        "v1_comparison": v1_summary,
        "judgment": judgment_full,
    }
    with open(out_dir / "task10_v2_decay_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save(out_dir / "task10_v2_decay.npy", decay)
    np.save(out_dir / "task10_v2_per_layer_means.npy", means)
    print(f"\n[report] saved {out_dir}/task10_v2_decay_report.json")
    print(f"\n=== Summary ===")
    print(f"  v1 (normalize=True)  CV = {v1_summary['cv'] if v1_summary else 'N/A'}")
    print(f"  v2 (normalize=False) CV = {cv}")
    print(f"  judgment: {judgment_full}")


if __name__ == "__main__":
    main()