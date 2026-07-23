"""
task2_v2_gain_only.py — 只跑 v2 增益诊断（跳过 v1 logreg，v1 已有 JSON）

输入：
  - RKMeans ckpt
  - LLM embedding
  - v1 4 特征 logreg JSON (task2_results/task2_gain_shape_report.json)
  - 重建误差（这里重新算）
输出：
  - gain vs recon error 相关性
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


def load_rkmeans_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    codebooks = []
    i = 0
    while f"quantization_layer_list.{i}.centroids" in state:
        codebooks.append(state[f"quantization_layer_list.{i}.centroids"])
        i += 1
    return codebooks, hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--v1_report", default="/home/wlia0047/ar57/wenyu/GeneRec/task2_results/task2_gain_shape_report.json")
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task12_v2_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding
    print(f"[load] embedding: {args.embedding}")
    embedding = torch.load(args.embedding, weights_only=True).float()
    N_total, D = embedding.shape

    # Load RKMeans
    codebooks, hp = load_rkmeans_ckpt(args.ckpt)
    n_layers = len(codebooks)
    normalize = hp.get("normalize_residuals", False)
    print(f"  n_layers={n_layers}, normalize={normalize}")

    # 算逐层 partial reconstruction（不需要 category filter，这里用全 N=11924 算相关性更稳）
    print("\n[compute] partial reconstruction per layer ...")
    orig = embedding.numpy()  # (N, D)
    r = embedding.float().clone()
    acc_recon = torch.zeros_like(r)
    residuals = []
    partial_recons = []
    for l, cb in enumerate(codebooks):
        if normalize:
            r = torch.nn.functional.normalize(r, dim=-1)
        dist = torch.cdist(r.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        q = cb[idx]
        r = r - q
        acc_recon = acc_recon + q
        residuals.append(r.detach().cpu().numpy())
        partial_recons.append(acc_recon.detach().cpu().numpy())

    # 重建误差
    recon_errors = {}
    for l in range(n_layers):
        err = np.linalg.norm(orig - partial_recons[l], axis=1)
        recon_errors[l] = err
        print(f"  layer {l} (partial recon 0..{l}): mean={err.mean():.4f}, std={err.std():.4f}")

    # 算第 3 层（layer=args.layer）残差
    R = residuals[args.layer]  # (N, D)
    gain = np.linalg.norm(R, axis=1)  # (N,)
    print(f"\n[gain] layer {args.layer} gain: mean={gain.mean():.4f}, std={gain.std():.4f}, "
          f"min={gain.min():.4f}, max={gain.max():.4f}")

    # Spearman fallback
    try:
        from scipy.stats import spearmanr
        def compute_spearman(a, b):
            return float(spearmanr(a, b).statistic)
    except ImportError:
        def compute_spearman(a, b):
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            ra -= ra.mean()
            rb -= rb.mean()
            denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
            return float((ra * rb).sum() / (denom + 1e-12))

    # 相关性
    print(f"\n[correlate] gain vs reconstruction error:")
    correlations = []
    for l in range(n_layers):
        err = recon_errors[l]
        pearson = float(np.corrcoef(gain, err)[0, 1])
        spearman = compute_spearman(gain, err)
        correlations.append({
            "layer": l,
            "partial_recon_layers": f"0..{l}",
            "recon_error_mean": float(err.mean()),
            "recon_error_std": float(err.std()),
            "pearson_corr_gain_vs_recon": pearson,
            "spearman_corr_gain_vs_recon": spearman,
        })
        print(f"  layer {l} (partial 0..{l}): Pearson={pearson:.4f}, Spearman={spearman:.4f}")

    # 加载 v1 JSON 作对照
    v1_data = None
    v1_path = Path(args.v1_report)
    if v1_path.exists():
        with open(v1_path) as f:
            v1_data = json.load(f)

    # 判断
    max_pearson = max(c["pearson_corr_gain_vs_recon"] for c in correlations)
    max_corr_layer = max(correlations, key=lambda c: c["pearson_corr_gain_vs_recon"])["layer"]
    if max_pearson > 0.7:
        gain_judgment = (
            f"max Pearson = {max_pearson:.3f} > 0.7 → **gain 强烈反映'前层重建误差' → gain 是误差信号，不是语义信号**\n"
            f"  → 这解释了 task14 v1 结论：mag_only 比 zero 基线还差（因为 gain 是'哪里没量准'的标记，不是商品本身的语义属性）"
        )
    elif max_pearson > 0.5:
        gain_judgment = (
            f"max Pearson = {max_pearson:.3f} ∈ [0.5, 0.7] → **gain 主要反映重建误差** → gain 是误差信号"
        )
    elif max_pearson > 0.3:
        gain_judgment = (
            f"max Pearson = {max_pearson:.3f} ∈ [0.3, 0.5] → gain 与重建误差中度相关，可能部分反映误差信号"
        )
    else:
        gain_judgment = (
            f"max Pearson = {max_pearson:.3f} < 0.3 → gain 与重建误差相关性弱，需排查其他来源"
        )

    # 组装 v2 报告
    report = {
        "version": "v2_gain_only",
        "ckpt_path": str(args.ckpt),
        "embedding_path": args.embedding,
        "layer_for_gain": args.layer,
        "n_layers": n_layers,
        "normalize_residuals": normalize,
        "embedding_shape": [N_total, D],
        "recon_error_per_layer": {
            f"layer_{l}": {"mean": float(recon_errors[l].mean()), "std": float(recon_errors[l].std())}
            for l in range(n_layers)
        },
        "gain_stats": {
            "mean": float(gain.mean()),
            "std": float(gain.std()),
            "min": float(gain.min()),
            "max": float(gain.max()),
        },
        "gain_vs_recon_correlation": correlations,
        "max_pearson": float(max_pearson),
        "max_pearson_layer": int(max_corr_layer),
        "v1_results_for_comparison": v1_data.get("results", {}) if v1_data else {},
        "v1_judgment": v1_data.get("judgment", "") if v1_data else "",
        "gain_judgment": gain_judgment,
    }
    with open(out_dir / "task2_v2_gain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save(out_dir / "task12_v2_gain.npy", gain)
    np.save(out_dir / "task12_v2_R_layer2.npy", R)
    for l in range(n_layers):
        np.save(out_dir / f"task12_v2_recon_err_layer_{l}.npy", recon_errors[l])

    print(f"\n[report] saved {out_dir}/task2_v2_gain_report.json")
    print(f"\n=== Summary ===")
    print(f"  max Pearson (gain vs recon_err) = {max_pearson:.4f} at layer {max_corr_layer}")
    print(f"  gain_judgment: {gain_judgment}")


if __name__ == "__main__":
    main()