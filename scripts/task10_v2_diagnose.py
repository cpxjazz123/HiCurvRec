"""
task2_v2_diagnose.py — Task 16 v2：gain 编码内容诊断

在 v1 基础上新增诊断：
  算每个商品"前 N 层重建误差" ||orig - quant_first_N_layers||
  与第 3 层 gain = ||R_layer3|| 算皮尔逊相关系数
  假设：gain 与"前 N 层重建误差"相关性高 → gain 是误差信号，不是语义信号

保留 v1 的 4 种特征对比结果作对照
"""

import sys
import json
import re
import argparse
import glob
from pathlib import Path
from collections import Counter
from typing import Dict, List

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

import tensorflow as tf


def extract_item_text(data_dir: str) -> Dict[int, str]:
    files = sorted(glob.glob(f"{data_dir}/training/partition_*.tfrecord.gz"))
    print(f"[load] {len(files)} training partition files")
    seen = {}
    for f in files:
        ds = tf.data.TFRecordDataset([f], compression_type="GZIP")
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.features.feature['sequence_data'].int64_list.value)
            if len(seq) == 0:
                continue
            item_id = int(seq[0])
            if item_id not in seen:
                text = ex.features.feature['text'].bytes_list.value[0].decode('utf-8', errors='ignore')
                seen[item_id] = text
    print(f"[load] got text for {len(seen)} items")
    return seen


def extract_l2_category(text: str) -> str:
    m = re.search(r"Categories:\s*\[(.+?)\]", text)
    if not m:
        return "Unknown"
    cats_str = m.group(1)
    cats = re.findall(r"'([^']+)'", cats_str)
    if len(cats) < 2:
        return "Unknown"
    return cats[1]


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


def quantize_layers(embedding: torch.Tensor, codebooks: list, normalize: bool, max_layer: int = None):
    """逐层量化，返回 (residuals_list, partial_recon_per_layer)
    partial_recon[l] = sum of quantized embeddings from layer 0 to layer l  (重建累加)
    """
    residuals = []
    partial_recon = []  # partial_recon[l] 是 layer 0..l 的累加重建
    r = embedding.float().clone()
    acc_recon = torch.zeros_like(r)
    L = len(codebooks) if max_layer is None else max_layer
    for l, cb in enumerate(codebooks[:L]):
        if normalize:
            r = torch.nn.functional.normalize(r, dim=-1)
        dist = torch.cdist(r.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        q = cb[idx]
        r = r - q
        acc_recon = acc_recon + q
        residuals.append(r.detach().cpu().numpy())
        partial_recon.append(acc_recon.detach().cpu().numpy())
    return residuals, partial_recon


def build_feature_variants(R: np.ndarray) -> Dict[str, np.ndarray]:
    gain = np.linalg.norm(R, axis=1, keepdims=True)
    shape = R / (gain + 1e-8)
    ones = np.ones_like(shape)
    return {
        "full": R.astype(np.float32),
        "mag_only": (gain * ones).astype(np.float32),
        "shape_only": (1.0 * shape).astype(np.float32),
        "zero": np.zeros_like(R, dtype=np.float32),
    }


def train_eval_logreg(X: np.ndarray, y: np.ndarray, n_runs: int = 5, test_size: float = 0.2) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import f1_score

    n_components = min(64, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_low = svd.fit_transform(X)

    accs, f1s_macro, f1s_weighted = [], [], []
    for seed in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(X_low, y, test_size=test_size, random_state=seed, stratify=y)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=300, multi_class="multinomial", solver="lbfgs", n_jobs=-1, random_state=seed)
        clf.fit(X_tr, y_tr)
        accs.append(clf.score(X_te, y_te))
        pred = clf.predict(X_te)
        f1s_macro.append(f1_score(y_te, pred, average="macro", zero_division=0))
        f1s_weighted.append(f1_score(y_te, pred, average="weighted", zero_division=0))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s_macro)),
        "f1_weighted_mean": float(np.mean(f1s_weighted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--data_dir", default="/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--v1_report", default="/home/wlia0047/ar57/wenyu/GeneRec/task2_results/task2_gain_shape_report.json")
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task12_v2_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========== v1 部分（保留对照）==========
    print("\n[step 1] extract item text + L2 category")
    item_text = extract_item_text(args.data_dir)
    item_l2 = {iid: extract_l2_category(t) for iid, t in item_text.items()}

    print(f"\n[step 2] load embedding: {args.embedding}")
    embedding = torch.load(args.embedding, weights_only=True).float()
    N_total, D = embedding.shape
    print(f"  shape={tuple(embedding.shape)}")

    print("\n[step 3] filter items with category")
    has_cat = [i for i in range(N_total) if i in item_l2 and item_l2[i] != "Unknown"]
    cat_counter = Counter([item_l2[i] for i in has_cat])
    valid_cats = {c for c, n in cat_counter.items() if n >= args.min_class_count}
    keep = [i for i in has_cat if item_l2[i] in valid_cats]
    print(f"  kept {len(keep)} items, {len(valid_cats)} classes")

    print(f"\n[step 4] load RKMeans ckpt: {args.ckpt}")
    codebooks, hp = load_rkmeans_ckpt(args.ckpt)
    n_layers = len(codebooks)
    normalize = hp.get("normalize_residuals", False)
    print(f"  n_layers={n_layers}, normalize={normalize}")
    if args.layer >= n_layers:
        raise ValueError(f"layer {args.layer} > n_layers {n_layers}")

    print("\n[step 5] compute residuals and partial reconstructions")
    all_residuals, all_partial_recon = quantize_layers(embedding, codebooks, normalize=normalize)
    R = all_residuals[args.layer]
    R_keep = R[keep]
    y_keep = np.array([item_l2[i] for i in keep])
    print(f"  R layer {args.layer} shape={R.shape}, keep subset shape={R_keep.shape}")

    # 重建误差 = ||orig - partial_recon||
    orig = embedding.numpy()[keep]  # (N, D)
    recon_errors = {}  # layer_idx -> (N,) ||orig - partial_recon[layer]||
    for l in range(n_layers):
        recon = all_partial_recon[l][keep]
        err = np.linalg.norm(orig - recon, axis=1)
        recon_errors[l] = err
        print(f"  layer {l}: recon error mean = {err.mean():.4f}, std = {err.std():.4f}")

    # ========== v1 特征分类对照（保留）==========
    print(f"\n[step 6] build 4 feature variants and run logreg (v1 对照)")
    variants = build_feature_variants(R_keep)
    v1_results = {}
    for name, X in variants.items():
        print(f"  [{name}] training ...")
        v1_results[name] = train_eval_logreg(X, y_keep, n_runs=args.n_runs)

    # ========== v2 新增：gain vs 重建误差 相关性 ==========
    print(f"\n[step 7] diagnose gain encoding: gain vs reconstruction error correlation")
    gain = np.linalg.norm(R_keep, axis=1)  # (N,)
    # 尝试 import scipy，否则用纯 numpy 计算 spearman
    try:
        from scipy.stats import spearmanr
        def compute_spearman(a, b):
            res = spearmanr(a, b)
            return float(res.statistic)
    except ImportError:
        def compute_spearman(a, b):
            # 纯 numpy 实现 spearman
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
            return float((ra * rb).sum() / (denom + 1e-12))

    correlations = []
    for l in range(n_layers):
        err = recon_errors[l]
        pearson = float(np.corrcoef(gain, err)[0, 1])
        spearman = compute_spearman(gain, err)
        correlations.append({
            "layer": l,
            "recon_error_mean": float(err.mean()),
            "recon_error_std": float(err.std()),
            "pearson_corr_gain_vs_recon": pearson,
            "spearman_corr_gain_vs_recon": spearman,
        })
        print(f"  layer {l} partial recon (layers 0..{l}): Pearson(gain, recon_err) = {pearson:.4f}, Spearman = {spearman:.4f}")

    # 判断：相关性 > 0.5 → gain 主要是误差信号
    max_corr = max(c["pearson_corr_gain_vs_recon"] for c in correlations)
    max_corr_layer = max(correlations, key=lambda c: c["pearson_corr_gain_vs_recon"])["layer"]
    if max_corr > 0.7:
        gain_judgment = f"**max Pearson = {max_corr:.3f} > 0.7 → gain 强烈反映'前层重建误差' → gain 是误差信号，不是语义信号**"
    elif max_corr > 0.5:
        gain_judgment = f"**max Pearson = {max_corr:.3f} ∈ [0.5, 0.7] → gain 主要反映重建误差 → 增益有信息但不语义**"
    elif max_corr > 0.3:
        gain_judgment = f"max Pearson = {max_corr:.3f} ∈ [0.3, 0.5] → gain 与重建误差中度相关"
    else:
        gain_judgment = f"max Pearson = {max_corr:.3f} < 0.3 → gain 不是误差信号，需排查其他来源"

    # v1 judgment
    v1_judgment_lines = [
        f"v1 4 特征 acc: full={v1_results['full']['acc_mean']:.4f}, "
        f"shape={v1_results['shape_only']['acc_mean']:.4f}, "
        f"mag={v1_results['mag_only']['acc_mean']:.4f}, "
        f"zero={v1_results['zero']['acc_mean']:.4f}"
    ]
    ratio = v1_results['shape_only']['acc_mean'] / (v1_results['mag_only']['acc_mean'] + 1e-12)
    v1_judgment_lines.append(f"v1 shape/mag 比例 = {ratio:.2f}")

    # Load v1 for comparison
    v1_data = None
    v1_path = Path(args.v1_report)
    if v1_path.exists():
        with open(v1_path) as f:
            v1_data = json.load(f)

    report = {
        "version": "v2",
        "ckpt_path": str(args.ckpt),
        "embedding_path": str(args.embedding),
        "data_dir": args.data_dir,
        "layer": args.layer,
        "n_layers": n_layers,
        "normalize_residuals": normalize,
        "embedding_shape": [N_total, D],
        "n_items_kept": len(keep),
        "n_classes": len(valid_cats),
        # v1 对照
        "v1_feature_results": v1_results,
        "v1_judgment": "\n".join(v1_judgment_lines),
        # v2 新增
        "recon_error_per_layer": {f"layer_{l}": {"mean": float(recon_errors[l].mean()), "std": float(recon_errors[l].std())} for l in range(n_layers)},
        "gain_vs_recon_correlation": correlations,
        "gain_judgment": gain_judgment,
    }

    with open(out_dir / "task2_v2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save(out_dir / "task12_v2_gain.npy", gain)
    np.save(out_dir / "task12_v2_R_layer2.npy", R_keep)
    for l in range(n_layers):
        np.save(out_dir / f"task12_v2_recon_err_layer_{l}.npy", recon_errors[l])

    print(f"\n[report] saved {out_dir}/task2_v2_report.json")
    print(f"\n=== Summary ===")
    print(f"  v1 4 特征对照:")
    for name, r in v1_results.items():
        print(f"    {name}: acc={r['acc_mean']:.4f}±{r['acc_std']:.4f}")
    print(f"\n  v2 gain 诊断:")
    for c in correlations:
        print(f"    layer {c['layer']}: Pearson={c['pearson_corr_gain_vs_recon']:.4f}, Spearman={c['spearman_corr_gain_vs_recon']:.4f}")
    print(f"  {gain_judgment}")


if __name__ == "__main__":
    main()