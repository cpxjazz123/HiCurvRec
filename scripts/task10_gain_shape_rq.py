"""
task2_gain_shape_rq.py — Gain-Shape RQ 信息分解实验

目标：检验 Gain-Shape RQ 核心假设——残差方向（shape）vs 大小（gain）哪个携带更多下游信息

步骤：
1. 从 Toys training tfrecord 提取商品 text → 解析 L2 category 标签
2. 用 RKMeans ckpt 提取第 3 层残差矩阵 R (N_with_text, 2048)
3. 拆分：gain = ||R[i]||, shape = R[i] / ||R[i]||
4. 构造 4 种特征：R_full, R_mag_only, R_shape_only, R_zero
5. 训练 Logistic Regression，5 次随机种子平均
6. 对比 shape_only vs mag_only 准确率

判断：acc(shape_only) / acc(mag_only) > 1.5 → 方向信息量更大 → Gain-Shape 拆分有动机
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
    """从 training tfrecord 收集每条 sample 的 sequence[0] 对应商品的 text"""
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
    """从 text 解析 L2 category（如 'Games' / 'Puzzles'）"""
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


def quantize_layers(embedding: torch.Tensor, codebooks: list, normalize: bool):
    """逐层量化，返回每层残差 (N, D) np.ndarray 列表"""
    residuals = []
    r = embedding.float().clone()
    for cb in codebooks:
        if normalize:
            r = torch.nn.functional.normalize(r, dim=-1)
        dist = torch.cdist(r.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        q = cb[idx]
        r = r - q
        residuals.append(r.detach().cpu().numpy())
    return residuals


def build_feature_variants(R: np.ndarray) -> Dict[str, np.ndarray]:
    """构造 4 种特征变体"""
    gain = np.linalg.norm(R, axis=1, keepdims=True)  # (N, 1)
    shape = R / (gain + 1e-8)  # (N, D)
    ones = np.ones_like(shape)
    return {
        "full": R.astype(np.float32),                                   # 完整残差
        "mag_only": (gain * ones).astype(np.float32),                   # 只保留 gain
        "shape_only": (1.0 * shape).astype(np.float32),                 # 只保留 shape
        "zero": np.zeros_like(R, dtype=np.float32),                     # 零基线
    }


def train_eval_logreg(X: np.ndarray, y: np.ndarray, n_runs: int = 5, test_size: float = 0.2) -> Dict[str, float]:
    """训练 Logistic Regression，n_runs 次随机种子取平均"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    # 降维到 64 加速（2048 维太高，LogisticRegression 慢）
    n_components = min(64, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_low = svd.fit_transform(X)

    accs = []
    f1s_macro = []
    f1s_weighted = []
    for seed in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(X_low, y, test_size=test_size, random_state=seed, stratify=y)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(
            max_iter=300,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        accs.append(acc)
        from sklearn.metrics import f1_score
        f1s_macro.append(f1_score(y_te, clf.predict(X_te), average="macro", zero_division=0))
        f1s_weighted.append(f1_score(y_te, clf.predict(X_te), average="weighted", zero_division=0))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s_macro)),
        "f1_weighted_mean": float(np.mean(f1s_weighted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to RKMeans ckpt (≥3 layers)")
    parser.add_argument("--embedding", required=True, help="Path to LLM embedding pt")
    parser.add_argument("--data_dir", default="/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys")
    parser.add_argument("--layer", type=int, default=2, help="Which residual layer to analyze (0-indexed)")
    parser.add_argument("--min_class_count", type=int, default=10, help="Filter L2 categories with < N samples")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task12_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load item text + extract L2 category
    print("\n[step 1] extract item text and L2 category")
    item_text = extract_item_text(args.data_dir)
    item_l2 = {iid: extract_l2_category(t) for iid, t in item_text.items()}

    # 2. Load embedding (sorted by item_id)
    print(f"\n[step 2] load embedding: {args.embedding}")
    embedding = torch.load(args.embedding, weights_only=True).float()  # (N, D), N=11924
    N_total, D = embedding.shape
    print(f"  shape={tuple(embedding.shape)}")

    # 3. Filter items that have category
    has_cat = [i for i in range(N_total) if i in item_l2 and item_l2[i] != "Unknown"]
    print(f"\n[step 3] {len(has_cat)}/{N_total} items have L2 category")

    # Filter low-frequency categories
    cat_counter = Counter([item_l2[i] for i in has_cat])
    valid_cats = {c for c, n in cat_counter.items() if n >= args.min_class_count}
    keep = [i for i in has_cat if item_l2[i] in valid_cats]
    print(f"  after filter cat >= {args.min_class_count}: {len(keep)} items, {len(valid_cats)} classes")
    cat_counter_final = Counter([item_l2[i] for i in keep])
    print(f"  class distribution (top 10): {cat_counter_final.most_common(10)}")

    # 4. Load RKMeans ckpt + extract residual at given layer
    print(f"\n[step 4] load RKMeans ckpt: {args.ckpt}")
    codebooks, hp = load_rkmeans_ckpt(args.ckpt)
    n_layers = len(codebooks)
    normalize = hp.get("normalize_residuals", False)
    print(f"  n_layers={n_layers}, normalize_residuals={normalize}, input_dim={hp.get('input_dim', D)}")
    if args.layer >= n_layers:
        raise ValueError(f"layer {args.layer} > n_layers {n_layers}")

    # Compute residuals for all items
    print("[step 4.1] compute residuals for all items ...")
    all_residuals = quantize_layers(embedding, codebooks, normalize=normalize)
    R = all_residuals[args.layer]  # (N_total, D)
    print(f"  R shape={R.shape}, mean norm={np.linalg.norm(R, axis=1).mean():.4f}")

    # 5. Subset to keep items
    R_keep = R[keep]
    y_keep = np.array([item_l2[i] for i in keep])
    print(f"\n[step 5] keep subset shape={R_keep.shape}, n_classes={len(set(y_keep))}")

    # 6. Build 4 variants
    print("\n[step 6] build 4 feature variants")
    variants = build_feature_variants(R_keep)
    for name, X in variants.items():
        print(f"  {name}: shape={X.shape}, mean_abs={np.abs(X).mean():.6f}, std={X.std():.6f}")

    # 7. Train + Eval each variant
    print(f"\n[step 7] train logistic regression, n_runs={args.n_runs}")
    results = {}
    for name, X in variants.items():
        print(f"  [{name}] training ...")
        res = train_eval_logreg(X, y_keep, n_runs=args.n_runs)
        results[name] = res
        print(f"    acc={res['acc_mean']:.4f}±{res['acc_std']:.4f}, "
              f"f1_macro={res['f1_macro_mean']:.4f}, f1_weighted={res['f1_weighted_mean']:.4f}")

    # 8. Judgment
    acc_mag = results["mag_only"]["acc_mean"]
    acc_shape = results["shape_only"]["acc_mean"]
    acc_full = results["full"]["acc_mean"]
    acc_zero = results["zero"]["acc_mean"]
    ratio = acc_shape / (acc_mag + 1e-12)
    diff = acc_full - acc_shape

    judgment_lines = [
        f"acc(full)   = {acc_full:.4f}",
        f"acc(shape)  = {acc_shape:.4f}",
        f"acc(mag)    = {acc_mag:.4f}",
        f"acc(zero)   = {acc_zero:.4f}",
        f"acc(shape)/acc(mag) = {ratio:.2f}",
        f"acc(full) - acc(shape) = {diff:.4f}",
        "",
    ]
    if ratio > 1.5:
        judgment_lines.append(f"**shape 比 mag 准确率高 {ratio:.1f}x → 方向携带大部分信息 → Gain-Shape 拆分有强动机**")
    elif ratio > 1.1:
        judgment_lines.append(f"**shape 略胜 mag ({ratio:.2f}x) → 方向信息更多 → Gain-Shape 拆分有边际收益**")
    elif acc_mag > acc_shape * 1.1:
        judgment_lines.append(f"**mag 反而比 shape 信息量大 → Gain-Shape 假设可能错**")
    else:
        judgment_lines.append(f"**shape ≈ mag → 两者信息量相当 → Gain-Shape 拆分无显著收益**")
    if diff < 0.01 * acc_full:
        judgment_lines.append(f"  acc(full) - acc(shape) = {diff:.4f} < 1% × acc(full) → shape 已捕捉大部分信息")

    judgment = "\n".join(judgment_lines)

    # 9. Save report
    report = {
        "ckpt_path": args.ckpt,
        "embedding_path": args.embedding,
        "data_dir": args.data_dir,
        "layer": args.layer,
        "n_layers": n_layers,
        "normalize_residuals": normalize,
        "embedding_shape": [N_total, D],
        "n_items_with_category": len(has_cat),
        "n_items_kept": len(keep),
        "n_classes": len(valid_cats),
        "class_distribution": dict(cat_counter_final),
        "n_runs": args.n_runs,
        "results": results,
        "judgment": judgment,
    }
    with open(out_dir / "task2_gain_shape_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[report] saved {out_dir}/task2_gain_shape_report.json")
    print("\n=== Summary ===")
    print(judgment)


if __name__ == "__main__":
    main()