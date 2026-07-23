"""
task2_v3_1d_gain.py — 用 gain 一个标量直接做 18 类目多分类

直接方案：feature = gain[N,]，1D LogReg / 决策树 / 朴素贝叶斯
对比基线：5.6% = 1/18（均匀瞎猜）和 11.3%（top class 占比 = Games 789/6959）

如果 acc(gain) > 5.6% → 哪怕窄范围，gain 仍含信息
如果 acc(gain) ≈ 5.6% → gain 确实没用
如果 acc(gain) > 11.3% → gain 是强类别信号
"""

import sys
import json
import re
import argparse
import glob
from pathlib import Path
from collections import Counter
from typing import Dict

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--data_dir", default="/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task12_v3_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load items + L2 category
    print("\n[step 1] extract item text + L2 category")
    item_text = extract_item_text(args.data_dir)
    item_l2 = {iid: extract_l2_category(t) for iid, t in item_text.items()}

    # Load embedding
    print(f"\n[step 2] load embedding: {args.embedding}")
    embedding = torch.load(args.embedding, weights_only=True).float()
    N_total, D = embedding.shape

    # Filter items
    has_cat = [i for i in range(N_total) if i in item_l2 and item_l2[i] != "Unknown"]
    cat_counter = Counter([item_l2[i] for i in has_cat])
    valid_cats = {c for c, n in cat_counter.items() if n >= args.min_class_count}
    keep = [i for i in has_cat if item_l2[i] in valid_cats]
    y_keep = np.array([item_l2[i] for i in keep])
    n_classes = len(valid_cats)
    print(f"\n[step 3] kept {len(keep)} items, {n_classes} classes")
    top_class_count = max(cat_counter[c] for c in valid_cats)
    top_class_name = max((c for c in valid_cats if cat_counter[c] == top_class_count), key=str)
    top_class_freq = top_class_count / len(keep)
    print(f"  top class: '{top_class_name}' freq={top_class_freq:.4f} (={top_class_count}/{len(keep)})")

    # Load RKMeans + extract layer=2 residual
    print(f"\n[step 4] load RKMeans ckpt: {args.ckpt}")
    codebooks, hp = load_rkmeans_ckpt(args.ckpt)
    n_layers = len(codebooks)
    normalize = hp.get("normalize_residuals", False)
    print(f"  n_layers={n_layers}, normalize={normalize}")

    print(f"\n[step 5] compute residuals for layer {args.layer}")
    r = embedding.float().clone()
    for l, cb in enumerate(codebooks):
        if l > args.layer:
            break
        if normalize:
            r = torch.nn.functional.normalize(r, dim=-1)
        dist = torch.cdist(r.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        r = r - cb[idx]
    R_layer = r.detach().cpu().numpy()  # (N, D)
    gain = np.linalg.norm(R_layer, axis=1)  # (N,)  ← 1D 标量
    gain_keep = gain[keep]
    print(f"  gain: mean={gain.mean():.4f}, std={gain.std():.4f}, "
          f"min={gain.min():.4f}, max={gain.max():.4f}")

    # ================== v3 核心：用 1D gain 做分类 ==================
    print(f"\n[step 6] 1D gain -> 18-class classification")
    print(f"  baselines:")
    print(f"    均匀瞎猜      = {1.0/n_classes:.4f}")
    print(f"    top-class 占比 = {top_class_freq:.4f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score, accuracy_score

    X_1d = gain_keep.reshape(-1, 1).astype(np.float64)  # (N, 1)

    results = {}
    seeds = list(range(args.n_runs))

    # 方法 1: 1D LogReg
    print(f"\n  [1D LogReg]")
    accs, f1_macros, f1_weighteds = [], [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X_1d, y_keep, test_size=0.2, random_state=seed, stratify=y_keep)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs", n_jobs=-1, random_state=seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1_macros.append(f1_score(y_te, pred, average="macro", zero_division=0))
        f1_weighteds.append(f1_score(y_te, pred, average="weighted", zero_division=0))
    results["1d_logreg"] = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_weighted_mean": float(np.mean(f1_weighteds)),
    }
    print(f"    acc = {results['1d_logreg']['acc_mean']:.4f} ± {results['1d_logreg']['acc_std']:.4f}")

    # 方法 2: 朴素贝叶斯（对 1D 连续特征是经典 baseline）
    print(f"\n  [Gaussian NB]")
    accs, f1_macros, f1_weighteds = [], [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X_1d, y_keep, test_size=0.2, random_state=seed, stratify=y_keep)
        clf = GaussianNB()
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1_macros.append(f1_score(y_te, pred, average="macro", zero_division=0))
        f1_weighteds.append(f1_score(y_te, pred, average="weighted", zero_division=0))
    results["1d_gnb"] = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_weighted_mean": float(np.mean(f1_weighteds)),
    }
    print(f"    acc = {results['1d_gnb']['acc_mean']:.4f} ± {results['1d_gnb']['acc_std']:.4f}")

    # 方法 3: 决策树（捕捉非线性关系）
    print(f"\n  [Decision Tree]")
    accs, f1_macros, f1_weighteds = [], [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X_1d, y_keep, test_size=0.2, random_state=seed, stratify=y_keep)
        clf = DecisionTreeClassifier(max_depth=8, random_state=seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1_macros.append(f1_score(y_te, pred, average="macro", zero_division=0))
        f1_weighteds.append(f1_score(y_te, pred, average="weighted", zero_division=0))
    results["1d_tree"] = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_weighted_mean": float(np.mean(f1_weighteds)),
    }
    print(f"    acc = {results['1d_tree']['acc_mean']:.4f} ± {results['1d_tree']['acc_std']:.4f}")

    # 方法 4: KNN（局部估计）
    print(f"\n  [KNN k=5]")
    from sklearn.neighbors import KNeighborsClassifier
    accs, f1_macros, f1_weighteds = [], [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X_1d, y_keep, test_size=0.2, random_state=seed, stratify=y_keep)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1_macros.append(f1_score(y_te, pred, average="macro", zero_division=0))
        f1_weighteds.append(f1_score(y_te, pred, average="weighted", zero_division=0))
    results["1d_knn5"] = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_weighted_mean": float(np.mean(f1_weighteds)),
    }
    print(f"    acc = {results['1d_knn5']['acc_mean']:.4f} ± {results['1d_knn5']['acc_std']:.4f}")

    # ================== ANOVA F-test: gain 对类目的区分能力 ==================
    print(f"\n[step 7] ANOVA F-test: gain 对 18 个类目的区分能力")
    from scipy.stats import f_oneway
    groups = [gain_keep[y_keep == c] for c in sorted(valid_cats)]
    f_stat, p_value = f_oneway(*groups)
    # 效应量 eta squared = SS_between / SS_total
    grand_mean = gain_keep.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((gain_keep - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total
    print(f"  F = {f_stat:.4f}, p = {p_value:.2e}")
    print(f"  eta² = {eta_sq:.4f}（< 0.01 小效应，0.06 中效应，0.14 大效应）")

    # 加载 v1 对照
    v1_acc_mag = None
    v1_path = Path("/home/wlia0047/ar57/wenyu/GeneRec/task2_results/task2_gain_shape_report.json")
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        v1_acc_mag = v1["results"]["mag_only"]["acc_mean"]
        v1_acc_shape = v1["results"]["shape_only"]["acc_mean"]
        print(f"\n[v1 对照] mag_only (2048 维展开) acc = {v1_acc_mag:.4f}, shape_only = {v1_acc_shape:.4f}")

    # ================== 判断 ==================
    acc_best_1d = max(r["acc_mean"] for r in results.values())
    blind_guess = 1.0 / n_classes
    if acc_best_1d > top_class_freq:
        verdict = (
            f"**acc(1D gain) 最高 = {acc_best_1d:.4f} > top-class 占比 {top_class_freq:.4f}**"
        )
    elif acc_best_1d > 2 * blind_guess:
        verdict = (
            f"**acc(1D gain) 最高 = {acc_best_1d:.4f} > 2 × 均匀瞎猜 {2*blind_guess:.4f}**"
        )
    elif acc_best_1d > blind_guess:
        verdict = (
            f"**acc(1D gain) 最高 = {acc_best_1d:.4f} > 均匀瞎猜 {blind_guess:.4f} 但 < top-class {top_class_freq:.4f}**"
        )
    else:
        verdict = (
            f"**acc(1D gain) 最高 = {acc_best_1d:.4f} ≤ 均匀瞎猜 {blind_guess:.4f} → gain 真的没用**"
        )

    if eta_sq < 0.01:
        eta_judgment = f"eta² = {eta_sq:.4f} < 0.01 → **小效应 → gain 对类目几乎无解释力**"
    elif eta_sq < 0.06:
        eta_judgment = f"eta² = {eta_sq:.4f} ∈ [0.01, 0.06] → **小到中效应 → gain 对类目有微弱解释力**"
    elif eta_sq < 0.14:
        eta_judgment = f"eta² = {eta_sq:.4f} ∈ [0.06, 0.14] → **中效应 → gain 对类目有可检测的解释力**"
    else:
        eta_judgment = f"eta² = {eta_sq:.4f} ≥ 0.14 → **大效应 → gain 对类目有强解释力**"

    final_judgment = (
        f"基线: 均匀瞎猜 = {blind_guess:.4f}, top-class 占比 = {top_class_freq:.4f}\n"
        f"1D gain 分类:\n"
        + "\n".join(f"  {k}: acc = {v['acc_mean']:.4f} ± {v['acc_std']:.4f}" for k, v in results.items())
        + f"\n\n{verdict}"
        + f"\nANOVA: F = {f_stat:.4f}, p = {p_value:.2e}, {eta_judgment}"
    )

    report = {
        "version": "v3_1d_gain",
        "ckpt_path": str(args.ckpt),
        "embedding_path": args.embedding,
        "data_dir": args.data_dir,
        "layer": args.layer,
        "n_layers": n_layers,
        "normalize_residuals": normalize,
        "n_items_kept": len(keep),
        "n_classes": n_classes,
        "top_class": top_class_name,
        "top_class_freq": top_class_freq,
        "blind_guess_acc": blind_guess,
        "gain_stats": {"mean": float(gain.mean()), "std": float(gain.std()),
                       "min": float(gain.min()), "max": float(gain.max())},
        "results_1d": results,
        "anova": {
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "eta_squared": float(eta_sq),
        },
        "v1_comparison": {
            "v1_mag_only_acc_2048d": v1_acc_mag,
            "v1_shape_only_acc_2048d": v1_acc_shape,
        } if v1_acc_mag is not None else None,
        "judgment": final_judgment,
    }

    with open(out_dir / "task2_v3_1d_gain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[report] saved {out_dir}/task2_v3_1d_gain_report.json")
    print(f"\n=== Final Summary ===")
    print(final_judgment)


if __name__ == "__main__":
    main()