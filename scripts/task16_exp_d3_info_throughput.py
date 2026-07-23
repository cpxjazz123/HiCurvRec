"""
Task #67 — Exp D3: 各层残差信息通过率分析

验证 RQ-VAE 各层残差是否真有明确的信息分工（通过率曲线），
还是仅数值递减。

通过率定义:
  throughput(L_i+1 → L_i, attr) = acc(z_≤i+1, attr) / acc(z_≤i, attr)
  raw → L1: acc(z_≤1, attr) / acc(x_raw, attr)

启动:
  python scripts/task6_exp_d3_info_throughput.py \
      --stage1_pt <path/to/merged_predictions_tensor.pt> \
      --rqvae_ckpt products/task16/from_task62/rqvae_ckpt.pt \
      --metadata_pt products/task16/from_task62/item_metadata.pt \
      --out_json products/task16/from_task67/exp_d3_info_throughput.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============== 线性 probe ==============

def train_probe(X, y, test_size=0.2, random_state=42):
    """训练 logistic regression probe，返回 test acc"""
    # 过滤罕见类别（< 5 样本）
    unique, counts = np.unique(y, return_counts=True)
    valid = unique[counts >= 5]
    mask = np.isin(y, valid)
    X, y = X[mask], y[mask]

    if len(np.unique(y)) < 2:
        return 0.0

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=1, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def random_baseline(y):
    """随机基线 = max class 占比"""
    unique, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


# ============== 累积量化 ==============

def cumulative_quantize(embeddings, model):
    """对每个输入 embedding 计算 z_≤1, z_≤2, z_≤3 累积量化表示

    z_≤l = sum of codewords for layers 0..l-1
    返回: dict with 'raw', 'z_<=1', 'z_<=2', 'z_<=3'
    """
    codebooks = [q.codebook for q in model.quantizers]
    n_layers = len(codebooks)

    # raw
    reps = {"raw": embeddings.numpy()}

    # 累积量化
    residual = embeddings
    cumulative = torch.zeros_like(embeddings)
    with torch.no_grad():
        for i, cb in enumerate(codebooks):
            dist = torch.cdist(residual, cb, p=2)
            indices = dist.argmin(dim=1)
            q = cb[indices]
            cumulative = cumulative + q
            residual = residual - q
            reps[f"z_le_{i+1}"] = cumulative.numpy()

    return reps


# ============== 主实验 ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_pt", required=True)
    parser.add_argument("--rqvae_ckpt", required=True)
    parser.add_argument("--metadata_pt", required=True,
                        help="包含 brand/category 的 metadata tensor")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_items", type=int, default=11924)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Task #67 — Exp D3: info throughput")
    print("=" * 60)

    # ============== 加载 RQ-VAE ==============
    print(f"\n[1/5] Loading RQ-VAE: {args.rqvae_ckpt}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from task62_train_rqvae import ResidualQuantization

    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu", weights_only=False)
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ============== 加载 embeddings ==============
    print(f"\n[2/5] Loading embeddings: {args.stage1_pt}")
    embeddings = torch.load(args.stage1_pt, map_location="cpu", weights_only=False)
    if embeddings.dim() > 2:
        embeddings = embeddings.squeeze()
    print(f"  raw shape: {tuple(embeddings.shape)}")
    if embeddings.shape[0] > args.max_items:
        idx = torch.randperm(embeddings.shape[0])[:args.max_items]
        embeddings = embeddings[idx]
        print(f"  sampled to: {tuple(embeddings.shape)}")

    # ============== 加载 metadata ==============
    print(f"\n[3/5] Loading metadata: {args.metadata_pt}")
    metadata = torch.load(args.metadata_pt, map_location="cpu", weights_only=False)

    # 处理多种 metadata 格式
    if isinstance(metadata, list):
        # Task 62 格式: list[dict{title, brand, categories, ...}]
        print(f"  list of dicts, len={len(metadata)}")
        brands = np.array([
            m.get("brand", "Unknown") if isinstance(m, dict) else "Unknown"
            for m in metadata
        ])
        categories = np.array([
            m["categories"][0] if isinstance(m, dict) and m.get("categories")
            else "Unknown"
            for m in metadata
        ])
    elif isinstance(metadata, dict):
        # dict 格式: {'brand': [...], 'category': [...]}
        print(f"  dict keys: {list(metadata.keys())[:5]}")
        brands = np.array(metadata.get("brand", metadata.get("brands", [])))
        categories = np.array(metadata.get("category", metadata.get("categories", [])))
    else:
        # tensor 格式: 假设后两列
        print(f"  tensor shape: {tuple(metadata.shape)}")
        categories = metadata[:, 0].numpy()
        brands = metadata[:, 1].numpy()

    # 对齐样本数
    n = embeddings.shape[0]
    if len(brands) > n:
        brands = brands[:n]
    if len(categories) > n:
        categories = categories[:n]
    if len(brands) < n:
        brands = np.concatenate([brands, ["Unknown"] * (n - len(brands))])
    if len(categories) < n:
        categories = np.concatenate([categories, ["Unknown"] * (n - len(categories))])

    print(f"  brands: {len(np.unique(brands))} unique, "
          f"random baseline={random_baseline(brands):.4f}")
    print(f"  categories: {len(np.unique(categories))} unique, "
          f"random baseline={random_baseline(categories):.4f}")

    # ============== 累积量化 ==============
    print(f"\n[4/5] Cumulative quantization for {model.n_layers} layers...")
    reps = cumulative_quantize(embeddings, model)
    for k, v in reps.items():
        print(f"  {k}: {v.shape}")

    # ============== 训练 probe + 计算通过率 ==============
    print(f"\n[5/5] Training linear probes for each (layer × attribute)...")
    attrs = {
        "brand": brands,
        "category": categories,
    }
    layers = ["raw", "z_le_1", "z_le_2", "z_le_3"]

    acc_table = {}  # acc_table[attr][layer] = acc
    for attr_name, y in attrs.items():
        acc_table[attr_name] = {}
        baseline = random_baseline(y)
        print(f"\n  --- {attr_name} (baseline={baseline:.4f}) ---")
        for layer_name in layers:
            X = reps[layer_name]
            t0 = time.time()
            acc = train_probe(X, y, random_state=args.seed)
            dt = time.time() - t0
            acc_table[attr_name][layer_name] = acc
            print(f"    {layer_name:>8s}: acc={acc:.4f}  ({dt:.1f}s)")

    # ============== 通过率矩阵 ==============
    print("\n" + "=" * 60)
    print("Information Throughput Matrix")
    print("=" * 60)

    throughput_table = {}
    layer_pairs = [
        ("raw", "z_le_1", "raw→L1"),
        ("z_le_1", "z_le_2", "L1→L2"),
        ("z_le_2", "z_le_3", "L2→L3"),
    ]

    for attr_name in attrs.keys():
        throughput_table[attr_name] = {}
        print(f"\n  [{attr_name}]")
        print(f"    {'Transition':<12} {'From→To':<20} {'Throughput':<12}")
        for src, dst, label in layer_pairs:
            from_acc = acc_table[attr_name][src]
            to_acc = acc_table[attr_name][dst]
            if from_acc > 1e-6:
                throughput = to_acc / from_acc
            else:
                throughput = 0.0
            throughput_table[attr_name][label] = throughput
            print(f"    {label:<12} "
                  f"{from_acc:.4f} → {to_acc:.4f}     {throughput:.3f}")

    # ============== 分工度评估 ==============
    print("\n" + "=" * 60)
    print("Layer Specialization Assessment")
    print("=" * 60)

    # 对每层，看哪个属性 throughput 最高
    layer_specialist = {}
    for src, dst, label in layer_pairs:
        # 该层 transition 中各属性 throughput
        ths = {attr: throughput_table[attr][label] for attr in attrs.keys()}
        specialist = max(ths, key=ths.get)
        layer_specialist[label] = {
            "best_attr": specialist,
            "best_throughput": ths[specialist],
            "all_throughputs": ths,
        }
        print(f"  {label}: specialist = {specialist} "
              f"(throughput={ths[specialist]:.3f})")

    # 决策
    max_th = max(
        throughput_table[attr][lbl]
        for attr in attrs.keys()
        for lbl in ["raw→L1", "L1→L2", "L2→L3"]
    )
    decision_special = "DIVIDED" if max_th > 0.80 else "UNIFORM"
    print(f"\n  >>> Max throughput across all attrs/layers: {max_th:.3f}")
    print(f"  >>> Threshold: > 0.80 → DIVIDED, ∈ [0.30, 0.60] → UNIFORM")
    print(f"  >>> DECISION: {decision_special}")

    # ============== 保存 ==============
    out = {
        "task": "#67 Exp D3",
        "n_items": int(embeddings.shape[0]),
        "n_layers_rqvae": int(model.n_layers),
        "attrs": list(attrs.keys()),
        "random_baseline": {
            attr_name: random_baseline(y)
            for attr_name, y in attrs.items()
        },
        "acc_table": acc_table,
        "throughput_table": throughput_table,
        "layer_specialist": layer_specialist,
        "max_throughput": max_th,
        "decision": decision_special,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()