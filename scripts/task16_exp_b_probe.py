"""
Task #71 — Exp B: 三残差向量信息保留能力（brand probe）

复用 Task #71 Exp A 的三残差（用 --use_kmeans），对 r_E/r_H/r_S 分别训练
LogisticRegression 预测 brand 标签，对比 probe 准确率。

启动:
  python scripts/task6_exp_b_probe.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def project_to_poincare(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(norm) * (x / norm)


def project_to_sphere(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return x / norm


def poincare_log_map(q, r):
    q_sqnorm = (q ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    r_sqnorm = (r ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    inner = (q * r).sum(dim=-1, keepdim=True)
    num = (1 + 2 * inner + r_sqnorm) * q + (1 - q_sqnorm) * r
    denom = (1 + 2 * inner + q_sqnorm * r_sqnorm).clamp(min=1e-7)
    u = num / denom
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    diff_sq = ((r - q) ** 2).sum(dim=-1, keepdim=True)
    arg = (1 + 2 * diff_sq / ((1 - q_sqnorm) * (1 - r_sqnorm))).clamp(min=1 + 1e-7)
    d = torch.acosh(arg)
    lambda_q = 2 / (1 - q_sqnorm)
    return (d / (lambda_q * u_norm)) * u


def spherical_log_map(q, r):
    q_norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos_sim = (q * r).sum(dim=-1, keepdim=True) / (q_norm * r_norm)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    d = torch.arccos(cos_sim)
    inner = (q * r).sum(dim=-1, keepdim=True)
    proj = r - (inner / (q_norm ** 2)) * q
    proj_norm = proj.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (d / proj_norm) * proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--metadata_pt", default="products/task16/from_task62/item_metadata.pt")
    parser.add_argument("--out_json", default="products/task16/from_task71/exp_b_probe.json")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k_brands", type=int, default=20,
                        help="Use top K most frequent brands for probe (limit classes)")
    args = parser.parse_args()

    print("=" * 60)
    print("Task #71 — Exp B: 三残差 brand probe 信息保留对比")
    print("=" * 60)

    # 加载 embeddings
    print(f"\n[1/5] Loading embeddings: {args.embeddings_pt}")
    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    print(f"  shape: {tuple(embeddings.shape)}")

    # 加载 metadata + 提取 brand labels
    print(f"[2/5] Loading metadata: {args.metadata_pt}")
    metadata = torch.load(args.metadata_pt, map_location="cpu", weights_only=False)
    brands_raw = [item.get("brand", "") for item in metadata]
    print(f"  Total items: {len(brands_raw)}")
    print(f"  Unique brands: {len(set(brands_raw))}")

    # 取 top K 高频 brand 作为 ground truth
    from collections import Counter
    brand_counts = Counter(brands_raw)
    top_brands = set(b for b, _ in brand_counts.most_common(args.top_k_brands))
    print(f"  Top {args.top_k_brands} brands cover: "
          f"{sum(c for b, c in brand_counts.items() if b in top_brands)} items")

    # 过滤 + 编码
    mask = np.array([b in top_brands for b in brands_raw])
    print(f"  Items in top brands: {mask.sum()}/{len(brands_raw)}")

    le = LabelEncoder()
    brand_labels = le.fit_transform([b for b, m in zip(brands_raw, mask) if m])
    print(f"  Encoded brand labels: {len(le.classes_)} classes")
    print(f"  Classes: {le.classes_[:10]}...")

    # 过滤 embeddings 到 top brands
    embeddings_filt = embeddings[mask]
    print(f"  Filtered embeddings shape: {tuple(embeddings_filt.shape)}")

    # 训练 MiniBatchKMeans L1 码本
    print(f"\n[3/5] Training MiniBatchKMeans (n_clusters={args.n_clusters}, seed={args.seed})...")
    emb_np = embeddings_filt.numpy()
    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=1024,
        n_init=3,
        max_iter=100,
    )
    km.fit(emb_np)
    q_idx = km.predict(emb_np)
    c_E = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    q = c_E[torch.tensor(q_idx)]
    r_1 = embeddings_filt - q

    # 算三残差
    print(f"[4/5] Computing 3 residuals...")
    r_E = r_1
    r_H = poincare_log_map(project_to_poincare(q), project_to_poincare(r_1))
    r_S = spherical_log_map(project_to_sphere(q), project_to_sphere(r_1))

    # Train/test split (stratified)
    print(f"\n[5/5] Training brand probes...")
    X_E = r_E.numpy()
    X_H = r_H.numpy()
    X_S = r_S.numpy()
    y = brand_labels

    X_E_tr, X_E_te, y_tr, y_te = train_test_split(
        X_E, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    X_H_tr, X_H_te, _, _ = train_test_split(
        X_H, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    X_S_tr, X_S_te, _, _ = train_test_split(
        X_S, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    # 训练 LogisticRegression（multinomial）
    results = {}
    for name, X_tr, X_te in [("r_E", X_E_tr, X_E_te),
                              ("r_H", X_H_tr, X_H_te),
                              ("r_S", X_S_tr, X_S_te)]:
        print(f"  Training probe on {name}...")
        lr = LogisticRegression(
            max_iter=200, solver="lbfgs", n_jobs=-1, random_state=args.seed
        )
        lr.fit(X_tr, y_tr)
        train_acc = accuracy_score(y_tr, lr.predict(X_tr))
        test_acc = accuracy_score(y_te, lr.predict(X_te))
        results[name] = {
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
        }
        print(f"    {name}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # Baseline: random + majority
    majority_acc = max(np.bincount(y)) / len(y)
    random_acc = 1.0 / len(le.classes_)
    print(f"\n  Baseline majority: {majority_acc:.4f}, random: {random_acc:.4f}")

    # 决策
    print(f"\n{'='*60}")
    print("Decision (R2: 信息保留是否不同?)")
    print(f"{'='*60}")
    test_accs = {k: v["test_acc"] for k, v in results.items()}
    max_acc = max(test_accs.values())
    min_acc = min(test_accs.values())
    diff = max_acc - min_acc
    print(f"  Best probe: {max(test_accs, key=test_accs.get)} = {max_acc:.4f}")
    print(f"  Worst probe: {min(test_accs, key=test_accs.get)} = {min_acc:.4f}")
    print(f"  Diff (max - min): {diff:.4f}")

    if diff > 0.05:
        decision = "R2_CONFIRMED (某个流形保留更多信息 → 标记优先)"
    elif diff > 0.02:
        decision = "R2_MARGINAL (差异小但存在 → 继续 C/D)"
    else:
        decision = "R2_REJECTED (probe diff < 2% → 关闭 B 路线)"

    print(f"  Decision: {decision}")

    # 保存
    out = {
        "task": "#71 Exp B — brand probe",
        "n_items_used": int(mask.sum()),
        "n_classes": int(len(le.classes_)),
        "results": results,
        "baselines": {"majority": majority_acc, "random": random_acc},
        "diff": float(diff),
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()