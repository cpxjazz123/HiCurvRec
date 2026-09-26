"""Compute FORGE proxy metrics on the *current iteration's* SID and compare against the TIGER baseline.

与 /tmp/sid_metrics_any.py 的区别:
  - /tmp/sid_metrics_any.py 接收 SID 路径 + 标签位置参数 (Project Rules §1 例外),
    适合在 shell 里直接调用做 gate check;
  - 本脚本走 Project Rules §1 主流 (0 argparse / 0 CLI flag), 顶部常量硬编码,
    对当前 iter 的 SID 与 TIGER baseline 同时算 4 个 FORGE 指标并打印对比表,
    结果可重定向到日志形成审计。

调用:
  python3 scripts/eval_sids.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from collections import Counter

# === 硬编码输入路径 (Project Rules §1: 0 argparse / 0 CLI flag) ===
ITER_ID  = os.environ.get("ITER_ID", "iter1")        # 由 train_iter.sh 注入
ITER_DIR = os.environ.get("ITER_DIR", "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter1")
ITER_SID = f"{ITER_DIR}/dataset/Instruments/sids_for_hgrec.npy"

TIGER_SID = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/sids_for_hgrec_recbole.npy"
ITEM_EMB  = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"
INTER_JS  = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments/Instruments.inter.json"

KS = [50]


def _load_sid(path):
    sids = np.load(path)
    if sids.ndim == 2 and sids.shape[1] >= 3:
        return sids[:, :3]
    return sids


def _gini(counts):
    counts = np.sort(np.array(counts, dtype=np.float64))
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    cum = np.cumsum(counts)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def _metrics(sids):
    tuples = [tuple(s[:3]) for s in sids]
    c = Counter(tuples)
    counts = list(c.values())
    n_unique = len(counts)
    n_items  = len(tuples)
    return {
        "n_items":       n_items,
        "n_unique":      n_unique,
        "collision_rate": (n_items - n_unique) / n_items,
        "full_3token_gini": _gini(counts),
        "per_layer_mean_gini": float(np.mean([
            _gini(np.bincount(sids[:, i], minlength=256).tolist()) for i in range(3)
        ])),
    }


def _hitrate(emb_n, inter):
    user_items = {u: [int(i) for i in items] for u, items in inter.items() if len(items) >= 5}
    sample_users = list(user_items.keys())[:500]
    pairs = [(u, t) for u in sample_users for t in user_items[u][int(len(user_items[u]) * 0.8):]]
    hit = {k: 0 for k in KS}
    total = 0
    for u, t in pairs:
        if t >= emb_n.shape[0]:
            continue
        h = set(user_items[u][:int(len(user_items[u]) * 0.8)])
        if not h:
            continue
        sim = emb_n @ emb_n[t]
        sim[t] = -1
        top = np.argpartition(-sim, max(KS))[:max(KS)]
        top = top[np.argsort(-sim[top])]
        for k in KS:
            if len(set(top[:k].tolist()) & h) > 0:
                hit[k] += 1
        total += 1
    return {k: hit[k] / total for k in KS}


def main():
    df = pd.read_parquet(ITEM_EMB) if ITEM_EMB.endswith(".parquet") else None
    if df is not None and "embedding" in df.columns:
        embs = np.stack(df["embedding"].values).astype(np.float32)
    else:
        embs = np.load(ITEM_EMB.replace(".parquet", ".npy")).astype(np.float32) \
               if not os.path.exists(ITEM_EMB) and os.path.exists(ITEM_EMB.replace(".parquet", ".npy")) \
               else np.load(ITEM_EMB, allow_pickle=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    emb_n = embs / (norms + 1e-8)

    with open(INTER_JS) as f:
        inter = json.load(f)
    hr = _hitrate(emb_n, inter)

    iter_metrics  = _metrics(_load_sid(ITER_SID))
    tiger_metrics = _metrics(_load_sid(TIGER_SID))

    print(f"=== {ITER_ID} vs TIGER (FORGE 4 指标) ===")
    print(f"  HitRate@50                 : iter={hr[50]:.4f}")
    print(f"  full_3token_gini           : iter={iter_metrics['full_3token_gini']:.4f} "
          f"| tiger={tiger_metrics['full_3token_gini']:.4f}")
    print(f"  per_layer_mean_gini        : iter={iter_metrics['per_layer_mean_gini']:.4f} "
          f"| tiger={tiger_metrics['per_layer_mean_gini']:.4f}")
    print(f"  collision_rate             : iter={iter_metrics['collision_rate']:.4%} "
          f"| tiger={tiger_metrics['collision_rate']:.4%}")
    print(f"  iter n_unique={iter_metrics['n_unique']} n_items={iter_metrics['n_items']}")
    print(f"  tiger n_unique={tiger_metrics['n_unique']} n_items={tiger_metrics['n_items']}")


if __name__ == "__main__":
    main()