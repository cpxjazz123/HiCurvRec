import os, sys, json
import numpy as np
import pandas as pd
from collections import Counter

EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/item_emb.parquet"
SIDS = {
    "TIGER":      "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/sids_for_hgrec_recbole.npy",
    "HG":         "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/HG_RQ-VAE/sids_for_hgrec_v2_extended.npy",
    "curvature":  "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/sids_for_hgrec_recbole.npy",
    "LETTER":     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/LETTER_RQ-VAE/sids_for_hgrec_extended.npy",
}
INTER_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments/Instruments.inter.json"
KS = [10, 500]

df = pd.read_parquet(EMB_PATH)
if "embedding" in df.columns:
    embs = np.stack(df["embedding"].values).astype(np.float32)
else:
    embs = df.values.astype(np.float32)
N, D = embs.shape
print(f"[emb] shape={embs.shape}, dtype={embs.dtype}")

norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs_n = embs / (norms + 1e-8)
print(f"[emb] L2-normalized: mean norm = {norms.mean():.3f}, std = {norms.std():.3f}")

with open(INTER_JSON) as f:
    user_items_raw = json.load(f)
# dict: user_id -> [item_id, ...]
user_items = {u: [int(i) for i in items] for u, items in user_items_raw.items() if len(items) >= 5}

user_history = {}
user_target = {}
for u, items in user_items.items():
    if len(items) < 5: continue
    n = len(items)
    split = int(n * 0.8)
    user_history[u] = set(items[:split])
    user_target[u] = items[split:]
print(f"[data] {len(user_history)} users with >=5 interactions, mean target items = {np.mean([len(user_target[u]) for u in user_history]):.1f}")

print("\n=== Embedding HitRate (FORGE 指标 1) ===")
sample_users = list(user_history.keys())[:500]
all_targets = []
for u in sample_users:
    for t in user_target[u]:
        all_targets.append((u, t))
print(f"[hitrate] eval {len(all_targets)} (user, target) pairs")

hit_at_k = {k: 0 for k in KS}
total = 0
for u, t in all_targets:
    if t >= N: continue
    if u not in user_history: continue
    h = user_history[u]
    if len(h) == 0: continue
    sim = embs_n @ embs_n[t]
    sim[t] = -1
    top_idx = np.argpartition(-sim, max(KS))[:max(KS)]
    top_idx = top_idx[np.argsort(-sim[top_idx])]
    for k in KS:
        top_k_idx = set(top_idx[:k].tolist())
        if len(top_k_idx & h) > 0:
            hit_at_k[k] += 1
    total += 1

for k in KS:
    print(f"  HitRate@{k} = {hit_at_k[k]/total:.4f}")

print("\n=== 完整 SID tuple 的 Gini (FORGE 指标 2) ===")
def gini_coefficient(counts):
    counts = np.sort(np.array(counts, dtype=np.float64))
    n = len(counts)
    if n == 0 or counts.sum() == 0: return 0.0
    cum = np.cumsum(counts)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n

results = {}
for name, path in SIDS.items():
    sids = np.load(path)
    if sids.ndim == 2 and sids.shape[1] >= 3:
        tuples = [tuple(s[:3]) for s in sids]
    else:
        tuples = [tuple(s) for s in sids]
    c = Counter(tuples)
    counts = list(c.values())
    n_unique = len(counts)
    n_items = len(tuples)
    n_collision = n_items - n_unique
    collision_rate = n_collision / n_items
    g = gini_coefficient(counts)
    theoretical_max = 256 ** 3
    util_rate = n_unique / min(theoretical_max, n_items)
    results[name] = (collision_rate, g)
    print(f"\n  [{name}] {path.split('/')[-1]}")
    print(f"    shape={sids.shape}, unique SID={n_unique}, items={n_items}")
    print(f"    collision_rate = {collision_rate:.4%} ({n_collision} collisions)")
    print(f"    SID occupancy Gini = {g:.4f}")
    print(f"    unique/full ratio = {n_unique}/{min(theoretical_max, n_items)} = {util_rate:.4%}")

print("\n=== 总结 (FORGE 2 个核心指标) ===")
print(f"{'Method':<12} {'CollRate':>10} {'Gini':>8} {'HR@10':>8} {'HR@500':>8}")
for name in SIDS.keys():
    cr, g = results[name]
    print(f"{name:<12} {cr:>9.4%} {g:>8.4f} {hit_at_k[10]/total:>8.4f} {hit_at_k[500]/total:>8.4f}")
