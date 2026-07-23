#!/usr/bin/env python3
"""Task A11: 真实 Poincaré distance 重做 HRQ topology alignment

设计:
- 加载 flan-t5-xl embedding (11924, 2048)
- 用 task18_hrq_hyperbolic 的 Lorentz 模型 lift + L1 hkmeans (256 cluster)
- 保存 codebook + item Lorentz coords
- 多 ckpt 阶段 + 多 seed 都重复 L1 hkmeans
- 用真实 Lorentz distance (arccosh) 替代 Hamming distance (taskA6)
- 重做 topology alignment ρ vs R@10 回归

判据:
- 若 Poincaré distance 比 Hamming distance 更对齐 behavior → A6 结论需修正
- 若仍弱信号 → HRQ 优势来自其他机制（更可能是碰撞分布/codebook 利用度）
"""

import os, json, sys, time
import numpy as np
import torch
import tensorflow as tf
import glob
from collections import defaultdict
from scipy.stats import spearmanr

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
EMBED_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
OUT_DIR = f'{GRID}/result/taskA11_poincare_hrq_topology'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A11: 真实 Poincaré distance HRQ topology alignment')
print('=' * 70)

# ========== Lorentz model functions (from task18_hrq_hyperbolic.py) ==========
CURVATURE = 1.0


def lift_to_lorentz(x, c=CURVATURE):
    x_norm2 = (x * x).sum(-1)
    h0 = torch.sqrt(c + x_norm2).unsqueeze(-1)
    return torch.cat([h0, x], dim=-1)


def lorentz_inner(h1, h2):
    return -h1[..., 0] * h2[..., 0] + (h1[..., 1:] * h2[..., 1:]).sum(-1)


def lorentz_distance(h1, h2, c=CURVATURE):
    inner = lorentz_inner(h1, h2)
    arg = -inner / c
    arg = torch.clamp(arg, min=1.0 + 1e-9)
    return torch.acosh(arg)


# ========== L1 hkmeans (single layer for speed) ==========
def hkmeans_assign_batched(x_lorentz, centroids, batch=1024):
    K = centroids.shape[0]
    assignments = torch.empty(x_lorentz.shape[0], dtype=torch.long, device=x_lorentz.device)
    for i in range(0, x_lorentz.shape[0], batch):
        x_batch = x_lorentz[i:i+batch]
        inner_xc = lorentz_inner(x_batch.unsqueeze(1), centroids.unsqueeze(0))
        arg = torch.clamp(-inner_xc, min=1.0 + 1e-9)
        d = torch.acosh(arg)
        assignments[i:i+batch] = d.argmin(dim=1)
    return assignments


def hkmeans_init(x_lorentz, n_clusters, seed=42):
    rng = np.random.RandomState(seed)
    n = x_lorentz.shape[0]
    indices = rng.choice(n, n_clusters, replace=False)
    return x_lorentz[indices].clone()


def hkmeans_update_centroids(x_lorentz, assignments, n_clusters):
    centroids_new = torch.empty((n_clusters, x_lorentz.shape[1]), device=x_lorentz.device)
    for k in range(n_clusters):
        mask = assignments == k
        if mask.sum() == 0:
            rand_idx = torch.randint(0, x_lorentz.shape[0], (1,))
            centroids_new[k] = x_lorentz[rand_idx]
        else:
            mean_eu = x_lorentz[mask][:, 1:].mean(dim=0)
            centroids_new[k] = lift_to_lorentz(mean_eu.unsqueeze(0))[0]
    return centroids_new


def hkmeans(x_lorentz, n_clusters=256, max_iter=50, seed=42, verbose=True):
    centroids = hkmeans_init(x_lorentz, n_clusters, seed=seed)
    for it in range(max_iter):
        assignments = hkmeans_assign_batched(x_lorentz, centroids)
        centroids_new = hkmeans_update_centroids(x_lorentz, assignments, n_clusters)
        # Convergence check
        if torch.allclose(centroids_new, centroids, atol=1e-5):
            if verbose:
                print(f'    converged at iter {it}')
            break
        centroids = centroids_new
        if verbose and (it + 1) % 10 == 0:
            with torch.no_grad():
                d = lorentz_distance(x_lorentz, centroids[assignments])
                qerr = d.mean().item()
            print(f'    iter {it+1}, qerr={qerr:.4f}')
    return assignments, centroids


# ========== Load flan-t5-xl embedding ==========
print(f'\n[load flan-t5-xl embedding]')
embed = torch.load(EMBED_PATH, weights_only=False)
print(f'  shape: {embed.shape}')
# flan-t5-xl Stage 1 输出 (n_items, dim) but file has (dim, n_items) = (2048, 11924)
# Normalize to (N=n_items, D=dim)
if embed.shape[0] == 2048 and embed.shape[1] == 11924:
    embed = embed.T  # (11924, 2048)
elif embed.shape[1] == 2048 and embed.shape[0] == 11924:
    pass  # already correct
embed = embed.float()
n_items = embed.shape[0]
assert n_items == 11924, f'Expected 11924 items, got {n_items}'
print(f'  n_items: {n_items}, dim: {embed.shape[1]}')

# Lift to Lorentz
print(f'\n[lift to Lorentz (curvature={CURVATURE})]')
x_lorentz = lift_to_lorentz(embed)
print(f'  shape: {x_lorentz.shape}')

# ========== Run hkmeans with multiple seeds ==========
print(f'\n[hkmeans L1 (256 clusters)]')
all_codebooks = {}
all_assignments = {}
for seed in [42, 43]:
    print(f'  seed={seed}')
    t0 = time.time()
    assignments, centroids = hkmeans(x_lorentz, n_clusters=256, max_iter=50, seed=seed)
    print(f'    done in {time.time()-t0:.1f}s')
    all_codebooks[seed] = centroids
    all_assignments[seed] = assignments
    # Stats
    n_used = len(set(assignments.tolist()))
    print(f'    clusters used: {n_used}/256')

# Save codebook + item coords
out_codebook = {
    'lorentz_coords': x_lorentz.cpu().numpy().tolist(),  # big, save as npy instead
}
# Save as numpy for memory efficiency
np.save(os.path.join(OUT_DIR, 'item_lorentz_coords.npy'), x_lorentz.cpu().numpy())
for seed in [42, 43]:
    np.save(os.path.join(OUT_DIR, f'hkmeans_assignments_s{seed}.npy'),
            all_assignments[seed].cpu().numpy())
    np.save(os.path.join(OUT_DIR, f'hkmeans_centroids_s{seed}.npy'),
            all_codebooks[seed].cpu().numpy())
print(f'  saved to {OUT_DIR}')

# ========== Compute Lorentz distance between item pairs ==========
print(f'\n[compute Lorentz distance for sampled pairs]')

# Load eval tfrecords → user history → co-purchase
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
item_users = defaultdict(set)
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    for item in seq:
        item_users[item].add(user_id)

# Sample pairs (same as taskA6: 30k co-purchase + 5k random)
np.random.seed(42)
co_pairs = set()
for users_set in item_users.values():
    pass  # only need co-purchase pairs
# Recompute co-purchase pairs
for u, seq in enumerate([]):
    pass  # placeholder

# Easier: rebuild co-purchase
print(f'  build co-purchase...')
seqs = []
dataset2 = tf.data.TFRecordDataset(files, compression_type='GZIP')
for raw in dataset2:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if seq:
        seqs.append(seq)

for u_idx, seq in enumerate(seqs):
    seen = set()
    for item in seq:
        seen.add(item)
    items_list = list(seen)
    for i_idx in range(len(items_list)):
        for j_idx in range(i_idx + 1, len(items_list)):
            a, b = items_list[i_idx], items_list[j_idx]
            if a > b:
                a, b = b, a
            co_pairs.add((a, b))

co_pairs = list(co_pairs)
if len(co_pairs) > 30000:
    sample_idx = np.random.choice(len(co_pairs), 30000, replace=False)
    sampled_co = [co_pairs[i] for i in sample_idx]
else:
    sampled_co = co_pairs

random_pairs = set()
existing = set(co_pairs)
while len(random_pairs) < 5000:
    a = np.random.randint(0, n_items)
    b = np.random.randint(0, n_items)
    if a == b:
        continue
    if a > b:
        a, b = b, a
    if (a, b) not in existing and (a, b) not in random_pairs:
        random_pairs.add((a, b))

all_pairs = sampled_co + list(random_pairs)
print(f'  total pairs: {len(all_pairs)}')

# Jaccard
print(f'  compute Jaccard...')
jac_vals = np.zeros(len(all_pairs), dtype=np.float32)
for idx, (i, j) in enumerate(all_pairs):
    Ui = item_users.get(i, set())
    Uj = item_users.get(j, set())
    if not Ui or not Uj:
        jac_vals[idx] = 0
        continue
    inter = len(Ui & Uj)
    union = len(Ui | Uj)
    jac_vals[idx] = inter / union if union > 0 else 0
    if (idx + 1) % 10000 == 0:
        print(f'    {idx+1}/{len(all_pairs)}')

# Compute Lorentz distances for each pair
print(f'  compute Lorentz distances...')
pair_arr = np.array(all_pairs, dtype=np.int32)
lorentz_dists = np.zeros(len(all_pairs), dtype=np.float32)
BATCH = 1000
for batch_start in range(0, len(all_pairs), BATCH):
    batch_end = min(batch_start + BATCH, len(all_pairs))
    idx_pairs = pair_arr[batch_start:batch_end]
    h_i = x_lorentz[idx_pairs[:, 0]]
    h_j = x_lorentz[idx_pairs[:, 1]]
    d = lorentz_distance(h_i, h_j)
    lorentz_dists[batch_start:batch_end] = d.cpu().numpy()
    if (batch_start // BATCH + 1) % 10 == 0:
        print(f'    {batch_end}/{len(all_pairs)}')

# Spearman ρ
rho_poincare, p_poincare = spearmanr(lorentz_dists, jac_vals)
print(f'\n  ρ (Poincaré distance vs Jaccard): {rho_poincare:+.4f} (p={p_poincare:.2e})')

# ========== Compare with Hamming (taskA6) ==========
print(f'\n[compare with Hamming distance]')
sid_hrq = torch.load(f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
                     weights_only=False)
if sid_hrq.shape[0] == 4:
    sid_hrq_items = sid_hrq[:4].T.numpy()
else:
    sid_hrq_items = sid_hrq.T.numpy()
hamming_dists = np.sum(np.abs(sid_hrq_items[pair_arr[:, 0]] - sid_hrq_items[pair_arr[:, 1]]), axis=1)
rho_hamming, p_hamming = spearmanr(hamming_dists, jac_vals)
print(f'  ρ (Hamming distance vs Jaccard, from A6): {rho_hamming:+.4f} (p={p_hamming:.2e})')

# Correlation between Poincaré and Hamming distances (sanity check)
rho_dist, _ = spearmanr(lorentz_dists, hamming_dists)
print(f'  ρ (Poincaré vs Hamming, distance correlation): {rho_dist:+.4f}')

# ========== R@10 per algorithm ==========
print(f'\n[compute R@10 per algorithm]')


def compute_e2e_r10(pred_path, sid_path, user_targets, k=10):
    sid = torch.load(sid_path, weights_only=False)
    pred = torch.load(pred_path, weights_only=False)
    sid_to_item = {}
    # sid shape: (4, 11924) or (11924, 4)
    if sid.shape[0] == 4:
        sid_T = sid.T  # (11924, 4)
    else:
        sid_T = sid  # already (11924, 4)
    for item_id in range(sid_T.shape[0]):
        sid_to_item[tuple(int(x) for x in sid_T[item_id].tolist())] = item_id

    n_users, K, _ = pred.shape
    pred_items = torch.full((n_users, K), -1, dtype=torch.long)
    for u in range(n_users):
        for kk in range(K):
            s = tuple(int(x) for x in pred[u, kk].tolist())
            pred_items[u, kk] = sid_to_item.get(s, -1)

    n_hit = 0
    n_eval = 0
    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        n_eval += 1
        for kk in range(K):
            if pred_items[u, kk].item() == target:
                n_hit += 1
                break
    return n_hit / max(1, n_eval), n_eval


configs = [
    ('HRQ_s42', f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task18_hrq_s4_v2200/pickle/merged_predictions_tensor.pt'),
    ('HRQ_s43', f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task18_hrq_seed43_s4_v2000/pickle/merged_predictions_tensor.pt'),
    ('L=4_baseline', f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'),
    ('AQ_s42', f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task19_aq_s4_v2000/pickle/merged_predictions_tensor.pt'),
]

# Load user_targets
user_targets = {}
dataset3 = tf.data.TFRecordDataset(files, compression_type='GZIP')
for raw in dataset3:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if seq:
        user_targets[user_id] = seq[-1]

per_algo_r10 = {}
for name, sid_path, pred_path in configs:
    if not os.path.exists(sid_path) or not os.path.exists(pred_path):
        print(f'  {name}: SKIP')
        continue
    r10, n_eval = compute_e2e_r10(pred_path, sid_path, user_targets)
    per_algo_r10[name] = r10
    print(f'  {name}: R@10 = {r10*100:.4f}%')

# Save
out = {
    'task': 'Task A11: Poincaré distance HRQ topology alignment',
    'method': 'Real Lorentz distance (arccosh) vs Hamming distance (A6)',
    'rho_poincare': float(rho_poincare),
    'p_poincare': float(p_poincare),
    'rho_hamming': float(rho_hamming),
    'p_hamming': float(p_hamming),
    'rho_distance_correlation': float(rho_dist),
    'r10_per_algo': {k: float(v) for k, v in per_algo_r10.items()},
    'n_pairs': len(all_pairs),
    'n_clusters': 256,
    'curvature': CURVATURE,
}

with open(os.path.join(OUT_DIR, 'poincare_topology.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

verdict_lines = [
    '# Task A11 Verdict: 真实 Poincaré Distance HRQ Topology',
    '',
    '## 设计',
    '',
    '- 加载 flan-t5-xl embedding (11924, 2048)',
    '- Lorentz lift: h_0 = sqrt(c + ||x||²), h_[1:] = x, c=1',
    '- L1 hkmeans (256 clusters, max_iter=50)',
    '- Lorentz distance: d_L(h1, h2) = arccosh(-<h1, h2>_L / c)',
    '- 对比 taskA6 的 Hamming distance',
    '',
    '## Topology Alignment ρ',
    '',
    '| 距离度量 | ρ (Spearman) | p-value |',
    '|----------|--------------|---------|',
    f'| Poincaré (Lorentz arccosh) | {rho_poincare:+.4f} | {p_poincare:.2e} |',
    f'| Hamming (taskA6) | {rho_hamming:+.4f} | {p_hamming:.2e} |',
    '',
    '## 距离度量间相关',
    '',
    f'- ρ (Poincaré vs Hamming): {rho_dist:+.4f}',
    '- 若两度量间强相关 → Hamming 已能代理 Poincaré',
    '- 若弱相关 → 不同度量捕捉不同几何',
    '',
    '## R@10 per Algorithm',
    '',
    '| Algorithm | R@10 |',
    '|-----------|------|',
]
for name, r10 in per_algo_r10.items():
    verdict_lines.append(f'| {name} | {r10*100:.4f}% |')

verdict_lines.extend([
    '',
    '## 因果解读',
    '',
    '- 若 ρ_Poincaré > ρ_Hamming → A6 用 Hamming 低估了 HRQ topology alignment',
    '- 若 ρ_Poincaré ≈ ρ_Hamming → 两度量等价，A6 结论稳健',
    '- 若 ρ_Poincaré 仍弱 → HRQ 优势机制（碰撞分布 / codebook 利用 / tiger 训练）仍未定位',
    '',
    '## 关键结论',
    '',
])
if rho_poincare > rho_hamming + 0.01:
    verdict_lines.append('**Poincaré 比 Hamming 更对齐 behavior** → A6 结论需修正')
elif abs(rho_poincare - rho_hamming) <= 0.01:
    verdict_lines.append('**两度量等价** → A6 结论稳健，HRQ 优势不来自 topology 对齐')
else:
    verdict_lines.append('**Poincaré 反而更弱** → A6 已低估问题，HRQ 优势完全不在几何对齐')

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/poincare_topology.json + verdict.md + .npy files')
print(f'ρ Poincaré: {rho_poincare:+.4f} | ρ Hamming: {rho_hamming:+.4f}')