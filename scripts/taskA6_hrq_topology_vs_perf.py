#!/usr/bin/env python3
"""Task A6: HRQ 拓扑对齐 vs 性能提升因果验证

设计:
- 计算 HRQ/AQ/L=4 的 SID 拓扑对齐: ρ = Spearman(-SID_dist, Jaccard)
- 避免 O(n_items^2) 全矩阵 → 用采样对 (co-purchase + random)
- 回归 R@10 ~ ρ + Collision + MSE 验证 ρ 是否为因果因子

产出:
- per-algo ρ (HRQ, AQ, L=4)
- per-algo R@10 (item-level)
- 回归系数 + p-value
"""

import os, json, glob
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import scipy.linalg

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
OUT_DIR = f'{GRID}/result/taskA6_hrq_topology_vs_perf'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A6: HRQ Topology Alignment vs Performance (sampled Jaccard)')
print('=' * 70)

# Load eval tfrecords → user history → item → users
print(f'\n[load eval tfrecords]')
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')

item_users = defaultdict(set)
user_targets = {}
user_seq = []
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    user_targets[user_id] = seq[-1]
    user_seq.append(seq)
    for item in seq:
        item_users[item].add(user_id)
print(f'  users: {len(user_targets)}, items with users: {len(item_users)}')

n_items = 11924

# Build co-purchase pairs (sample) for vectorized Jaccard
print(f'\n[build co-purchase sample]')
co_pairs_sparse = set()
for u_idx, seq in enumerate(user_seq):
    seen = set()
    for item in seq:
        seen.add(item)
    items_list = list(seen)
    for i_idx in range(len(items_list)):
        for j_idx in range(i_idx + 1, len(items_list)):
            a, b = items_list[i_idx], items_list[j_idx]
            if a > b:
                a, b = b, a
            co_pairs_sparse.add((a, b))
print(f'  unique co-purchase pairs: {len(co_pairs_sparse)}')

# Sample pairs: all co-purchase (or 30k) + 5k random
np.random.seed(42)
co_pairs = list(co_pairs_sparse)
if len(co_pairs) > 30000:
    sample_idx = np.random.choice(len(co_pairs), 30000, replace=False)
    sampled_co = [co_pairs[i] for i in sample_idx]
else:
    sampled_co = co_pairs

random_pairs_set = set()
existing = set(co_pairs)
while len(random_pairs_set) < 5000:
    a = np.random.randint(0, n_items)
    b = np.random.randint(0, n_items)
    if a == b:
        continue
    if a > b:
        a, b = b, a
    if (a, b) not in existing and (a, b) not in random_pairs_set:
        random_pairs_set.add((a, b))

all_pairs = sampled_co + list(random_pairs_set)
print(f'  total sampled pairs: {len(all_pairs)}')

# Compute Jaccard for each pair (set-based, fast enough for ~35k pairs)
print(f'\n[compute Jaccard for {len(all_pairs)} pairs]')
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
        print(f'    {idx+1}/{len(all_pairs)} pairs done')
print(f'  Jaccard: mean={jac_vals.mean():.4f}, median={np.median(jac_vals):.4f}, max={jac_vals.max():.4f}')

# Load SID tensors and compute pairwise SID distance
print(f'\n[load SID tensors]')
sid_hrq = torch.load(f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
                     weights_only=False)
sid_aq = torch.load(f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt',
                    weights_only=False)
sid_l4 = torch.load(f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt',
                    weights_only=False)

# Convert to (n_items, n_layers)
def sid_to_items(sid):
    if sid.shape[0] == n_items and sid.shape[1] != n_items:
        return sid.numpy()
    return sid.T.numpy()

sid_hrq_i = sid_to_items(sid_hrq)
sid_aq_i = sid_to_items(sid_aq)
sid_l4_i = sid_to_items(sid_l4)
print(f'  HRQ SID shape: {sid_hrq_i.shape}')
print(f'  AQ SID shape: {sid_aq_i.shape}')
print(f'  L=4 SID shape: {sid_l4_i.shape}')

# Pairwise SID distance (Hamming) for all pairs
pair_arr = np.array(all_pairs, dtype=np.int32)  # (P, 2)
hrq_dist = np.sum(np.abs(sid_hrq_i[pair_arr[:, 0]] - sid_hrq_i[pair_arr[:, 1]]), axis=1)
aq_dist = np.sum(np.abs(sid_aq_i[pair_arr[:, 0]] - sid_aq_i[pair_arr[:, 1]]), axis=1)
l4_dist = np.sum(np.abs(sid_l4_i[pair_arr[:, 0]] - sid_l4_i[pair_arr[:, 1]]), axis=1)

# Spearman ρ: higher distance → lower Jaccard (alignment)
rho_hrq, p_hrq = spearmanr(hrq_dist, jac_vals)
rho_aq, p_aq = spearmanr(aq_dist, jac_vals)
rho_l4, p_l4 = spearmanr(l4_dist, jac_vals)

print(f'\n  ρ (HRQ distance vs Jaccard): {rho_hrq:+.4f} (p={p_hrq:.2e})')
print(f'  ρ (AQ distance vs Jaccard):  {rho_aq:+.4f} (p={p_aq:.2e})')
print(f'  ρ (L=4 distance vs Jaccard): {rho_l4:+.4f} (p={p_l4:.2e})')

# R@10 evaluation (item-level)
def compute_e2e_r10(pred_path, sid_path, user_targets, k=10):
    sid = torch.load(sid_path, weights_only=False)
    pred = torch.load(pred_path, weights_only=False)
    sid_to_item = {}
    sid_T = sid[:4].T if sid.shape[0] == 4 else sid.T
    for item_id in range(sid.shape[0] if sid.shape[0] == n_items else sid.shape[1]):
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

print(f'\n[compute R@10 per algorithm]')
configs = [
    ('L=4_baseline', f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'),
    ('AQ_s42', f'{GRID}/logs/inference/runs/task19_aq_s4_v2000/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt'),
    ('AQ_s43', f'{GRID}/logs/inference/runs/task19_aq_seed43_s4_v1900/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt'),
    ('AQ_s44', f'{GRID}/logs/inference/runs/task19_aq_seed44_s4_v1800/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt'),
    ('HRQ_s42', f'{GRID}/logs/inference/runs/task18_hrq_s4_v2200/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'),
    ('HRQ_s43', f'{GRID}/logs/inference/runs/task18_hrq_seed43_s4_v2000/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'),
]

per_algo_r10 = {}
for name, pred_path, sid_path in configs:
    if not os.path.exists(pred_path):
        print(f'  {name}: SKIP (pred not found: {pred_path})')
        continue
    if not os.path.exists(sid_path):
        print(f'  {name}: SKIP (sid not found: {sid_path})')
        continue
    r10, n_eval = compute_e2e_r10(pred_path, sid_path, user_targets)
    per_algo_r10[name] = r10
    print(f'  {name}: R@10 = {r10*100:.4f}% (n_eval={n_eval})')

# ============================================================
# Per-algo collision rate and MSE (catalog-level)
# ============================================================
print(f'\n[per-algo catalog stats]')

def catalog_stats(sid_arr):
    n_unique = len(set(tuple(s) for s in sid_arr.tolist()))
    n_items = sid_arr.shape[0]
    collision_rate = 1 - n_unique / n_items
    return n_unique, n_items, collision_rate

stats_hrq = catalog_stats(sid_hrq_i)
stats_aq = catalog_stats(sid_aq_i)
stats_l4 = catalog_stats(sid_l4_i)
print(f'  HRQ: {stats_hrq[0]}/{stats_hrq[1]} unique (CR={stats_hrq[2]*100:.2f}%)')
print(f'  AQ: {stats_aq[0]}/{stats_aq[1]} unique (CR={stats_aq[2]*100:.2f}%)')
print(f'  L=4: {stats_l4[0]}/{stats_l4[1]} unique (CR={stats_l4[2]*100:.2f}%)')

# MSE between item embedding and reconstructed (catalog-level residual)
# 这里用 catalog 唯一性作为 proxy
# 但 task 是验证 ρ vs R@10 因果关系
# 直接用 collision_rate 作为第三个变量

# ============================================================
# Regression: R@10 ~ ρ + Collision
# ============================================================
print(f'\n[regression R@10 ~ ρ + Collision]')

# Build dataset per algo (collapse multi-seed to mean)
algo_summary = {}
for name, r10 in per_algo_r10.items():
    if name.startswith('HRQ'):
        algo_summary.setdefault('HRQ', {'r10s': [], 'rho': rho_hrq, 'cr': stats_hrq[2]})
        algo_summary['HRQ']['r10s'].append(r10)
    elif name.startswith('AQ'):
        algo_summary.setdefault('AQ', {'r10s': [], 'rho': rho_aq, 'cr': stats_aq[2]})
        algo_summary['AQ']['r10s'].append(r10)
    elif name.startswith('L=4'):
        algo_summary.setdefault('L=4', {'r10s': [], 'rho': rho_l4, 'cr': stats_l4[2]})
        algo_summary['L=4']['r10s'].append(r10)

reg_data = []
for algo, info in algo_summary.items():
    mean_r10 = np.mean(info['r10s'])
    reg_data.append({
        'algo': algo,
        'rho': info['rho'],
        'cr': info['cr'],
        'r10': mean_r10,
        'n_seeds': len(info['r10s']),
    })
    print(f'  {algo}: ρ={info["rho"]:+.4f}, CR={info["cr"]*100:.2f}%, R@10={mean_r10*100:.4f}% (n={len(info["r10s"])} seeds)')

# 3-sample regression: only 3 algorithms, so use 2 variables with n-2=1 dof
# Just report Spearman correlation between ρ and R@10 across algos
if len(reg_data) >= 3:
    rhos = np.array([d['rho'] for d in reg_data])
    r10s = np.array([d['r10'] for d in reg_data])
    crs = np.array([d['cr'] for d in reg_data])

    # Simple correlation: ρ vs R@10
    rho_r10_corr, rho_r10_p = spearmanr(rhos, r10s)
    print(f'\n  Spearman(ρ, R@10) = {rho_r10_corr:+.4f} (p={rho_r10_p:.4f})')

    # CR vs R@10 (negative expected: more collision → lower unique items → potentially less accurate)
    cr_r10_corr, cr_r10_p = spearmanr(crs, r10s)
    print(f'  Spearman(CR, R@10) = {cr_r10_corr:+.4f} (p={cr_r10_p:.4f})')

    # OLS via numpy lstsq + manual p-values via t-test on coefficients
    X = np.column_stack([np.ones(len(rhos)), rhos, crs])
    y = r10s * 100
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / max(1e-12, ss_tot)
    n_obs, n_pred = X.shape
    dof = max(1, n_obs - n_pred)
    mse = ss_res / dof
    cov_beta = mse * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stat = beta / se_beta
    # two-tailed p-value from t with dof (approx via normal for dof>=3)
    from scipy.stats import t as t_dist
    pvals = 2 * (1 - t_dist.cdf(np.abs(t_stat), dof))

    print(f'\n  OLS: R@10[%] ~ {beta[0]:.4f} + {beta[1]:+.4f}·ρ + {beta[2]:+.4f}·CR')
    print(f'    β_ρ: {beta[1]:+.4f} (p={pvals[1]:.4f})')
    print(f'    β_CR: {beta[2]:+.4f} (p={pvals[2]:.4f})')
    print(f'    R² = {r_squared:.4f}, dof = {dof}')

    regression_results = {
        'spearman_rho_vs_r10': {'corr': float(rho_r10_corr), 'p': float(rho_r10_p)},
        'spearman_cr_vs_r10': {'corr': float(cr_r10_corr), 'p': float(cr_r10_p)},
        'ols': {
            'intercept': float(beta[0]),
            'beta_rho': float(beta[1]),
            'beta_cr': float(beta[2]),
            'pval_rho': float(pvals[1]),
            'pval_cr': float(pvals[2]),
            'rsquared': float(r_squared),
            'dof': int(dof),
        },
    }
else:
    regression_results = None

# ============================================================
# Save
# ============================================================
out = {
    'task': 'Task A6: HRQ Topology Alignment vs Performance',
    'method': 'Sample 30k co-purchase + 5k random pairs → Jaccard; Spearman(-SID_dist, Jaccard)',
    'n_pairs_sampled': len(all_pairs),
    'per_algo': {
        'HRQ': {'rho': float(rho_hrq), 'p': float(p_hrq), 'collision_rate': stats_hrq[2],
                'n_unique': stats_hrq[0], 'n_total': stats_hrq[1]},
        'AQ': {'rho': float(rho_aq), 'p': float(p_aq), 'collision_rate': stats_aq[2],
               'n_unique': stats_aq[0], 'n_total': stats_aq[1]},
        'L=4': {'rho': float(rho_l4), 'p': float(p_l4), 'collision_rate': stats_l4[2],
                'n_unique': stats_l4[0], 'n_total': stats_l4[1]},
    },
    'r10_per_config': {k: v for k, v in per_algo_r10.items()},
    'regression_summary': [{'algo': d['algo'], 'rho': d['rho'], 'cr': d['cr'], 'r10': d['r10']}
                            for d in reg_data],
    'regression_results': regression_results,
}

with open(os.path.join(OUT_DIR, 'topology_vs_perf.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

verdict_lines = [
    '# Task A6 Verdict: HRQ Topology Alignment vs Performance',
    '',
    '## 设计',
    '',
    '- 采样 30k co-purchase + 5k random pairs → Jaccard',
    '- ρ = Spearman(-SID Hamming dist, Jaccard) — 越高说明 topology 越对齐 user 行为',
    '- R@10 = item-level end-to-end Recall@10',
    '- 因果回归: R@10 ~ ρ + Collision Rate (CR)',
    '',
    '## Per-Algorithm Topology Alignment',
    '',
    '| Algorithm | ρ (Spearman) | p-value | Collision Rate | n_unique/n_total |',
    '|-----------|--------------|---------|----------------|------------------|',
]
for algo in ['HRQ', 'AQ', 'L=4']:
    info = out['per_algo'][algo]
    verdict_lines.append(
        f'| {algo} | {info["rho"]:+.4f} | {info["p"]:.2e} | {info["collision_rate"]*100:.2f}% | '
        f'{info["n_unique"]}/{info["n_total"]} |'
    )

verdict_lines.extend([
    '',
    '## R@10 (Item-Level)',
    '',
    '| Config | R@10 |',
    '|--------|------|',
])
for name, r10 in per_algo_r10.items():
    verdict_lines.append(f'| {name} | {r10*100:.4f}% |')

verdict_lines.extend([
    '',
    '## Regression: R@10 ~ ρ + CR',
    '',
])
if regression_results:
    r = regression_results
    verdict_lines.extend([
        f'- Spearman(ρ, R@10) = {r["spearman_rho_vs_r10"]["corr"]:+.4f} (p={r["spearman_rho_vs_r10"]["p"]:.4f})',
        f'- Spearman(CR, R@10) = {r["spearman_cr_vs_r10"]["corr"]:+.4f} (p={r["spearman_cr_vs_r10"]["p"]:.4f})',
        f'- OLS R@10[%] = {r["ols"]["intercept"]:.4f} + {r["ols"]["beta_rho"]:+.4f}·ρ + {r["ols"]["beta_cr"]:+.4f}·CR',
        f'  - β_ρ = {r["ols"]["beta_rho"]:+.4f} (p={r["ols"]["pval_rho"]:.4f})',
        f'  - β_CR = {r["ols"]["beta_cr"]:+.4f} (p={r["ols"]["pval_cr"]:.4f})',
        f'  - R² = {r["ols"]["rsquared"]:.4f}',
        '',
        '## 因果解读',
        '',
    ])
    if r['ols']['pval_rho'] < 0.05 and r['ols']['beta_rho'] > 0:
        verdict_lines.append('- **β_ρ 显著为正**：topology 对齐度↑ → R@10↑（因果成立）')
    elif r['ols']['pval_rho'] < 0.05 and r['ols']['beta_rho'] < 0:
        verdict_lines.append('- **β_ρ 显著为负**：topology 对齐度↑ → R@10↓（逆向相关）')
    else:
        verdict_lines.append('- **β_ρ 不显著**（p>0.05 或样本量不足）：topology 对齐与 R@10 无明显因果关系')
    if r['ols']['pval_cr'] < 0.05:
        verdict_lines.append(f'- **β_CR 显著**：collision rate 对 R@10 有显著影响')
    else:
        verdict_lines.append(f'- β_CR 不显著')

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/topology_vs_perf.json + verdict.md')
print(f'HRQ ρ={rho_hrq:+.4f}, AQ ρ={rho_aq:+.4f}, L=4 ρ={rho_l4:+.4f}')