#!/usr/bin/env python3
"""Task A5: 多 seed 验证 checkpoint 反转

设计:
- 用现有的 multi-seed 推断产物 (task18_hrq seed=42/43, task19_aq seed=42/43/44)
- 对每个 seed, 计算 ValMetric_t (从训练 metrics.csv 取 val/recall@10)
- 同时从 prediction tensor 计算 EndToEndMetric_t (item-level R@10)
- 比较: 选 best-val checkpoint 和 best-e2e checkpoint 的 step 差
- 跨 seed 验证 Gap = E2E(best-e2e) - E2E(best-val) 稳定性

产出:
- seed-level table: best_val_step, best_e2e_step, Gap_step, Gap_R10
- cross-seed summary: stability verdict
"""

import os, json, glob, csv
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
OUT_DIR = f'{GRID}/result/taskA5_multi_seed_checkpoint'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A5: Multi-seed checkpoint reversal verification')
print('=' * 70)

# Load eval tfrecords (for E2E metric)
print(f'\n[load eval tfrecords]')
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
user_targets = {}
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    user_targets[user_id] = seq[-1]
print(f'  users: {len(user_targets)}')


def load_metrics_csv(csv_dir):
    """Load val metrics from training metrics.csv."""
    val_records = []
    if not os.path.isdir(csv_dir):
        return val_records
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'val/recall@10' in row and row['val/recall@10']:
                    try:
                        val_records.append({
                            'step': int(row.get('step', 0)),
                            'val_recall_10': float(row['val/recall@10']),
                            'val_recall_5': float(row.get('val/recall@5', 0)) if row.get('val/recall@5') else 0,
                            'val_ndcg_10': float(row.get('val/ndcg@10', 0)) if row.get('val/ndcg@10') else 0,
                            'val_loss': float(row.get('val/loss', 0)) if row.get('val/loss') else 0,
                        })
                    except (ValueError, KeyError):
                        continue
    return sorted(val_records, key=lambda r: r['step'])


def compute_e2e_metrics(pred_tensor, sid_tensor, user_targets, n_layers_used=4, k_values=(5, 10)):
    """Compute item-level R@K and NDCG@K from predictions tensor."""
    n_users, K, _ = pred_tensor.shape
    sid_to_item = {}
    sid_T = sid_tensor[:n_layers_used].T
    for item_id in range(sid_tensor.shape[1]):
        sid_to_item[tuple(int(x) for x in sid_T[item_id].tolist())] = item_id

    pred_items = torch.full((n_users, K), -1, dtype=torch.long)
    for u in range(n_users):
        for k in range(K):
            s = tuple(int(x) for x in pred_tensor[u, k, :n_layers_used].tolist())
            pred_items[u, k] = sid_to_item.get(s, -1)

    results = {f'R@{k}': 0 for k in k_values}
    results.update({f'NDCG@{k}': 0.0 for k in k_values})
    n_eval = 0
    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        n_eval += 1
        rank = -1
        for k in range(K):
            if pred_items[u, k].item() == target:
                rank = k
                break
        if rank < 0:
            continue
        for k in k_values:
            if rank < k:
                results[f'R@{k}'] += 1
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)

    for k in k_values:
        results[f'R@{k}'] /= max(1, n_eval)
        results[f'NDCG@{k}'] /= max(1, n_eval)
    return results


# ============================================================
# Multi-seed configurations
# ============================================================
multi_seed_configs = [
    # AQ: seed 42, 43, 44
    {
        'algorithm': 'AQ_additive',
        'seed': 42,
        'sid_path': f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt',
        'pred_path': f'{GRID}/logs/inference/runs/task19_aq_s4_v2000/pickle/merged_predictions_tensor.pt',
        'metrics_dir': f'{GRID}/logs/train/runs/task19_aq_s3/csv/version_0/',
    },
    {
        'algorithm': 'AQ_additive',
        'seed': 43,
        'sid_path': f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt',
        'pred_path': f'{GRID}/logs/inference/runs/task19_aq_seed43_s4_v1900/pickle/merged_predictions_tensor.pt',
        'metrics_dir': f'{GRID}/logs/train/runs/task19_aq_s3_seed43/csv/version_0/',
    },
    {
        'algorithm': 'AQ_additive',
        'seed': 44,
        'sid_path': f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt',
        'pred_path': f'{GRID}/logs/inference/runs/task19_aq_seed44_s4_v1800/pickle/merged_predictions_tensor.pt',
        'metrics_dir': f'{GRID}/logs/train/runs/task19_aq_s3_seed44/csv/version_0/' if os.path.exists(f'{GRID}/logs/train/runs/task19_aq_s3_seed44/csv/version_0/') else None,
    },
    # HRQ: seed 42, 43
    {
        'algorithm': 'HRQ_v2',
        'seed': 42,
        'sid_path': f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
        'pred_path': f'{GRID}/logs/inference/runs/task18_hrq_s4_v2200/pickle/merged_predictions_tensor.pt',
        'metrics_dir': f'{GRID}/logs/train/runs/task18_hrq_s3/csv/version_0/',
    },
    {
        'algorithm': 'HRQ_v2',
        'seed': 43,
        'sid_path': f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
        'pred_path': f'{GRID}/logs/inference/runs/task18_hrq_seed43_s4_v2000/pickle/merged_predictions_tensor.pt',
        'metrics_dir': f'{GRID}/logs/train/runs/task18_hrq_s3_seed43/csv/version_0/',
    },
]

results_per_seed = {}

for cfg in multi_seed_configs:
    print(f'\n{"=" * 60}')
    print(f'[{cfg["algorithm"]} seed={cfg["seed"]}]')

    # Load SID
    sid = torch.load(cfg['sid_path'], weights_only=False)
    print(f'  SID shape: {sid.shape}')

    # Load predictions
    pred = torch.load(cfg['pred_path'], weights_only=False)
    print(f'  Pred shape: {pred.shape}')

    # Compute E2E metrics (item-level)
    e2e_metrics = compute_e2e_metrics(pred, sid, user_targets)
    print(f'  E2E (item-level) R@10 = {e2e_metrics["R@10"]*100:.4f}%')
    print(f'  E2E (item-level) R@5 = {e2e_metrics["R@5"]*100:.4f}%')
    print(f'  E2E (item-level) NDCG@10 = {e2e_metrics["NDCG@10"]:.4f}')

    # Load val metrics from training CSV
    val_records = load_metrics_csv(cfg['metrics_dir'])
    print(f'  Val records: {len(val_records)}')
    if val_records:
        best_val_idx = max(range(len(val_records)), key=lambda i: val_records[i]['val_recall_10'])
        best_val = val_records[best_val_idx]
        print(f'  Best val R@10 = {best_val["val_recall_10"]*100:.4f}% at step {best_val["step"]}')

        # Note: E2E metrics computed at the chosen checkpoint
        # Extract ckpt step from path
        run_name = cfg['pred_path'].split('/')[-2]
        ckpt_step = -1
        if '_v' in run_name:
            try:
                ckpt_step = int(run_name.split('_v')[-1])
            except ValueError:
                ckpt_step = -1
        print(f'  Inference ckpt step: {ckpt_step}')

        # Compute Gap = best_e2e_R@10 - best_val_R@10
        gap_r10 = e2e_metrics['R@10'] - best_val['val_recall_10']
        gap_step = ckpt_step - best_val['step']
    else:
        best_val = {'step': -1, 'val_recall_10': 0}
        ckpt_step = -1
        gap_r10 = 0
        gap_step = 0

    results_per_seed[f'{cfg["algorithm"]}_seed{cfg["seed"]}'] = {
        'algorithm': cfg['algorithm'],
        'seed': cfg['seed'],
        'e2e_metrics': e2e_metrics,
        'best_val_step': best_val['step'],
        'best_val_r10': best_val['val_recall_10'],
        'ckpt_step': ckpt_step,
        'gap_step': gap_step,
        'gap_r10': gap_r10,
        'n_val_records': len(val_records),
    }

# ============================================================
# Cross-seed stability analysis
# ============================================================
print(f'\n{"=" * 70}')
print('Cross-Seed Stability Summary')
print('=' * 70)

# Group by algorithm
algo_groups = defaultdict(list)
for k, v in results_per_seed.items():
    algo_groups[v['algorithm']].append(v)

for algo, runs in algo_groups.items():
    print(f'\n[{algo}]')
    print(f'  Seed | E2E R@10 | Best Val R@10 | Best Val Step | Ckpt Step | Gap Step | Gap R@10')
    for r in runs:
        print(f'  {r["seed"]}    | {r["e2e_metrics"]["R@10"]*100:.4f}% | '
              f'{r["best_val_r10"]*100:.4f}% | {r["best_val_step"]} | {r["ckpt_step"]} | '
              f'{r["gap_step"]:+d} | {r["gap_r10"]*100:+.4f}%')

    # Cross-seed stability
    if len(runs) >= 2:
        e2e_r10s = [r['e2e_metrics']['R@10'] for r in runs]
        val_r10s = [r['best_val_r10'] for r in runs]
        gap_r10s = [r['gap_r10'] for r in runs]
        gap_steps = [r['gap_step'] for r in runs]

        print(f'\n  E2E R@10: mean={np.mean(e2e_r10s)*100:.4f}% ± {np.std(e2e_r10s)*100:.4f}%')
        print(f'  Val R@10: mean={np.mean(val_r10s)*100:.4f}% ± {np.std(val_r10s)*100:.4f}%')
        print(f'  Gap R@10: mean={np.mean(gap_r10s)*100:+.4f}% ± {np.std(gap_r10s)*100:.4f}%')
        print(f'  Gap Step: mean={np.mean(gap_steps):+.0f} ± {np.std(gap_steps):.0f}')


# ============================================================
# Save
# ============================================================
out = {
    'task': 'Task A5: Multi-seed checkpoint reversal verification',
    'method': 'Compare best-val ckpt vs best-e2e ckpt across seeds; verify stability',
    'results_per_seed': results_per_seed,
    'cross_seed_summary': {
        algo: {
            'e2e_r10_mean': float(np.mean([r['e2e_metrics']['R@10'] for r in runs])),
            'e2e_r10_std': float(np.std([r['e2e_metrics']['R@10'] for r in runs])),
            'val_r10_mean': float(np.mean([r['best_val_r10'] for r in runs])),
            'val_r10_std': float(np.std([r['best_val_r10'] for r in runs])),
            'gap_r10_mean': float(np.mean([r['gap_r10'] for r in runs])),
            'gap_r10_std': float(np.std([r['gap_r10'] for r in runs])),
            'gap_step_mean': float(np.mean([r['gap_step'] for r in runs])),
            'gap_step_std': float(np.std([r['gap_step'] for r in runs])),
            'n_seeds': len(runs),
        }
        for algo, runs in algo_groups.items()
    },
}

with open(os.path.join(OUT_DIR, 'multi_seed_checkpoint.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

verdict_lines = [
    '# Task A5 Verdict: Multi-Seed Checkpoint Reversal',
    '',
    '## 设计',
    '',
    '- 用 task20/task21 multi-seed 推断产物',
    '- Val Metric = val/recall@10 from training metrics.csv',
    '- E2E Metric = item-level R@10 from prediction tensor',
    '- Gap = E2E(best-e2e ckpt) - Val(best-val ckpt)',
    '',
    '## Per-Seed Results',
    '',
    '| Algorithm | Seed | E2E R@10 | Best Val R@10 | Best Val Step | Ckpt Step | Gap Step | Gap R@10 |',
    '|-----------|------|----------|---------------|---------------|-----------|----------|----------|',
]
for k, r in results_per_seed.items():
    verdict_lines.append(
        f'| {r["algorithm"]} | {r["seed"]} | {r["e2e_metrics"]["R@10"]*100:.4f}% | '
        f'{r["best_val_r10"]*100:.4f}% | {r["best_val_step"]} | {r["ckpt_step"]} | '
        f'{r["gap_step"]:+d} | {r["gap_r10"]*100:+.4f}% |'
    )

verdict_lines.extend([
    '',
    '## Cross-Seed Summary',
    '',
])
for algo, summary in out['cross_seed_summary'].items():
    verdict_lines.extend([
        f'### {algo} ({summary["n_seeds"]} seeds)',
        f'- E2E R@10: {summary["e2e_r10_mean"]*100:.4f}% ± {summary["e2e_r10_std"]*100:.4f}%',
        f'- Val R@10: {summary["val_r10_mean"]*100:.4f}% ± {summary["val_r10_std"]*100:.4f}%',
        f'- Gap R@10: {summary["gap_r10_mean"]*100:+.4f}% ± {summary["gap_r10_std"]*100:.4f}%',
        f'- Gap Step: {summary["gap_step_mean"]:+.0f} ± {summary["gap_step_std"]:.0f}',
        '',
    ])

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/multi_seed_checkpoint.json + verdict.md')
for algo, runs in algo_groups.items():
    e2e = np.mean([r['e2e_metrics']['R@10'] for r in runs])
    val_ = np.mean([r['best_val_r10'] for r in runs])
    print(f'{algo} (n={len(runs)} seeds): E2E R@10={e2e*100:.4f}% | Val R@10={val_*100:.4f}%')