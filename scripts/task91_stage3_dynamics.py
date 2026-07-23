#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #91 — Stage 3 T5 Training Dynamics Comparison.

Parse 8 HG_Rec.log files and compare:
- Convergence speed (epoch to reach valid R@10 ≥ 0.08)
- Best epoch (NDCG@20 highest)
- Training duration
- Valid vs test R@10 gap (generalization)

CPU only - pure regex parsing.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Optional

# Map log path -> config name
LOG_PATHS = {
    'vanilla (phonism)': '/home/wlia0047/ar57/wenyu/GeneRec/logs/Instruments/Jul-23-2026_20-24-44/HG_Rec.log',
    'HG-Rec c111 (Task #84/88)': '/home/wlia0047/ar57/wenyu/GeneRec/logs/Instruments/Jul-23-2026_23-44-03/HG_Rec.log',
    'HG-Rec c222': '/home/wlia0047/ar57/wenyu/GeneRec/logs/Instruments/Jul-24-2026_00-06-10/HG_Rec.log',
    'HG-Rec c555': '/home/wlia0047/ar57/wenyu/GeneRec/logs/Instruments/Jul-23-2026_23-48-37/HG_Rec.log',
    'HG-Rec c512': '/home/wlia0047/ar57/wenyu/GeneRec/logs_curv_0.5_1.0_2.0/Instruments/Jul-24-2026_00-17-36/HG_Rec.log',
    'HG-Rec c215': '/home/wlia0047/ar57/wenyu/GeneRec/logs_curv_2.0_1.0_0.5/Instruments/Jul-24-2026_00-58-58/HG_Rec.log',
    'HG-Rec c1055': '/home/wlia0047/ar57/wenyu/GeneRec/logs_curv_1.0_0.5_0.5/Instruments/Jul-24-2026_00-59-22/HG_Rec.log',
    'HG-Rec free-curv': '/home/wlia0047/ar57/wenyu/GeneRec/logs_curv_free_M1/Instruments/Jul-24-2026_02-11-11/HG_Rec.log',
}

# Stage 4 test R@10 (from verdicts)
TEST_R10 = {
    'vanilla (phonism)': 0.1058,
    'HG-Rec c111 (Task #84/88)': 0.1020,  # Task #84 (vanilla c111 was rerun)
    'HG-Rec c222': 0.1036,
    'HG-Rec c555': 0.1051,
    'HG-Rec c512': 0.0998,
    'HG-Rec c215': 0.1028,
    'HG-Rec c1055': 0.1015,
    'HG-Rec free-curv': 0.1015,
}


def parse_log(path: str) -> dict:
    """Parse one HG_Rec.log file. Return dict with epoch-level metrics + summary."""
    if not os.path.exists(path):
        return {'exists': False, 'path': path}

    epochs = []  # list of dicts {epoch, train_loss, val_r5, val_r10, val_r20, val_n5, val_n10, val_n20, best_n20}
    best_n20 = 0.0
    best_epoch = 0
    first_r10_ge_008 = None
    first_r10_ge_009 = None
    first_r10_ge_010 = None
    timestamp_first = None
    timestamp_last = None

    # For 00-06-10 log: contains both c512 and c222. We need to parse only c222 portion
    # (c222 has curv_2.0_2.0_2.0, the SECOND config in the log).
    parse_all = 'c222' in os.path.basename(os.path.dirname(os.path.dirname(path))) or \
                not path.endswith('00-06-10/HG_Rec.log')

    current_section = 'unknown'
    skip_until_section = None

    with open(path) as f:
        for line in f:
            # Detect config code_path (which curvature is being used)
            if "code_path'" in line:
                m = re.search(r"code_path':\s*'([^']+)'", line)
                if m:
                    curv = m.group(1)
                    if 'curv_2.0_2.0_2.0' in curv:
                        current_section = 'c222'
                    elif 'curv_0.5_1.0_2.0' in curv:
                        current_section = 'c512'
                    elif 'curv_0.5_0.5_0.5' in curv:
                        current_section = 'c555'
                    elif 'curv_1.0_1.0_1.0' in curv:
                        current_section = 'c111'
                    elif 'curv_2.0_1.0_0.5' in curv:
                        current_section = 'c215'
                    elif 'curv_1.0_0.5_0.5' in curv:
                        current_section = 'c1055'
                    elif 'curv_free_M1' in curv:
                        current_section = 'free-curv'
                    elif '_t5_hrqvae_poincare.npy' in curv and 'curv_' not in curv:
                        current_section = 'vanilla'

            # For 00-06-10 log, only parse c222 section
            if '00-06-10' in path:
                # Skip until we see curv_2.0_2.0_2.0 then parse until end of file
                if current_section == 'c512' and skip_until_section != 'c222':
                    continue

            # Parse epoch header
            m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Epoch (\d+)/200', line)
            if m:
                ts_str = m.group(1)
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                if timestamp_first is None:
                    timestamp_first = ts
                timestamp_last = ts
                current_epoch = int(m.group(2))
                epochs.append({'epoch': current_epoch})

            # Parse training loss
            m = re.search(r'Training loss: ([0-9.]+)', line)
            if m and epochs:
                epochs[-1]['train_loss'] = float(m.group(1))

            # Parse validation metrics
            m = re.search(r"Validation Dataset: \{'Recall@5': ([0-9.]+), 'Recall@10': ([0-9.]+), 'Recall@20': ([0-9.]+)\}", line)
            if m and epochs:
                epochs[-1]['val_r5'] = float(m.group(1))
                epochs[-1]['val_r10'] = float(m.group(2))
                epochs[-1]['val_r20'] = float(m.group(3))

            m = re.search(r"Validation Dataset: \{'NDCG@5': ([0-9.]+), 'NDCG@10': ([0-9.]+), 'NDCG@20': ([0-9.]+)\}", line)
            if m and epochs:
                epochs[-1]['val_n5'] = float(m.group(1))
                epochs[-1]['val_n10'] = float(m.group(2))
                epochs[-1]['val_n20'] = float(m.group(3))

            # Parse best NDCG@20
            m = re.search(r'Best NDCG@20: ([0-9.]+)', line)
            if m and epochs:
                n20 = float(m.group(1))
                epochs[-1]['best_n20'] = n20
                if n20 > best_n20:
                    best_n20 = n20
                    best_epoch = epochs[-1]['epoch']

    # Find first epoch reaching R@10 thresholds
    for ep in epochs:
        if 'val_r10' not in ep:
            continue
        if first_r10_ge_008 is None and ep['val_r10'] >= 0.08:
            first_r10_ge_008 = ep['epoch']
        if first_r10_ge_009 is None and ep['val_r10'] >= 0.09:
            first_r10_ge_009 = ep['epoch']
        if first_r10_ge_010 is None and ep['val_r10'] >= 0.10:
            first_r10_ge_010 = ep['epoch']

    # Compute best epoch metrics
    best_ep_metrics = None
    for ep in epochs:
        if ep['epoch'] == best_epoch:
            best_ep_metrics = ep
            break

    # Training duration
    duration_minutes = None
    if timestamp_first and timestamp_last:
        duration_minutes = (timestamp_last - timestamp_first).total_seconds() / 60.0

    return {
        'exists': True,
        'path': path,
        'n_epochs': len(epochs),
        'best_n20': best_n20,
        'best_epoch': best_epoch,
        'first_epoch_r10_ge_008': first_r10_ge_008,
        'first_epoch_r10_ge_009': first_r10_ge_009,
        'first_epoch_r10_ge_010': first_r10_ge_010,
        'duration_minutes': duration_minutes,
        'best_epoch_metrics': best_ep_metrics,
        'last_epoch': epochs[-1] if epochs else None,
    }


def main():
    print("=" * 100)
    print("Task #91 — Stage 3 T5 Training Dynamics Comparison")
    print("=" * 100)

    results = {}
    for name, path in LOG_PATHS.items():
        r = parse_log(path)
        results[name] = r

    # Print summary table
    print(f"\n{'Config':<32} {'Epochs':<8} {'BestNDCG20':<12} {'BestEp':<8} "
          f"{'FirstR10≥0.08':<14} {'FirstR10≥0.09':<14} {'FirstR10≥0.10':<14} "
          f"{'ValidR10@best':<14} {'TestR10':<10} {'Gap':<10} {'Dur(min)':<10}")
    print("-" * 140)
    for name, r in results.items():
        if not r['exists']:
            print(f"{name:<32} MISSING")
            continue
        valid_r10_at_best = r['best_epoch_metrics'].get('val_r10', None) if r['best_epoch_metrics'] else None
        test_r10 = TEST_R10.get(name, None)
        gap = (valid_r10_at_best - test_r10) if (valid_r10_at_best and test_r10) else None
        dur = r['duration_minutes']
        print(f"{name:<32} {r['n_epochs']:<8} {r['best_n20']:<12.4f} {r['best_epoch']:<8} "
              f"{str(r['first_epoch_r10_ge_008'] or '-'):<14} {str(r['first_epoch_r10_ge_009'] or '-'):<14} "
              f"{str(r['first_epoch_r10_ge_010'] or '-'):<14} "
              f"{(f'{valid_r10_at_best:.4f}' if valid_r10_at_best else '-'):<14} "
              f"{(f'{test_r10:.4f}' if test_r10 else '-'):<10} "
              f"{(f'{gap:+.4f}' if gap is not None else '-'):<10} "
              f"{(f'{dur:.1f}' if dur else '-'):<10}")

    # Convergence speed ranking
    print("\n" + "=" * 100)
    print("Convergence Speed Ranking (First epoch reaching valid R@10 ≥ 0.09, lower = faster)")
    print("=" * 100)
    speed_rank = []
    for name, r in results.items():
        if r['exists'] and r['first_epoch_r10_ge_009'] is not None:
            speed_rank.append((name, r['first_epoch_r10_ge_009']))
    speed_rank.sort(key=lambda x: x[1])
    for i, (name, ep) in enumerate(speed_rank, 1):
        print(f"  {i}. {name:<32} epoch {ep}")

    # Best valid R@10 ranking
    print("\n" + "=" * 100)
    print("Best Valid R@10 Ranking (from logs)")
    print("=" * 100)
    valid_rank = []
    for name, r in results.items():
        if r['exists'] and r['best_epoch_metrics'] and 'val_r10' in r['best_epoch_metrics']:
            valid_rank.append((name, r['best_epoch_metrics']['val_r10'], r['best_epoch']))
    valid_rank.sort(key=lambda x: -x[1])
    for i, (name, vr10, ep) in enumerate(valid_rank, 1):
        print(f"  {i}. {name:<32} valid_R@10={vr10:.4f} @ epoch {ep}")

    # Save results
    out_path = 'verdicts/task91_stage3_dynamics.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'task': 'Task #91 Stage 3 dynamics', 'per_config': results,
                   'test_R10_baseline': TEST_R10}, f, indent=2, default=str)
    print(f"\n[Done] Results saved to {out_path}")


if __name__ == '__main__':
    main()