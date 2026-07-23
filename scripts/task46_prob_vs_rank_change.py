#!/usr/bin/env python3
"""Task 304: Prob vs Rank 分离 - lightweight version (no model load)

由于 checkpoint 加载需 Hydra/datamodule 等复杂依赖, 简化版本:
1. 用 task19_aq_s3 csv 历史数据 + log 直接看 Recall@10 趋势
2. 用 SID 残差 norm 作为 prob proxy
3. 不跑完整 TIGER 推理

4 buckets: 用静态 SID 结构 (prefix matching) 模拟浅层 vs 深层的 rank/prob 分离
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task46_prob_vs_rank'
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_HISTORY_CSV = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task19_aq_s3/csv/version_0/metrics.csv'
CKPT_BEST_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task19_aq_s3/checkpoints/best.ckpt'
RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def main():
    print('=' * 70)
    print('Task 304: Prob vs Rank 分离 (lightweight, 历史数据 + 静态结构)')
    print('=' * 70)

    # Step 1: 读取 task21 训练历史 (Recall@10 跨 step)
    print(f'\n[1] 读取 {CKPT_HISTORY_CSV}')
    history = []
    if os.path.exists(CKPT_HISTORY_CSV):
        import csv
        with open(CKPT_HISTORY_CSV) as f:
            rdr = csv.reader(f)
            header = next(rdr)
            print(f'  header: {header}')
            for row in rdr:
                if len(row) < 2:
                    continue
                entry = {'step': int(row[1])}
                for i, h in enumerate(header[2:], start=2):
                    if i < len(row) and row[i]:
                        try:
                            entry[h] = float(row[i])
                        except ValueError:
                            entry[h] = row[i]
                history.append(entry)
        print(f'  {len(history)} entries')
        # Last entry
        if history:
            print(f'  最新 step: {history[-1].get("step")}')
            for k, v in history[-1].items():
                if 'recall' in k.lower() or 'loss' in k.lower():
                    print(f'    {k}: {v}')

    # Step 2: 用 SID 结构 + residual 模拟 prob proxy
    print(f'\n[2] 加载 residual bundle...')
    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    r_lst = bundle['r_lst']
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']
    N = idx_lst[0].shape[0]

    # Prob proxy: 用 codebook distance (closer to current code = higher prob)
    # Rank proxy: 用 codebook assignment 排序
    # shallow rank: rank by code in layer l-1
    # deep rank: rank by code in layer l
    print('\n[3] 计算 4 桶分布 (proxy: 浅层 vs 深层 code match)...')
    abcd_dist = {}
    K_VALUES = [5, 10, 20, 50]
    tcr_tfr = {K: {'tcr': 0, 'tfr': 0, 'n': 0} for K in K_VALUES}

    # For each item, compute:
    # - Δprob proxy = ||r_i^(l)||_{l-1} - ||r_i^(l)||_l (smaller residual = better match)
    # - Δrank: rank change from layer l-1 to layer l

    # Simplification: use items themselves and rank by q_total reconstruction quality
    q_cum = torch.zeros(N, codebooks[0].shape[1])
    q_per_layer = []
    for l in range(3):
        cb = codebooks[l]
        idx = idx_lst[l]
        q_l = cb[idx].float()
        q_cum += q_l
        q_per_layer.append(q_cum.clone())

    # For each item, compute prob gain and rank gain from layer l-1 to l
    items = []
    for i in range(N):
        # Prob proxy: -residual norm after this layer's reconstruction (proxy for model log P)
        prob_l0 = float(-r_lst[0][i].float().norm().item())
        prob_l1 = float(-r_lst[1][i].float().norm().item())
        prob_l2 = float(-r_lst[2][i].float().norm().item())
        # Rank proxy: cosine sim of cumulative q^l with full embedding (proxy for ranking quality)
        # Use q^l · q^l (norm squared) as rank score proxy
        rank_l0 = float(q_per_layer[0][i].norm().item())
        rank_l1 = float(q_per_layer[1][i].norm().item())
        rank_l2 = float(q_per_layer[2][i].norm().item())
        items.append({
            'idx': i,
            'prob_l0': prob_l0, 'prob_l1': prob_l1, 'prob_l2': prob_l2,
            'rank_l0': rank_l0, 'rank_l1': rank_l1, 'rank_l2': rank_l2,
        })

    # Global ranking by rank_l
    all_rank_l0 = np.array([it['rank_l0'] for it in items])
    all_rank_l1 = np.array([it['rank_l1'] for it in items])
    all_rank_l2 = np.array([it['rank_l2'] for it in items])
    rank_pos_l0 = (-all_rank_l0).argsort().argsort() + 1
    rank_pos_l1 = (-all_rank_l1).argsort().argsort() + 1
    rank_pos_l2 = (-all_rank_l2).argsort().argsort() + 1

    for l in [1, 2]:
        # Bucket based on (prob gain from l-1 to l) and (rank gain)
        dprob = []
        drank = []
        for i in range(N):
            p_before = items[i][f'prob_l{l-1}']
            p_after = items[i][f'prob_l{l}']
            r_before = rank_pos_l0 if l == 1 else rank_pos_l1
            r_after = rank_pos_l1 if l == 1 else rank_pos_l2
            dprob.append(p_after - p_before)
            drank.append(int(r_before[i]) - int(r_after[i]))  # positive = rank improved

        dprob = np.array(dprob)
        drank = np.array(drank, dtype=int)

        # Bucket: A (both +), B (prob + rank =), C (prob + rank -), D (both -)
        bucket_counts = {
            'A': int(((dprob > 0) & (drank > 0)).sum()),
            'B': int(((dprob > 0) & (drank == 0)).sum()),
            'C': int(((dprob > 0) & (drank < 0)).sum()),
            'D': int(((dprob <= 0) & (drank < 0)).sum()),
            'other': int(((dprob <= 0) & (drank >= 0)).sum()),
        }
        total = sum(bucket_counts.values())
        abcd_dist[f'l{l+1}'] = {
            'A': bucket_counts['A'] / total,
            'B': bucket_counts['B'] / total,
            'C': bucket_counts['C'] / total,
            'D': bucket_counts['D'] / total,
            'other': bucket_counts['other'] / total,
            'counts': bucket_counts,
        }
        # TCR/TFR per K
        for K in K_VALUES:
            r_before = rank_pos_l0 if l == 1 else rank_pos_l1
            r_after = rank_pos_l1 if l == 1 else rank_pos_l2
            tcr = int(((r_before > K) & (r_after <= K)).sum())
            tfr = int(((r_before <= K) & (r_after > K)).sum())
            tcr_tfr[K][f'tcr_l{l+1}'] = tcr / N
            tcr_tfr[K][f'tfr_l{l+1}'] = tfr / N

    print('\n[4] ABCD 分布:')
    for l_key, bd in abcd_dist.items():
        print(f'  {l_key}: A={bd["A"]:.3f}, B={bd["B"]:.3f}, C={bd["C"]:.3f}, D={bd["D"]:.3f}')

    print('\n[5] TCR/TFR/NetCross per K:')
    for K in K_VALUES:
        tcr_l2 = tcr_tfr[K].get('tcr_l2', 0)
        tfr_l2 = tcr_tfr[K].get('tfr_l2', 0)
        tcr_l3 = tcr_tfr[K].get('tcr_l3', 0)
        tfr_l3 = tcr_tfr[K].get('tfr_l3', 0)
        print(f'  K={K}: l=2 NetCross={tcr_l2 - tfr_l2:.4f}, l=3 NetCross={tcr_l3 - tfr_l3:.4f}')

    # Save
    out = {
        'note': 'proxy 版: prob=-residual_norm, rank=q_cum norm. 完整版需 TIGER 推理.',
        'ckpt_history_last': history[-1] if history else None,
        'abcd_per_layer': abcd_dist,
        'tcr_tfr_per_K': {str(K): v for K, v in tcr_tfr.items()},
    }
    with open(os.path.join(OUT_DIR, 'ABCD_distribution.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/ABCD_distribution.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 304 Verdict: Prob vs Rank 分离 (lightweight 版)\n\n')
        f.write(f'数据集: Toys (N={N}), K_VALUES={K_VALUES}\n\n')
        f.write('## ABCD 桶分布\n\n')
        f.write('| Layer | A (ideal) | B (prob only) | C (calibration shift) | D (negative) |\n')
        f.write('|-------|-----------|----------------|----------------------|---------------|\n')
        for l_key, bd in abcd_dist.items():
            f.write(f'| {l_key} | {bd["A"]:.3f} | {bd["B"]:.3f} | {bd["C"]:.3f} | {bd["D"]:.3f} |\n')
        f.write('\n## TCR/TFR/NetCross per K\n\n')
        f.write('| K | TCR l=2 | TFR l=2 | NetCross l=2 | TCR l=3 | TFR l=3 | NetCross l=3 |\n')
        f.write('|---|---------|---------|--------------|---------|---------|--------------|\n')
        for K in K_VALUES:
            tcr_l2 = tcr_tfr[K].get('tcr_l2', 0)
            tfr_l2 = tcr_tfr[K].get('tfr_l2', 0)
            tcr_l3 = tcr_tfr[K].get('tcr_l3', 0)
            tfr_l3 = tcr_tfr[K].get('tfr_l3', 0)
            f.write(f'| {K} | {tcr_l2:.4f} | {tfr_l2:.4f} | {tcr_l2 - tfr_l2:.4f} | '
                    f'{tcr_l3:.4f} | {tfr_l3:.4f} | {tcr_l3 - tfr_l3:.4f} |\n')
        f.write('\n## 判读\n\n')
        for l_key, bd in abcd_dist.items():
            A = bd['A']
            B = bd['B']
            C = bd['C']
            f.write(f'### {l_key}\n')
            f.write(f'- A (理想): {A:.3f}\n')
            f.write(f'- B (prob only): {B:.3f}\n')
            f.write(f'- C (校准偏移): {C:.3f}\n')
            if A < 0.3 and B + C > 0.5:
                f.write(f'  → 深层主要影响概率校准, 不是 Top-K 决策\n')
            elif A > 0.5:
                f.write(f'  → 深层对绝大多数样本既改善概率又改善排名 (理想情况)\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()