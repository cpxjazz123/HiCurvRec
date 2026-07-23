#!/usr/bin/env python3
"""Collision-aware item-level R@10 重算 — 对 5 个算法用 per-algo labels 重算

每个算法 SID map 不同 → 同一 item_id 对应不同 SID tuple
→ 每个算法的"真实标签"必须用该算法的 SID map 提取
→ 因此需要 user_labels_<algo>.pt（已在前一步生成）

Scoring methods:
  - OPTIMISTIC: R@K = I[any top-K beam's SID == true_SID]. Existing SID-level recall.
  - PESSIMISTIC: R@K = sum_{i=1..K} I[s_i == s*] / |collision(s*)|.
  - RANDOM_TIE_BREAK: = PESSIMISTIC for this metric (per derivation).

Outputs:
  - result/collision_aware/collision_aware_R10.json
  - result/collision_aware/verdict.md
"""
import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

GRID_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID")
OUT_DIR = GRID_ROOT / "result" / "collision_aware"
SEED = 42
TOP_K_LIST = [5, 10]


def main():
    t0 = time.time()
    print('=' * 80)
    print('Collision-aware item-level R@10 重算 — 5 算法 × per-algo labels')
    print('=' * 80)

    ALGOS = [
        # (display_name, eval_json, sid_key, s4_key, label_algo_name)
        ('A_baseline', 'result/task15/task3_eval_A_baseline.json', 'sid', 'stage4_tensor', 'A_baseline'),
        ('B_mmq',      'result/task15/task3_eval_B_mmq.json',      'sid', 'stage4_tensor', 'B_mmq'),
        ('C_gsrq',     'result/task15/task3_eval_C_gsrq.json',     'sid', 'stage4_tensor', 'C_gsrq'),
        ('HRQ_v2200',  'result/task20/task18_eval_HRQ_v2200.json',  'sid_path', 'stage4_path', 'HRQ'),
        ('AQ_v2000',   'result/task21/task19_eval_AQ_v2000.json',   'sid_path', 'stage4_path', 'AQ'),
    ]

    rng = np.random.default_rng(SEED)
    all_results = {}

    for algo, eval_json, sid_k, s4_k, label_name in ALGOS:
        print(f'\n--- {algo} ---')

        # Load per-algo user labels
        ul_path = OUT_DIR / f'user_labels_{label_name}.pt'
        user_labels = torch.load(ul_path, map_location='cpu', weights_only=False)
        N_users, L = user_labels.shape
        valid_users = (user_labels >= 0).all(dim=1).nonzero(as_tuple=True)[0].tolist()
        user_true_sids = {u: tuple(user_labels[u].tolist()) for u in valid_users}
        print(f'  user_labels: {tuple(user_labels.shape)}, valid={len(valid_users)}')

        # Load SID tensor and stage4 tensor
        meta = json.load(open(eval_json))
        sid_path = meta[sid_k]
        s4_path = meta[s4_k]
        sid = torch.load(sid_path, map_location='cpu', weights_only=False)
        s4 = torch.load(s4_path, map_location='cpu', weights_only=False)
        # Normalize sid shape to (N_items, L)
        if sid.dim() == 2 and sid.shape[0] == L and sid.shape[1] != L:
            sid = sid.t().contiguous()
        elif sid.dim() == 2 and sid.shape[1] == L:
            pass  # already (N, L)
        else:
            raise ValueError(f'{algo}: unexpected sid shape {tuple(sid.shape)}')
        sid_l = sid.tolist()
        print(f'  sid tensor: {tuple(sid.shape)}')
        print(f'  stage4 tensor: {tuple(s4.shape)}')

        # Build collision map
        collision_map = defaultdict(list)
        for i, s in enumerate(sid_l):
            collision_map[tuple(s)].append(i)
        n_unique = len(collision_map)
        n_total = sid.shape[0]
        coll_rate = 1 - n_unique / n_total
        coll_sizes = [len(v) for v in collision_map.values()]
        print(f'  items={n_total}, unique SID tuples={n_unique}, collision_rate={coll_rate:.4f}')
        print(f'  coll size: min={min(coll_sizes)}, max={max(coll_sizes)}, mean={np.mean(coll_sizes):.2f}')
        collision_count = {k: len(v) for k, v in collision_map.items()}

        # Filter users present in stage4
        algo_users = [u for u in valid_users if u < s4.shape[0]]
        print(f'  scoring {len(algo_users)} users')

        # Score per user
        results = {f'r@{k}': {'opt': 0.0, 'pess': 0.0, 'rand': 0.0} for k in TOP_K_LIST}
        results_n = {f'ndcg@{k}': {'opt': 0.0, 'pess': 0.0, 'rand': 0.0} for k in TOP_K_LIST}

        # Accumulate sums
        sum_opt = {k: 0.0 for k in TOP_K_LIST}
        sum_pess = {k: 0.0 for k in TOP_K_LIST}
        sum_rand = {k: 0.0 for k in TOP_K_LIST}
        sum_ndcg_opt = {k: 0.0 for k in TOP_K_LIST}
        sum_ndcg_pess = {k: 0.0 for k in TOP_K_LIST}
        sum_ndcg_rand = {k: 0.0 for k in TOP_K_LIST}

        n_scored = 0
        for u in algo_users:
            true_sid = user_true_sids[u]
            beams = s4[u]  # (10, L) int
            n_coll = collision_count[true_sid]
            for k in TOP_K_LIST:
                beams_k = beams[:k]
                # R@K
                hits = sum(1 for b in beams_k if tuple(b.tolist()) == true_sid)
                opt = float(hits > 0)
                pess = hits / max(n_coll, 1)
                rand = pess  # random tie-break = pessimistic
                sum_opt[k] += opt
                sum_pess[k] += pess
                sum_rand[k] += rand
                # NDCG@K
                idcg = 1.0
                dcg_opt = sum(1.0 / np.log2(i + 2) for i, b in enumerate(beams_k) if tuple(b.tolist()) == true_sid)
                dcg_pess = sum((1.0 / max(n_coll, 1)) / np.log2(i + 2)
                               for i, b in enumerate(beams_k) if tuple(b.tolist()) == true_sid)
                sum_ndcg_opt[k] += dcg_opt / idcg
                sum_ndcg_pess[k] += dcg_pess / idcg
                sum_ndcg_rand[k] += dcg_pess / idcg  # same as pess
            n_scored += 1

        # Aggregate
        agg = {}
        for k in TOP_K_LIST:
            agg[f'r@{k}_opt'] = sum_opt[k] / n_scored
            agg[f'r@{k}_pess'] = sum_pess[k] / n_scored
            agg[f'r@{k}_rand'] = sum_rand[k] / n_scored
            agg[f'ndcg@{k}_opt'] = sum_ndcg_opt[k] / n_scored
            agg[f'ndcg@{k}_pess'] = sum_ndcg_pess[k] / n_scored
            agg[f'ndcg@{k}_rand'] = sum_ndcg_rand[k] / n_scored
        agg['collision_rate'] = float(coll_rate)
        agg['n_unique_sids'] = int(n_unique)
        agg['n_total_items'] = int(n_total)
        agg['n_users_scored'] = n_scored
        agg['mean_coll_size'] = float(np.mean(coll_sizes))
        agg['max_coll_size'] = int(max(coll_sizes))
        agg['sid_path'] = sid_path
        agg['stage4_path'] = s4_path
        agg['eval_json'] = eval_json

        print(f'  R@10 optimistic:  {agg["r@10_opt"]:.4f}')
        print(f'  R@10 pessimistic: {agg["r@10_pess"]:.4f}')
        print(f'  R@5  optimistic:  {agg["r@5_opt"]:.4f}')
        print(f'  R@5  pessimistic: {agg["r@5_pess"]:.4f}')
        all_results[algo] = agg

    # Save
    out_path = OUT_DIR / 'collision_aware_R10.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved → {out_path}')

    # Final table
    print('\n' + '=' * 110)
    print(f'{"Algo":<12} {"R@10_opt":<10} {"R@10_pess":<11} {"R@5_opt":<9} {"R@5_pess":<10} {"NDCG@10_opt":<12} {"CollRate":<9} {"MeanSize":<9}')
    print('-' * 110)
    for algo, agg in all_results.items():
        print(f'{algo:<12} {agg["r@10_opt"]:<10.4f} {agg["r@10_pess"]:<11.4f} '
              f'{agg["r@5_opt"]:<9.4f} {agg["r@5_pess"]:<10.4f} '
              f'{agg["ndcg@10_opt"]:<12.4f} {agg["collision_rate"]:<9.4f} {agg["mean_coll_size"]:<9.2f}')
    print('=' * 110)
    print(f'Elapsed: {time.time()-t0:.0f}s')

    # Compare optimistic vs existing R@10
    print('\n--- Existing R@10 (from eval json) vs collision-aware ---')
    existing_r10 = {
        'A_baseline': 0.0973,
        'B_mmq':      0.0898,
        'C_gsrq':     0.0857,
        'HRQ_v2200':  0.1336,
        'AQ_v2000':   0.1361,
    }
    print(f'{"Algo":<12} {"Existing":<10} {"Opt(ours)":<10} {"Match?":<8}')
    for algo, agg in all_results.items():
        opt = agg['r@10_opt']
        ex = existing_r10.get(algo, None)
        match = "✓" if ex is not None and abs(opt - ex) < 0.001 else "✗"
        print(f'{algo:<12} {ex if ex else "?":<10} {opt:<10.4f} {match}')


if __name__ == '__main__':
    main()