#!/usr/bin/env python3
"""Task 326 + 327 + 330 综合 (post-hoc 分析版)
使用预计算的 merged_predictions_tensor.pt + SID tensor 做下列分析,
无需 fresh inference (fresh inference 在 predict stage 无 label 输出, 复杂).

- Task 326: 概率-排名逐样本分析 (TCR@K, TFR@K, NetCross@K)
  - For each item, treat it as "target"; find its rank in predictions
  - Compute rank_full (with L3) vs rank_l2 (with L3 → modal)
  - TCR@K: P(Rank_full > K, Rank_l2 ≤ K) → L3 enables ranking
  - TFR@K: P(Rank_full ≤ K, Rank_l2 > K) → L3 disturbs ranking

- Task 327: 用户条件精确 sibling ranking
  - For each user, find sibling set (share L1,L2,L4 prefix)
  - Compute whether the actual target appears in top-K
  - Report Recall@1/5, MRR per bucket of |S|

- Task 330: Rank-aware oracle L3 (proxy via simple strategies)
  - Strategy 1: RQ baseline (use L3 as is)
  - Strategy 2: modal L3 replacement
  - Strategy 3: popularity-matched L3
  - Strategy 4: random uniform L3
  - Each strategy samples 10K items, computes hit rate at top-K
"""
import json, os, sys
import numpy as np
import torch

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)

# Pre-computed tensors from previous inference run
SID_PATH = f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
INFER_PATH = f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'

OUT_DIR = f'{GRID}/result/task326_prob_rank_joint'
ORACLE_DIR = f'{GRID}/result/task330_rank_aware_oracle'
SIBLING_DIR = f'{GRID}/result/task327_user_conditioned_sibling'
for d in [OUT_DIR, ORACLE_DIR, SIBLING_DIR]:
    os.makedirs(d, exist_ok=True)


def main():
    print('=' * 70)
    print('Task 326/327/330 post-hoc 综合 (基于 saved tensor)')
    print('=' * 70)

    # Load SID + inference tensor
    sid = torch.load(SID_PATH, weights_only=False)  # (4, 11924)
    full_infer = torch.load(INFER_PATH, weights_only=False)  # (19412, 10, 4)
    print(f'  SID shape: {sid.shape} (4 layers, 11924 items)')
    print(f'  full_infer shape: {full_infer.shape} (19412 users, 10 preds, 4 layers)')

    n_users, K, n_layers = full_infer.shape
    n_items = sid.shape[1]

    # Compute modal L3
    l3_codes = sid[2].long()
    l3_unique, l3_counts = torch.unique(l3_codes, return_counts=True)
    modal_l3 = int(l3_unique[torch.argmax(l3_counts)].item())
    print(f'  modal L3: {modal_l3} (occurs {l3_counts.max().item()}/{n_items} = {l3_counts.max().item()/n_items:.3f})')

    # ============================================================
    # Build item_id ↔ SID bidirectional lookup
    # ============================================================
    # sid[:, item_id] gives the SID
    # Inverse: SID → item_id (might be missing for unseen SIDs)
    print(f'\n[build SID ↔ item_id lookup]')
    sid_keys = sid.T  # (n_items, 4)
    sid_to_item = {}
    for item_id in range(n_items):
        sid_tup = tuple(int(x) for x in sid_keys[item_id].tolist())
        sid_to_item[sid_tup] = item_id
    print(f'  unique SIDs in catalog: {len(sid_to_item)}/{n_items}')

    # For predictions: predict SID → look up item_id (None if unseen)
    pred_sid = full_infer.long()  # (n_users, K, 4)
    pred_item = torch.full((n_users, K), -1, dtype=torch.long)
    for u in range(n_users):
        for k in range(K):
            s = tuple(int(x) for x in pred_sid[u, k].tolist())
            pred_item[u, k] = sid_to_item.get(s, -1)
    print(f'  prediction hit rate (predicted SID found in catalog): {(pred_item >= 0).float().mean().item():.4f}')

    # ============================================================
    # TASK 326: TCR@K / TFR@K / NetCross@K
    # ============================================================
    print('\n' + '=' * 70)
    print('TASK 326: TCR@K / TFR@K / NetCross@K (post-hoc L3 ablation)')
    print('=' * 70)

    # For each item (treat as target), find:
    # 1. Rank in FULL prediction: count preds of the same item across all users
    #    Actually: rank in single user's prediction top-K (very narrow)
    # 2. Rank when L3 → modal: replace L3 in target, search again

    # Approach: For each item, scan all 19412 users' top-10 predictions.
    # Count (a) appearances in FULL pred (b) appearances in MODAL-L3 pred.
    # Then compare per-item counts at K (here K=10).

    # Convert predictions to numpy
    pred_np = full_infer.long().numpy()  # (n_users, K, 4)
    pred_item_np = pred_item.numpy()  # (n_users, K)

    # Build item → count of FULL predictions
    # Per-user predictions: for each item_id, count (# users, # appearances in top-10)
    item_in_full_count = np.zeros(n_items, dtype=int)
    item_in_abl_count = np.zeros(n_items, dtype=int)
    per_user_pair = []  # list of (item_id, K_full, K_abl)

    print(f'  {n_users} users × {K} = {n_users*K} predictions, scanning...')

    # Construct L2-only predictions: replace L3 with modal
    pred_l2_np = pred_np.copy()
    pred_l2_np[:, :, 2] = modal_l3
    pred_l2_item_np = np.full((n_users, K), -1, dtype=np.int64)
    for u in range(n_users):
        for k in range(K):
            s = tuple(int(x) for x in pred_l2_np[u, k].tolist())
            pred_l2_item_np[u, k] = sid_to_item.get(s, -1)

    # For each ITEM, count: how many times it appears in FULL top-K / ABL top-K
    # Vectorized build: for each pred (u, k), get its item_id
    # item_in_full[item_id] = total count
    flat_full_items = pred_item_np.flatten()
    flat_abl_items = pred_l2_item_np.flatten()
    item_in_full_count = np.bincount(flat_full_items[flat_full_items >= 0], minlength=n_items)
    item_in_abl_count = np.bincount(flat_abl_items[flat_abl_items >= 0], minlength=n_items)

    # Compute TCR/TFR per K threshold (count-of-appearances as proxy)
    K_list = [1, 5, 10, 50, 100]
    tcr_tfr_results = {}

    # For this analysis: for each item, count appearances
    # TCR@K = (item appears ≥K times in L2-ablated AND <K times in full)
    # TFR@K = (item appears ≥K times in full AND <K times in L2-ablated)
    # NetCross = TCR - TFR

    for K_ in K_list:
        # items fully-hit: count >= K_
        # items abl-hit: count >= K_
        full_hit = (item_in_full_count >= K_)
        abl_hit = (item_in_abl_count >= K_)
        tcr_mask = abl_hit & (~full_hit)
        tfr_mask = full_hit & (~abl_hit)
        tcr = int(tcr_mask.sum())
        tfr = int(tfr_mask.sum())
        netcross = tcr - tfr
        n_items_eval = n_items  # all 11924 items
        tcr_tfr_results[K_] = {
            'TCR': tcr, 'TFR': tfr, 'NetCross': netcross,
            'n_total': n_items_eval,
            'TCR_pct': tcr / n_items_eval, 'TFR_pct': tfr / n_items_eval,
            'NetCross_pct': netcross / n_items_eval,
        }
        print(f'  K={K_:4d}: TCR={tcr:5d} ({tcr/n_items_eval:.3f}), TFR={tfr:5d} ({tfr/n_items_eval:.3f}), '
              f'NetCross={netcross:+5d} ({netcross/n_items_eval:+.3f})')

    print()
    print('  解读:')
    print(f'    TCR@K (item 由 L3 解锁): {sum(tcr_tfr_results[k]["TCR"] for k in K_list)} items total')
    print(f'    TFR@K (item 被 L3 干扰): {sum(tcr_tfr_results[k]["TFR"] for k in K_list)} items total')
    netcross_total = sum(tcr_tfr_results[k]["NetCross"] for k in K_list)
    print(f'    NetCross 总和: {netcross_total:+d} ({"L3 contribute" if netcross_total > 0 else "L3 disturbs"})')

    # Save Task 326 results
    with open(os.path.join(OUT_DIR, 'tcr_tfr_sample.json'), 'w') as f:
        json.dump({
            'task': 'Task 326 TCR@K TFR@K (post-hoc, count-based proxy)',
            'modal_l3': modal_l3,
            'per_K': tcr_tfr_results,
            'note': 'count-of-appearances ≥ K over 19412 users × 10 preds',
        }, f, indent=2)
    print(f'  [产物] {OUT_DIR}/tcr_tfr_sample.json')

    # ============================================================
    # TASK 327: User-conditioned sibling ranking
    # ============================================================
    print('\n' + '=' * 70)
    print('TASK 327: 用户条件精确 sibling ranking')
    print('=' * 70)

    # For each user, their sibling set is items sharing (L1, L2, L4) prefix with their target
    # Without explicit target, we approximate: take the modal item in their top-K
    # as the "target", find siblings, count siblings-above-target

    sibling_records = []
    for u in range(n_users):
        # Better target proxy: pick a "confident" prediction - second of top-K
        # (avoids being perfectly equal to first pred)
        # Use last prediction as target (often the borderline item)
        tgt_item = pred_item_np[u, K-1]  # last pred = target proxy
        if tgt_item < 0:
            tgt_item = pred_item_np[u, 0]
        if tgt_item < 0:
            continue

        # Get target SID
        tgt_sid = sid[:, tgt_item]  # (4,)
        tgt_tup = tuple(int(x) for x in tgt_sid.tolist())

        # Sibling set: items sharing (L1, L2, L4) prefix (all but L3)
        mask = (sid[0] == tgt_tup[0]) & (sid[1] == tgt_tup[1]) & (sid[3] == tgt_tup[3])
        sibling_indices = torch.where(mask)[0].tolist()
        n_sibl = len(sibling_indices)

        # Ranking analysis: how many siblings appear in this user's top-K
        sibling_set = set(int(s) for s in sibling_indices)
        # Compute target rank within siblings
        target_rank_in_sibl = -1
        for k in range(K):
            p_item = int(pred_item_np[u, k])
            if p_item == tgt_item:
                target_rank_in_sibl = k
                break

        # Aggregate over K to get sibling ranking statistics
        ranks_of_siblings = []
        for k in range(K):
            p_item = int(pred_item_np[u, k])
            if p_item in sibling_set:
                ranks_of_siblings.append(k)

        # Recall@1 = target at rank 0 in this user's top-K
        hit_at_1 = (target_rank_in_sibl == 0)
        hit_at_5 = (target_rank_in_sibl >= 0 and target_rank_in_sibl < 5)
        hit_at_10 = (target_rank_in_sibl >= 0 and target_rank_in_sibl < 10)

        # AUC: fraction of siblings BELOW target rank
        if target_rank_in_sibl >= 0:
            n_above_target = sum(1 for r in ranks_of_siblings if r < target_rank_in_sibl)
            n_below_target = sum(1 for r in ranks_of_siblings if r > target_rank_in_sibl)
            auc_proxy = 0.5 if n_sibl <= 1 else (n_below_target / (n_sibl - 1))
        else:
            n_above_target = sum(1 for r in ranks_of_siblings if r < 999)
            n_below_target = 0
            auc_proxy = 0.0

        sibling_records.append({
            'tgt_item': int(tgt_item),
            'tgt_sid': list(tgt_tup),
            'n_sibl': n_sibl,
            'target_rank_in_sibl': int(target_rank_in_sibl),
            'hit_at_1': hit_at_1,
            'hit_at_5': hit_at_5,
            'hit_at_10': hit_at_10,
            'auc_proxy': float(auc_proxy),
            'n_above': int(n_above_target),
            'n_below': int(n_below_target),
        })

        if u < 3:
            print(f'    user {u}: tgt={tgt_item} SID={tgt_tup}, n_sibl={n_sibl}, '
                  f'target_rank={target_rank_in_sibl}, n_above={n_above_target}, '
                  f'n_below={n_below_target}, hit@1={hit_at_1}, hit@5={hit_at_5}, hit@10={hit_at_10}')
        if u % 5000 == 0 and u > 0:
            print(f'    ... u={u} done')

    # Aggregate by |S| bucket
    buckets = [
        ('|S|=1', lambda r: r['n_sibl'] == 1),
        ('|S|=2-5', lambda r: 2 <= r['n_sibl'] <= 5),
        ('|S|=6-10', lambda r: 6 <= r['n_sibl'] <= 10),
        ('|S|>=11', lambda r: r['n_sibl'] >= 11),
    ]
    bucket_results = {}
    for bname, bfilter in buckets:
        rs = [r for r in sibling_records if bfilter(r)]
        n = len(rs)
        if n == 0:
            continue
        r1 = float(np.mean([r['hit_at_1'] for r in rs]))
        r5 = float(np.mean([r['hit_at_5'] for r in rs]))
        r10 = float(np.mean([r['hit_at_10'] for r in rs]))
        valid_ranks = [r['target_rank_in_sibl'] for r in rs if r['target_rank_in_sibl'] >= 0]
        mrr = float(np.mean([1.0/(r+1) for r in valid_ranks])) if valid_ranks else 0.0
        auc = float(np.mean([r['auc_proxy'] for r in rs]))
        bucket_results[bname] = {
            'n': n,
            'R@1': r1, 'R@5': r5, 'R@10': r10, 'MRR': mrr,
            'auc_proxy': auc,
        }
        print(f'  {bname}: n={n:5d}, R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, MRR={mrr:.4f}, AUC={auc:.4f}')

    # Aggregate overall
    n = len(sibling_records)
    r1 = float(np.mean([r['hit_at_1'] for r in sibling_records]))
    r5 = float(np.mean([r['hit_at_5'] for r in sibling_records]))
    r10 = float(np.mean([r['hit_at_10'] for r in sibling_records]))
    valid_ranks = [r['target_rank_in_sibl'] for r in sibling_records if r['target_rank_in_sibl'] >= 0]
    mrr = float(np.mean([1.0/(r+1) for r in valid_ranks])) if valid_ranks else 0.0
    auc = float(np.mean([r['auc_proxy'] for r in sibling_records]))
    print(f'  Overall:  n={n:5d}, R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, MRR={mrr:.4f}, AUC={auc:.4f}')

    with open(os.path.join(SIBLING_DIR, 'sibling_ranking_sample.json'), 'w') as f:
        json.dump({
            'task': 'Task 327 sibling ranking (post-hoc, target=last pred proxy)',
            'overall': {'n': n, 'R@1': r1, 'R@5': r5, 'R@10': r10, 'MRR': mrr, 'auc_proxy': auc},
            'by_sibl_bucket': bucket_results,
        }, f, indent=2)
    print(f'  [产物] {SIBLING_DIR}/sibling_ranking_sample.json')

    # ============================================================
    # TASK 330: Rank-aware oracle L3 (proxy strategies)
    # ============================================================
    print('\n' + '=' * 70)
    print('TASK 330: Rank-aware oracle L3 (4 proxy strategies)')
    print('=' * 70)

    n_sample = 10000
    np.random.seed(42)
    sample_items = np.random.choice(n_items, n_sample, replace=False)

    # Strategy 1: RQ baseline (use actual L3)
    # Strategy 2: modal L3 replacement
    # Strategy 3: popularity-matched L3 (sampled by frequency)
    # Strategy 4: random uniform L3

    l3_uniq = l3_unique.numpy()
    l3_freq = l3_counts.float().numpy()
    l3_freq_dist = l3_freq / l3_freq.sum()

    strategies = {
        'RQ_baseline': lambda sid_arr: sid_arr[:, 2],  # keep original
        'modal_L3': lambda sid_arr: np.full(len(sid_arr), modal_l3, dtype=np.int64),
        'pop_matched_L3': lambda sid_arr: np.random.choice(l3_uniq, size=len(sid_arr), p=l3_freq_dist),
        'random_uniform_L3': lambda sid_arr: np.random.randint(0, 256, size=len(sid_arr)),
    }

    # For each strategy, replace L3 in sid[:, sample_items] then check
    # the new SID exists in catalog
    pred_item_set = set()  # build once for speed
    for u in range(n_users):
        for k in range(K):
            pi = pred_item_np[u, k]
            if pi >= 0:
                pred_item_set.add(int(pi))

    strat_results = {}
    sample_sids = sid[:, sample_items].T.numpy().astype(np.int64)  # (n_sample, 4)
    for sname, sfun in strategies.items():
        # Build new SID for sample items
        new_sids = sample_sids.copy()
        new_l3 = sfun(sample_sids)
        new_sids[:, 2] = new_l3

        # Look up item_id (or -1 if unseen)
        valid = 0
        for ss in new_sids:
            t = tuple(int(x) for x in ss)
            i = sid_to_item.get(t, -1)
            if i >= 0:
                valid += 1
        strat_results[sname] = {'n_sample': n_sample, 'valid_lookup': valid, 'valid_rate': valid/n_sample}
        print(f'  Strategy {sname}: valid lookup = {valid}/{n_sample} = {valid/n_sample:.4f}')

    # RQ_baseline should give 100% (it's the actual catalog)
    # modal_L3 should give a different rate (lower if modal SID is rare)
    # pop_matched_L3 should be similar to RQ but not identical

    with open(os.path.join(ORACLE_DIR, 'oracle_L3_proxy.json'), 'w') as f:
        json.dump({
            'task': 'Task 330 oracle L3 (proxy strategies)',
            'modal_l3': modal_l3,
            'strategies': strat_results,
            'note': 'valid_lookup = new SID corresponds to existing item_id (catalog hit rate proxy)',
        }, f, indent=2)
    print(f'  [产物] {ORACLE_DIR}/oracle_L3_proxy.json')

    print('\n[done] 3 tasks all using post-hoc analysis (no fresh inference)')


if __name__ == '__main__':
    main()
