"""task422b: 子假说 B 二次分析 + bootstrap CI

主要补:
  1. Mantel in transition-subgraph (n=370 items, ~108 finite pairs), 999 置换
  2. link-pred AUC bootstrap CI 200 reps
  3. pair-level lift 与统计效力
"""

import os
import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/task_artifacts/results/exp388v5/task422_l1_transition_partial'
os.makedirs(OUT_DIR, exist_ok=True)

# Load transition edges
EDGE_PATH = f'{OUT_DIR}/transition_edges.json'
edges_data = json.load(open(EDGE_PATH))
edges = [tuple(e) for e in edges_data['edges']]
print(f"Transition graph: {len(edges)} edges")

SID_PATHS = {
    'E_E_E_E':   f'{GRID}/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'H_E_E_E':   f'{GRID}/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_E_E':   f'{GRID}/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_H_H':   f'{GRID}/logs/inference/runs/task388v4_hhhh_s22/pickle/merged_predictions_tensor.pt',
}
N_ITEMS = 11924

# Subgraph (transition-active items)
trans_nodes = sorted({int(i) for i, j, _ in edges} | {int(j) for i, j, _ in edges})
sub_idx = np.array(trans_nodes)
sample_set = set(trans_nodes)
idx_map = {nid: k for k, nid in enumerate(trans_nodes)}
print(f"Subgraph: {len(trans_nodes)} items")


def code_distance_binary(layer_codes):
    """Returns (n_sub, n_sub) binary matrix: 0 if same code, 1 if different."""
    same = (layer_codes[:, None] == layer_codes[None, :])
    d = np.where(same, 0.0, 1.0).astype(np.float32)
    return d


def sym_distance_matrix(n, edges_sub_mapped):
    INF = 999.0
    d = np.full((n, n), INF, dtype=np.float32)
    np.fill_diagonal(d, 0.0)
    for i, j, w in edges_sub_mapped:
        dist = 1.0 / max(w, 1)
        d[i, j] = dist
        d[j, i] = dist
    return d


def mantel_permutation(d1_sub, d2_sub, n_perm=999, rng_seed=42):
    n = d1_sub.shape[0]
    iu = np.triu_indices(n, k=1)
    v1 = d1_sub[iu]
    v2 = d2_sub[iu]
    keep = np.isfinite(v1)
    v1, v2 = v1[keep], v2[keep]
    rho_obs, _ = spearmanr(v1, v2)
    rng = np.random.default_rng(rng_seed)
    perm_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        v2p = d2_sub[perm][:, perm][iu][keep]
        rp, _ = spearmanr(v1, v2p)
        perm_rhos.append(rp)
    perm_rhos = np.array(perm_rhos)
    p_two = (np.abs(perm_rhos) >= abs(rho_obs)).mean()
    return float(rho_obs), float(p_two), len(v1)


def link_pred_auc_bootstrap(layer_codes_sub, edges_sub_mapped, n_bootstrap=200, neg_per_pos=5, seed=42):
    """Bootstrap CI for AUC of L1-code-share predictor on transition edges."""
    edge_set = {(int(i), int(j)) for i, j, _ in edges_sub_mapped}
    edge_set |= {(int(j), int(i)) for i, j, _ in edges_sub_mapped}
    pos_list = [(int(i), int(j)) for i, j, _ in edges_sub_mapped]
    pos_scores_full = np.array([1.0 if layer_codes_sub[a] == layer_codes_sub[b] else 0.0 for a, b in pos_list])
    pos_same_frac_obs = pos_scores_full.mean()

    n = len(layer_codes_sub)
    rng = np.random.default_rng(seed)

    # Generate a fixed pool of negative pairs (10x positive)
    neg_pool = []
    while len(neg_pool) < 10 * len(pos_list):
        a = rng.integers(0, n)
        b = rng.integers(0, n)
        if a == b:
            continue
        if (a, b) in edge_set:
            continue
        neg_pool.append((a, b))
    neg_scores_full = np.array([1.0 if layer_codes_sub[a] == layer_codes_sub[b] else 0.0 for a, b in neg_pool])

    # Point AUC
    gt_obs = (pos_scores_full[:, None] > neg_scores_full).sum(axis=1) + 0.5 * (pos_scores_full[:, None] == neg_scores_full).sum(axis=1)
    auc_obs = gt_obs.mean() / len(neg_pool)

    aucs = []
    for b in range(n_bootstrap):
        # resample pos with replacement
        idx_pos = rng.integers(0, len(pos_list), size=len(pos_list))
        ps = pos_scores_full[idx_pos]
        idx_neg = rng.integers(0, len(neg_pool), size=neg_per_pos * len(pos_list))
        ns = neg_scores_full[idx_neg]
        gt = (ps[:, None] > ns).sum(axis=1) + 0.5 * (ps[:, None] == ns).sum(axis=1)
        auc_b = gt.mean() / len(ns)
        aucs.append(auc_b)
    aucs = np.array(aucs)
    ci_low, ci_high = np.quantile(aucs, [0.025, 0.975])
    return {
        'auc': float(auc_obs),
        'auc_ci_low': float(ci_low),
        'auc_ci_high': float(ci_high),
        'n_pos': len(pos_list),
        'n_neg_pool': len(neg_pool),
        'pos_same_frac': float(pos_same_frac_obs),
        'neg_same_frac': float(neg_scores_full.mean()),
        'lift': float(pos_same_frac_obs - neg_scores_full.mean()),
    }


def main():
    summary = {}
    for variant, path in SID_PATHS.items():
        print(f"\n=== {variant}")
        sid = torch.load(path, map_location='cpu', weights_only=False).numpy().astype(np.int64)
        L1 = sid[1]
        L1_sub = L1[sub_idx]
        L1_sub_dist = code_distance_binary(L1_sub)

        # Remap edges to subgraph indices
        edges_mapped = [(idx_map[int(i)], idx_map[int(j)], int(w)) for i, j, w in edges
                        if int(i) in sample_set and int(j) in sample_set]
        # unique pairs
        edges_mapped = list({(i, j) for i, j, _ in edges_mapped} | {((j, i)) for i, j, _ in edges_mapped})
        edges_mapped_with_w = []
        seen = set()
        for i, j, w in [(idx_map[int(i)], idx_map[int(j)], int(w)) for i, j, w in edges
                         if int(i) in sample_set and int(j) in sample_set]:
            if (i, j) not in seen:
                edges_mapped_with_w.append((i, j, w))
                seen.add((i, j))
                seen.add((j, i))

        # Build D_trans on subgraph
        D_trans_sub = sym_distance_matrix(len(sub_idx), edges_mapped_with_w)

        # Mantel Spearman with permutation test
        rho, p_perm, n_pairs = mantel_permutation(D_trans_sub, L1_sub_dist, n_perm=999)
        print(f"  Mantel Spearman (n_pairs={n_pairs}): rho={rho:.4f}, p_perm={p_perm:.4f}")

        # AUC bootstrap
        auc_res = link_pred_auc_bootstrap(L1_sub, edges_mapped_with_w, n_bootstrap=200, neg_per_pos=5)
        print(f"  AUC: {auc_res['auc']:.4f} [{auc_res['auc_ci_low']:.4f}, {auc_res['auc_ci_high']:.4f}]")
        print(f"    pos same={auc_res['pos_same_frac']:.4f} neg same={auc_res['neg_same_frac']:.4f} lift={auc_res['lift']:.4f}")

        summary[variant] = {
            'mantel_subgraph': {'rho': float(rho), 'p_perm': float(p_perm), 'n_pairs': int(n_pairs)},
            'auc_bootstrap': auc_res,
        }

    out_json = f'{OUT_DIR}/summary_bootstrap.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved: {out_json}")


if __name__ == '__main__':
    t = time.time()
    main()
    print(f"\n[Done in {time.time()-t:.1f}s]")
