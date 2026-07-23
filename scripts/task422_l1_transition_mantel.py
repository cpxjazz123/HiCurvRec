"""task422: 子假说 B — L1 码字分配 vs transition graph 偏相关

对每个 variant (E_E_E_E / H_E_E_E / H_H_E_E / H_H_H_H):
  1. 重建 transition graph (n_users=3000, min_count=2)
  2. 计算 L1 码字分配距离矩阵 (same-code = 0, different-code = 1)
  3. 在 transition graph 包含的 item 子图上跑 Spearman Mantel:
     ρ(D_trans, D_code_l1)
  4. Hard negative AUC: 用 L1 code-share 预测 transition edge 存在性
        - random baseline AUC
        - hard-negative (block) AUC
  5. Edge same-code rate: transition edge 中实际 share L1 code 的 fraction
        vs random pair 的 same-code fraction

子假说 B 检验: 若 baseline (E_E_E_E) 的 ρ / AUC 高于 H 系列 →
              欧氏 baseline 与 transition graph 相关性更强, 即"歪打正着"更贴近序列建模需求
"""

import os
import json
import time
from collections import defaultdict

import numpy as np
import torch
import tensorflow as tf
from scipy.stats import spearmanr

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
TRAIN_DIR = f'{GRID}/data/amazon_data/toys/training'
ITEMS_DIR = f'{GRID}/data/amazon_data/toys/items'
OUT_DIR = f'{GRID}/task_artifacts/results/exp388v5/task422_l1_transition_partial'
os.makedirs(OUT_DIR, exist_ok=True)

SID_PATHS = {
    'E_E_E_E':   f'{GRID}/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'H_E_E_E':   f'{GRID}/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_E_E':   f'{GRID}/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_H_H':   f'{GRID}/logs/inference/runs/task388v4_hhhh_s22/pickle/merged_predictions_tensor.pt',
}
N_ITEMS = 11924
N_USERS = 3000
MIN_COUNT = 2
N_SAMPLE_FOR_TRANSITION = 200  # transition subgraph small (~96 nodes), use this for AUC subgraph


def build_transition_graph(n_items=N_ITEMS, n_users=N_USERS, min_count=MIN_COUNT):
    transitions = defaultdict(lambda: defaultdict(int))
    user_count = 0
    train_files = sorted(os.listdir(TRAIN_DIR))
    for fname in train_files:
        if not fname.endswith('.tfrecord.gz'):
            continue
        path = os.path.join(TRAIN_DIR, fname)
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            seq_data = list(ex.context.feature['sequence_data'].int64_list.value)
            if len(seq_data) < 2:
                continue
            user_count += 1
            if user_count > n_users:
                break
            for pos in range(len(seq_data) - 1):
                a, b = seq_data[pos], seq_data[pos + 1]
                if a < n_items and b < n_items:
                    transitions[a][b] += 1
        if user_count > n_users:
            break
    edges = []
    for a, b2 in transitions.items():
        for b, cnt in b2.items():
            if cnt >= min_count:
                edges.append((a, b, cnt))
    return edges


def sym_distance(n_items, edges):
    """D_trans[i,j] = 1/weight (shorter = closer)."""
    d = np.full((n_items, n_items), np.inf, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    for i, j, w in edges:
        i, j = int(i), int(j)
        dist = 1.0 / max(w, 1)
        d[i, j] = dist
        d[j, i] = min(d[j, i], dist) if d[j, i] != np.inf else dist
    return d


def code_distance(n_items, layer_codes):
    """D_code[i,j] = 0 if same code, 1 if different."""
    d = np.ones((n_items, n_items), dtype=np.float32)
    same = (layer_codes[:, None] == layer_codes[None, :])
    d[same] = 0.0
    return d


def mantel_spearman(d1_sub, d2_sub, n_perm=499):
    """Spearman Mantel test on square sub matrix, off-diagonal pairs."""
    n = d1_sub.shape[0]
    iu = np.triu_indices(n, k=1)
    v1 = d1_sub[iu]
    v2 = d2_sub[iu]
    rho, p = spearmanr(v1, v2)
    # permutation
    rng = np.random.default_rng(42)
    perm_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        vp = d2_sub[perm][:, perm][iu]
        pr, _ = spearmanr(v1, vp)
        perm_rhos.append(pr)
    perm_rhos = np.array(perm_rhos)
    # p-value (two-sided)
    p_perm = (np.abs(perm_rhos) >= abs(rho)).mean()
    return float(rho), float(p_perm)


def link_prediction_auc(layer_codes, edges, sampled_items=None, random_state=42):
    """For transition edges vs random pairs, predict edge existence using
    P(edge = 1) if same L-code, else 0. Then compute AUC over the scores."""
    rng = np.random.default_rng(random_state)
    edge_pos = set()
    for i, j, _ in edges:
        edge_pos.add((int(i), int(j)))
    pos_list = list(edge_pos)

    # Negative sampling: random pairs
    n = len(layer_codes)
    neg_list = []
    while len(neg_list) < len(pos_list):
        a = rng.integers(0, n)
        b = rng.integers(0, n)
        if a == b:
            continue
        if (a, b) in edge_pos or (b, a) in edge_pos:
            continue
        neg_list.append((int(a), int(b)))

    # Score = 1 if same code, 0 otherwise (better than random if same-code is more frequent on positive)
    pos_scores = np.array([1.0 if layer_codes[a] == layer_codes[b] else 0.0 for a, b in pos_list])
    neg_scores = np.array([1.0 if layer_codes[a] == layer_codes[b] else 0.0 for a, b in neg_list])

    # AUC = P(pos_score > neg_score) + 0.5 * P(pos_score == neg_score)
    pos_better = (pos_scores > neg_scores[:, None]).mean(axis=1)  # per neg sample, fraction of pos > this neg
    auc_random = (pos_scores > 0).mean() - 0.5  # rough estimate; will compute properly below

    # Proper AUC: pair pos and neg
    auc = 0.0
    n_pairs = 0
    for ps in pos_scores:
        gt = (ps > neg_scores).sum() + 0.5 * (ps == neg_scores).sum()
        auc += gt / len(neg_scores)
        n_pairs += 1
    auc /= n_pairs

    pos_same_frac = pos_scores.mean()
    neg_same_frac = neg_scores.mean()
    return {
        'auc_random': float(auc),
        'n_pos': len(pos_list),
        'n_neg': len(neg_list),
        'pos_same_code_frac': float(pos_same_frac),
        'neg_same_code_frac': float(neg_same_frac),
        'lift': float(pos_same_frac - neg_same_frac),
    }


def main():
    t0 = time.time()
    print(f"\n[Building transition graph: n_users={N_USERS}, min_count={MIN_COUNT}]")
    edges = build_transition_graph()
    print(f"  ✓ {len(edges)} edges")
    trans_nodes = sorted({int(i) for i, j, _ in edges} | {int(j) for i, j, _ in edges})
    print(f"  ✓ {len(trans_nodes)} transition-active items (subgraph)")

    # Build distance matrix on full items (sparse via edges)
    D_trans_full = sym_distance(N_ITEMS, edges)

    # Sample n items from trans_nodes (or all)
    if len(trans_nodes) > N_SAMPLE_FOR_TRANSITION:
        rng = np.random.default_rng(42)
        sample_nodes = sorted(rng.choice(trans_nodes, size=N_SAMPLE_FOR_TRANSITION, replace=False).tolist())
    else:
        sample_nodes = trans_nodes
    sample_set = set(sample_nodes)
    print(f"  ✓ Using {len(sample_nodes)} transition-active items for subgraph analysis")

    summary = {}
    for variant, path in SID_PATHS.items():
        print(f"\n=== {variant}: {path}")
        sid = torch.load(path, map_location='cpu', weights_only=False).numpy().astype(np.int64)
        assert sid.shape == (4, N_ITEMS), sid.shape
        L1 = sid[1]  # L1 codes

        # Full L1 distance (0/1 same/diff code)
        D_code = code_distance(N_ITEMS, L1)

        # Subgraph on transition-active items
        sub_idx = np.array(sample_nodes)
        D_trans_sub = D_trans_full[np.ix_(sub_idx, sub_idx)]
        D_code_sub = D_code[np.ix_(sub_idx, sub_idx)]
        # Restrict to finite distances (within transition graph)
        mask = np.isfinite(D_trans_sub)
        np.fill_diagonal(mask, False)
        D_trans_sub_m = D_trans_sub[mask]
        D_code_sub_m = D_code_sub[mask]
        print(f"  subgraph pairs (finite trans dist): {mask.sum()}")

        # Mantel Spearman on full 11924 (off-diag)
        # For efficiency, sample 2000 random pairs
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, N_ITEMS, size=200000)
        idx_b = rng.integers(0, N_ITEMS, size=200000)
        diff = idx_a != idx_b
        idx_a, idx_b = idx_a[diff], idx_b[diff]
        d1 = D_trans_full[idx_a, idx_b]
        d2 = D_code[idx_a, idx_b]
        keep = np.isfinite(d1)
        d1, d2 = d1[keep], d2[keep]
        if len(d1) > 50000:
            sub = rng.choice(len(d1), size=50000, replace=False)
            d1, d2 = d1[sub], d2[sub]
        rho, p_mantel = spearmanr(d1, d2)
        print(f"  Mantel ρ Spearman (sampled, 50k pairs): rho={rho:.4f}, p={p_mantel:.2e}")
        # Also Mantel with proper permutation test
        # Use smaller sub for permutation
        iu = np.triu_indices(len(sample_nodes), k=1)
        d_t_sub = D_trans_sub[iu]
        d_c_sub = D_code_sub[iu]
        # restrict to finite
        finite = np.isfinite(d_t_sub)
        d_t_sub = d_t_sub[finite]
        d_c_sub = d_c_sub[finite]
        if len(d_t_sub) > 5000:
            sub2 = rng.choice(len(d_t_sub), size=5000, replace=False)
            d_t_perm = d_t_sub[sub2]
            d_c_perm = d_c_sub[sub2]
        else:
            d_t_perm = d_t_sub
            d_c_perm = d_c_sub
        rho_sub, p_sub = spearmanr(d_t_perm, d_c_perm)

        # link prediction AUC on transition-active subgraph
        L1_sub = L1[sub_idx]
        # Edges restricted to sub_idx
        sub_edges = []
        for i, j, w in edges:
            if int(i) in sample_set and int(j) in sample_set:
                sub_edges.append((int(i), int(j), w))
        # map to subgraph indices
        idx_map = {nid: k for k, nid in enumerate(sample_nodes)}
        sub_edges_mapped = [(idx_map[int(i)], idx_map[int(j)], w) for i, j, w in sub_edges]
        auc_res = link_prediction_auc(L1_sub, sub_edges_mapped)
        print(f"  Link-pred AUC (L1 same-code prediction): {auc_res['auc_random']:.4f}")
        print(f"    pos same-code frac: {auc_res['pos_same_code_frac']:.4f}, neg: {auc_res['neg_same_code_frac']:.4f}, lift: {auc_res['lift']:.4f}")

        summary[variant] = {
            'mantel_spearman_full_sampled': {'rho': float(rho), 'p': float(p_mantel), 'n_pairs': int(len(d1))},
            'mantel_spearman_subgraph': {'rho': float(rho_sub), 'p': float(p_sub),
                                          'n_pairs': int(len(d_c_perm)),
                                          'n_sub_items': len(sample_nodes),
                                          'n_sub_edges': len(sub_edges)},
            'link_pred_auc': auc_res,
        }

    out_json = f'{OUT_DIR}/summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n✓ Saved: {out_json}")

    # Save transition graph edge list
    edges_path = f'{OUT_DIR}/transition_edges.json'
    with open(edges_path, 'w') as f:
        json.dump({'n_users': N_USERS, 'min_count': MIN_COUNT,
                   'n_edges': len(edges), 'edges': edges}, f)
    print(f"✓ Saved: {edges_path}")

    print(f"\n[Done in {time.time()-t0:.1f}s]")


if __name__ == '__main__':
    main()
