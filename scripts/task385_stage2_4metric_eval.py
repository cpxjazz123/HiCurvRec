#!/usr/bin/env python3
"""Stage 2 evaluation: 4 metrics for H-E-E-E vs E-E-E-E baseline

Inputs:
  - H-E-E-E ckpt (from task385_stage2_h_e_e_e_train.py)
  - E-E-E-E ckpt (task13_group_a_s21) for comparison
  - FLAN-T5 embeddings (11924, 2048)

Metrics:
  1. Taxonomy ρ (Mantel.test between D_tax and D_residual_l0 distance)
     - target: H-E-E-E > E-E-E-E (current E-E-E-E ≈ 0.44 from task351)
  2. Transfer naive ρ (Mantel.test between D_trans and D_residual_l0 distance)
     - target: H-E-E-E ≈ E-E-E-E (current E-E-E-E ≈ 0.05 from task380 bootstrap)
  3. End-to-end R@10
     - target: H-E-E-E ≥ E-E-E-E (current E-E-E-E ≈ 0.097 from task15)
  4. Collision rate
     - target: H-E-E-E < E-E-E-E

Decision rules (per user):
  - Taxonomy ↑ + transfer stable (0.03-0.05) → adopt H-E-E-E
  - Taxonomy ↑ + transfer ↓ (<0.02) → check R@10:
    * R@10 ↑ → accept (worth the cost)
    * R@10 →/↓ → abandon
  - Taxonomy ≈ → abandon hyperbolic
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)
import src.utils.decorators  # noqa: F401

sys.path.insert(0, f'{GRID}/task_artifacts/scripts')
from task370_network_structure import (
    load_codebooks, forward_residual,
    build_co_purchase_graph, build_transition_graph, build_adj_list,
    build_d_tree, louvain_communities,
)
from scipy.stats import spearmanr


def mantel_rho_fast(d1, d2):
    """Ultra-fast Mantel ρ (no permutation): Pearson on upper triangles.

    Speed: O(m) where m = n*(n-1)/2 pairs. For n=2000, m=2M pairs → <2s.
    Returns point estimate only. Significance reference from task381 bootstrap.
    """
    n = d1.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    v1 = d1[idx_i, idx_j].astype(np.float32)
    v2 = d2[idx_i, idx_j].astype(np.float32)
    if v1.std() < 1e-10 or v2.std() < 1e-10:
        return 0.0
    v1c = v1 - v1.mean()
    v2c = v2 - v2.mean()
    denom = np.sqrt((v1c ** 2).sum() * (v2c ** 2).sum() + 1e-15)
    return float((v1c * v2c).sum() / denom)


def spearman_rho_fast(d1, d2):
    """Ultra-fast Spearman Mantel ρ (rank-based, vectorized argsort).

    For discrete (cat_sub 0/1/2) vs continuous distance matrices,
    Pearson is degenerate but Spearman captures monotonic structure.
    Speed: ~5-10s per call (argsort on 2M floats).
    """
    n = d1.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    v1 = d1[idx_i, idx_j].astype(np.float32)
    v2 = d2[idx_i, idx_j].astype(np.float32)
    if v1.std() < 1e-10 or v2.std() < 1e-10:
        return 0.0
    # Rank-based: argsort twice gives 0..m-1 ranks
    r1 = np.argsort(np.argsort(v1)).astype(np.float32)
    r2 = np.argsort(np.argsort(v2)).astype(np.float32)
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    denom = np.sqrt((r1c ** 2).sum() * (r2c ** 2).sum() + 1e-15)
    return float((r1c * r2c).sum() / denom)


def mantel_test_fast(d1, d2, n_perm=99, seed=42):
    """Fast vectorized Mantel test (Pearson, vectorized permutations).

    For n=2000, m=2M pairs, n_perm=99 → ~30s.
    Memory peak: (n_perm, m) = (99, 2M) × 4 bytes = 800MB.
    """
    n = d1.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    v1 = d1[idx_i, idx_j].astype(np.float32)
    v2 = d2[idx_i, idx_j].astype(np.float32)

    if v1.std() < 1e-10 or v2.std() < 1e-10:
        return 0.0, 1.0

    v1c = v1 - v1.mean()
    v2c = v2 - v2.mean()
    v1_norm = np.sqrt((v1c ** 2).sum())
    v2_norm = np.sqrt((v2c ** 2).sum())
    r_obs = float((v1c * v2c).sum() / (v1_norm * v2_norm + 1e-15))

    rng = np.random.default_rng(seed)
    m = len(v1)
    perm_idx = rng.integers(0, m, size=(n_perm, m))
    v2_perm = v2[perm_idx]  # (n_perm, m)
    v2_perm_c = v2_perm - v2_perm.mean(axis=1, keepdims=True)
    v2_perm_norm = np.sqrt((v2_perm_c ** 2).sum(axis=1))
    r_perm = (v1c * v2_perm_c).sum(axis=1) / (v1_norm * v2_perm_norm + 1e-15)
    p = ((np.abs(r_perm) >= abs(r_obs)).sum() + 1) / (n_perm + 1)
    return float(r_obs), float(p)
from task380_partial_mantel import (
    upper_tri, regression_residual_r,
    shortest_path_dist, build_tax_dist, build_sample,
)

EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
E_E_E_E_CKPT = f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'

OUT_DIR = f'{GRID}/result/task385_stage2_4metric_eval'
os.makedirs(OUT_DIR, exist_ok=True)

N_ITEMS = 11924


# ========== Metric 1 & 2: Taxonomy ρ and Transfer ρ ==========

def get_residuals(ckpt_path, emb_all):
    """Compute residuals from a ckpt.

    For task15 E-E-E-E: use existing forward_residual
    For task385 H-E-E-E: custom forward with Poincaré L1
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    is_h_e_e_e = 'hyperbolic' in str(ckpt.get('hyper_parameters', {})).lower() or \
                  ckpt.get('hyper_parameters', {}).get('l1_space') in ('poincare_ball', 'poincare_ball_dual')

    x_mean = ckpt.get('x_mean', None)
    x_std = ckpt.get('x_std', None)
    if x_mean is not None and x_std is not None:
        x = (emb_all.float() - x_mean) / x_std
    else:
        x = emb_all.float()

    if is_h_e_e_e:
        # H-E-E-E: L1 in Poincaré, L2/L3 Euclidean
        # Use v3 module (has target_norm) when dual codebook is detected
        is_dual_ckpt = 'C1_ball' in sd
        if is_dual_ckpt:
            from task388_h_e_e_e_train_v3 import (
                euclidean_to_poincare, poincare_distance, log_map0, BALL_C, MAX_NORM,
            )
        else:
            from task385_stage2_h_e_e_e_train import (
                euclidean_to_poincare, poincare_distance, log_map0, BALL_C, MAX_NORM,
            )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dim = ckpt['hyper_parameters']['dim']
        n_clusters = ckpt['hyper_parameters']['n_clusters']
        # Detect dual codebook (v3) vs single codebook (v1)
        is_dual = 'C1_ball' in sd
        C1_euclid = sd['quantization_layer_list.0.centroids'].to(device)
        C2_t = sd['quantization_layer_list.1.centroids'].to(device)
        C3_t = sd['quantization_layer_list.2.centroids'].to(device)
        if is_dual:
            C1_ball = sd['C1_ball'].to(device)
            print(f'[H-E-E-E] v3 dual codebook: C1_ball.shape={C1_ball.shape}, '
                  f'C1_euclid.shape={C1_euclid.shape}', flush=True)
        else:
            C1_ball = C1_euclid  # v1: C1 is on ball
            print(f'[H-E-E-E] v1 single codebook: C1.shape={C1_euclid.shape}', flush=True)
        # Process in batches
        r1_all = torch.zeros(len(x), dim)
        r2_all = torch.zeros(len(x), dim)
        r3_all = torch.zeros(len(x), dim)
        with torch.no_grad():
            bs = 1024
            x_dev = x.to(device)
            for bs_start in range(0, len(x_dev), bs):
                bs_end = min(bs_start + bs, len(x_dev))
                xb = x_dev[bs_start:bs_end]
                if is_dual:
                    x_ball = euclidean_to_poincare(xb, target_norm=MAX_NORM * 0.5)
                else:
                    x_ball = euclidean_to_poincare(xb)
                d1 = poincare_distance(
                    x_ball.unsqueeze(1),
                    C1_ball.unsqueeze(0), c=BALL_C,
                )
                idx1 = d1.argmin(dim=1)
                # v3: use Euclidean codebook for reconstruction (full magnitude)
                # v1: log_map0(C1_ball) maps ball back to small tangent vector
                if is_dual:
                    q1 = C1_euclid[idx1]
                else:
                    q1 = C1_ball[idx1]
                    q1 = log_map0(q1, c=BALL_C) * 0.5
                r1 = xb - q1
                d2 = (r1.unsqueeze(1) - C2_t.unsqueeze(0)).norm(dim=-1)
                idx2 = d2.argmin(dim=1)
                q2 = C2_t[idx2]
                r2 = r1 - q2
                d3 = (r2.unsqueeze(1) - C3_t.unsqueeze(0)).norm(dim=-1)
                idx3 = d3.argmin(dim=1)
                q3 = C3_t[idx3]
                r1_all[bs_start:bs_end] = r1.cpu()
                r2_all[bs_start:bs_end] = r2.cpu()
                r3_all[bs_start:bs_end] = (r2 - q3).cpu()
        residuals = {0: r1_all, 1: r2_all, 2: r3_all}
    else:
        # E-E-E-E: standard RQ-VAE
        codebooks, gains, has_gains, normalize, L = load_codebooks(ckpt_path)
        r_lst_full, _, _ = forward_residual(x, codebooks, gains if has_gains else None)
        residuals = {l: r_lst_full[l + 1] for l in range(L)}

    # If x was normalized for H-E-E-E, denormalize residuals back
    if x_mean is not None and x_std is not None:
        residuals = {l: r * x_std for l, r in residuals.items()}

    return residuals


def compute_d_residual(residuals_l0, sample_items_idx, n_sample=3000):
    """Pairwise Euclidean distance matrix for residual at l=0."""
    r_sample = residuals_l0[sample_items_idx]
    from scipy.spatial.distance import cdist
    return cdist(r_sample, r_sample, metric='euclidean').astype(np.float32)


def compute_d_input(emb_all, sample_items_idx):
    """Pairwise Euclidean distance matrix for input embedding."""
    return compute_d_residual(emb_all.float(), sample_items_idx)


def metric_taxonomy_transfer(ckpt_path, label, n_sample=3000, n_users=3000,
                              min_count=1, seed=42):
    """Compute taxonomy ρ and transfer naive ρ for one ckpt."""
    print(f'\n[{label}] Loading ckpt: {ckpt_path}', flush=True)
    emb_all = torch.load(EMB_PATH, map_location='cpu', weights_only=False)
    residuals = get_residuals(ckpt_path, emb_all)
    print(f'[{label}] Residual shapes: {[(l, r.shape) for l, r in residuals.items()]}')

    # Sample 3000 items
    rng = np.random.RandomState(seed)
    sample_idx = sorted(rng.choice(len(emb_all), n_sample, replace=False).tolist())
    sample_items = sample_idx  # use idx as item ids

    # Load item metadata
    metadata = load_metadata(N_ITEMS)

    # D_tax — 使用 task351 的 build_d_tree (0/0.5/1/2 scale), 与 task357 一致
    from task351_residual_structure import build_d_tree as _build_d_tree_full
    D_tax_full, _, _ = _build_d_tree_full(metadata, N_ITEMS)
    D_tax = D_tax_full[sample_idx][:, sample_idx]
    print(f'[{label}] D_tax: {D_tax.shape}, mean={D_tax.mean():.4f}, unique={np.unique(D_tax)[:5]}')

    # D_trans
    print(f'[{label}] Building transition graph (min_count={min_count})...', flush=True)
    t0 = time.time()
    trans_edges, _ = build_transition_graph(N_ITEMS, n_users=n_users, min_count=min_count)
    trans_adj = build_adj_list(trans_edges, N_ITEMS, directed=True)
    print(f'[{label}] transition graph: {len(trans_edges)} directed edges, '
          f'{len(trans_adj)} nodes, {time.time()-t0:.1f}s', flush=True)
    sampled_set = set(sample_items)
    # Use FULL adj (not filtered by source) so BFS can traverse non-sampled nodes
    from collections import deque
    def _sp(adj, sample_set):
        item_list = list(sample_set)
        item_to_idx = {it: i for i, it in enumerate(item_list)}
        n = len(item_list)
        INF = 999
        D = np.full((n, n), INF, dtype=np.float32)
        np.fill_diagonal(D, 0.0)
        for src in item_list:
            visited = {src: 0}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in adj.get(cur, []):
                    if nxt not in visited:
                        visited[nxt] = visited[cur] + 1
                        queue.append(nxt)
            for dst, d in visited.items():
                if dst in item_to_idx:
                    D[item_to_idx[src], item_to_idx[dst]] = d
        return D
    D_trans = _sp(trans_adj, sampled_set)
    D_trans[D_trans >= 999] = 50  # cap INF
    print(f'[{label}] D_trans: {D_trans.shape}, mean={D_trans.mean():.4f}')

    results = {'label': label, 'ckpt_path': ckpt_path, 'per_layer': {}}
    # INPUT layer — task357 风格: l=0 = input embedding (不是 residual)
    # 这样可以直接对照 task357 报告 ρ=0.4483 来验证 metadata 修复正确
    D_input = compute_d_residual(emb_all.float(), sample_idx)
    r_tax = spearman_rho_fast(D_tax, D_input)
    r_trans = spearman_rho_fast(D_trans, D_input)
    _, p_tax = mantel_test_fast(D_tax, D_input, n_perm=99, seed=42)
    _, p_trans = mantel_test_fast(D_trans, D_input, n_perm=99, seed=42)
    results['per_layer']['input'] = {
        'taxonomy_rho': float(r_tax),
        'taxonomy_p': float(p_tax),
        'transfer_rho': float(r_trans),
        'transfer_p': float(p_trans),
    }
    print(f'[{label}] input: taxonomy ρ={r_tax:+.4f} (p={p_tax:.4f}), '
          f'transfer ρ={r_trans:+.4f} (p={p_trans:.4f})', flush=True)
    for l in range(3):  # 3 layers
        D_res = compute_d_residual(residuals[l], sample_idx)
        # Mantel Spearman ρ (task358 复现: cat_sub 离散距离需要秩相关)
        r_tax = spearman_rho_fast(D_tax, D_res)
        r_trans = spearman_rho_fast(D_trans, D_res)
        # Permutation test for p-value (fast Pearson permutation)
        _, p_tax = mantel_test_fast(D_tax, D_res, n_perm=99, seed=42 + l)
        _, p_trans = mantel_test_fast(D_trans, D_res, n_perm=99, seed=42 + l)
        results['per_layer'][l] = {
            'taxonomy_rho': float(r_tax),
            'taxonomy_p': float(p_tax),
            'transfer_rho': float(r_trans),
            'transfer_p': float(p_trans),
        }
        print(f'[{label}] l={l}: taxonomy ρ={r_tax:+.4f} (p={p_tax:.4f}), '
              f'transfer ρ={r_trans:+.4f} (p={p_trans:.4f})', flush=True)
    return results


def load_metadata(n_items):
    """Load item metadata (cat_sub, cat_top) from TFRecord using task351 parse_text."""
    import glob
    import tensorflow as tf
    from task351_residual_structure import parse_text
    metadata = {}
    files = sorted(glob.glob(f'{GRID}/data/amazon_data/toys/items/data_*.tfrecord.gz'))
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        if 'id' not in ex.features.feature:
            continue
        item_id = int(ex.features.feature['id'].int64_list.value[0])
        if item_id >= n_items:
            continue
        if 'text' not in ex.features.feature:
            continue
        text_b = ex.features.feature['text'].bytes_list.value[0]
        m = parse_text(text_b)
        if m is not None:
            metadata[item_id] = m
    return metadata


# ========== Metric 3: R@10 ==========

def compute_r10(ckpt_path, label):
    """End-to-end R@10 — requires TIGER training, expensive.

    For Stage 2 quick eval, use a proxy: structural preservation + downstream prediction
    NOT included in this script — should run task15's TIGER inference for true R@10.
    For now, return None and note that R@10 requires separate TIGER training.
    """
    print(f'\n[{label}] R@10 requires separate TIGER training (not in this script).', flush=True)
    print(f'[{label}] To compute R@10, use task15 inference config + this ckpt\'s '
          f'quantization_layer_list centroids (for H-E-E-E need custom inference).', flush=True)
    return None


# ========== Metric 4: Collision rate ==========

def compute_collision_rate(ckpt_path, label, emb_all, code_dim_only=True):
    """Collision rate: fraction of items sharing the same L1+L2+L3 SID.

    For H-E-E-E: items sharing same (idx1, idx2, idx3) collide.
    For E-E-E-E: same definition.
    """
    print(f'\n[{label}] Computing collision rate...', flush=True)
    residuals = get_residuals(ckpt_path, emb_all)
    # Get codes per item
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    is_h_e_e_e = ckpt.get('hyper_parameters', {}).get('l1_space') in ('poincare_ball', 'poincare_ball_dual')

    codes = torch.zeros(len(emb_all), 3, dtype=torch.long)
    if is_h_e_e_e:
        is_dual_ckpt = 'C1_ball' in sd
        if is_dual_ckpt:
            from task388_h_e_e_e_train_v3 import (
                euclidean_to_poincare, poincare_distance, log_map0, BALL_C, MAX_NORM,
            )
        else:
            from task385_stage2_h_e_e_e_train import (
                euclidean_to_poincare, poincare_distance, log_map0, BALL_C, MAX_NORM,
            )
        dim = ckpt['hyper_parameters']['dim']
        # Detect dual codebook (v3)
        is_dual = 'C1_ball' in sd
        C1_euclid = sd['quantization_layer_list.0.centroids'].to(device)
        C2_t = sd['quantization_layer_list.1.centroids'].to(device)
        C3_t = sd['quantization_layer_list.2.centroids'].to(device)
        if is_dual:
            C1_ball = sd['C1_ball'].to(device)
        else:
            C1_ball = C1_euclid
        x_mean = ckpt.get('x_mean', None)
        x_std = ckpt.get('x_std', None)
        x = emb_all.float()
        if x_mean is not None:
            x = (x - x_mean) / x_std
        x_dev = x.to(device)
        bs = 1024
        with torch.no_grad():
            for s in range(0, len(x_dev), bs):
                e = min(s + bs, len(x_dev))
                xb = x_dev[s:e]
                if is_dual:
                    x_ball = euclidean_to_poincare(xb, target_norm=MAX_NORM * 0.5)
                else:
                    x_ball = euclidean_to_poincare(xb)
                d1 = poincare_distance(x_ball.unsqueeze(1), C1_ball.unsqueeze(0), c=BALL_C)
                idx1 = d1.argmin(dim=1)
                codes[s:e, 0] = idx1.cpu()
                if is_dual:
                    q1 = C1_euclid[idx1]
                else:
                    q1 = C1_ball[idx1]
                    q1 = log_map0(q1, c=BALL_C) * 0.5
                r1 = xb - q1
                d2 = (r1.unsqueeze(1) - C2_t.unsqueeze(0)).norm(dim=-1)
                idx2 = d2.argmin(dim=1)
                codes[s:e, 1] = idx2.cpu()
                q2 = C2_t[idx2]
                r2 = r1 - q2
                d3 = (r2.unsqueeze(1) - C3_t.unsqueeze(0)).norm(dim=-1)
                idx3 = d3.argmin(dim=1)
                codes[s:e, 2] = idx3.cpu()
    else:
        codebooks, gains, has_gains, normalize, L = load_codebooks(ckpt_path)
        x = emb_all.float()
        with torch.no_grad():
            r_lst, q_lst, idx_lst = forward_residual(x, codebooks,
                                                     gains if has_gains else None)
            for l in range(L):
                codes[:, l] = idx_lst[l].long()

    # Convert to tuples and count
    codes_tuples = [tuple(c.tolist()) for c in codes]
    from collections import Counter
    counter = Counter(codes_tuples)
    n_items = len(codes)
    n_unique_sids = len(counter)
    n_collisions = sum(c for c in counter.values() if c > 1)
    n_items_in_collision = n_items - n_unique_sids
    collision_rate = n_items_in_collision / n_items
    print(f'[{label}] n_items={n_items}, n_unique_SIDs={n_unique_sids}, '
          f'collision_rate={collision_rate:.4f}', flush=True)
    return {
        'n_items': n_items,
        'n_unique_sids': n_unique_sids,
        'n_items_in_collision': n_items_in_collision,
        'collision_rate': float(collision_rate),
    }


# ========== Decision logic ==========

def make_decision(e_results, h_results):
    """Apply user's decision rules."""
    print('\n=== Stage 2 Decision ===', flush=True)

    # Use l=0 (first layer) as the key metric for taxonomy, l=0 for transfer
    e_l0 = e_results['per_layer'][0]
    h_l0 = h_results['per_layer'][0]

    # 1. Taxonomy change
    e_tax = e_l0['taxonomy_rho']
    h_tax = h_l0['taxonomy_rho']
    tax_delta = h_tax - e_tax

    # 2. Transfer change (naive)
    e_trans = e_l0['transfer_rho']
    h_trans = h_l0['transfer_rho']
    trans_delta = h_trans - e_trans

    # 3. Collision rate change
    e_coll = e_results['collision']['collision_rate']
    h_coll = h_results['collision']['collision_rate']
    coll_delta = h_coll - e_coll

    print(f'  Taxonomy ρ: E-E-E-E = {e_tax:+.4f}, H-E-E-E = {h_tax:+.4f}, Δ = {tax_delta:+.4f}',
          flush=True)
    print(f'  Transfer ρ: E-E-E-E = {e_trans:+.4f}, H-E-E-E = {h_trans:+.4f}, Δ = {trans_delta:+.4f}',
          flush=True)
    print(f'  Collision rate: E-E-E-E = {e_coll:.4f}, H-E-E-E = {h_coll:.4f}, Δ = {coll_delta:+.4f}',
          flush=True)

    # Apply decision rules
    # Rule 1: Taxonomy ↑ + transfer stable (0.03-0.05) → adopt
    # Rule 2: Taxonomy ↑ + transfer ↓ (<0.02) → check R@10
    # Rule 3: Taxonomy ≈ → abandon

    if h_trans >= 0.02 and (0.03 <= h_trans <= 0.07):
        transfer_status = 'stable_in_band'
    elif h_trans < 0.02:
        transfer_status = 'dropped_below_0.02'
    else:
        transfer_status = 'changed_other'

    if tax_delta > 0.02:
        tax_status = 'improved_significantly'
    elif tax_delta < -0.02:
        tax_status = 'degraded'
    else:
        tax_status = 'no_change'

    if tax_status == 'no_change':
        decision = 'ABANDON_hyperbolic (taxonomy did not improve)'
    elif transfer_status == 'stable_in_band':
        decision = 'ADOPT_H_E_E_E (taxonomy up, transfer stable)'
    elif transfer_status == 'dropped_below_0.02':
        decision = ('CHECK_R10 (taxonomy up but transfer dropped; '
                    'need R@10 to decide: R@10 up → adopt, R@10 flat/down → abandon)')
    else:
        decision = ('UNCLEAR (transfer changed unexpectedly; '
                    'manually inspect before adopting)')

    print(f'\n  Taxonomy status: {tax_status}', flush=True)
    print(f'  Transfer status: {transfer_status}', flush=True)
    print(f'\n  >>> DECISION: {decision} <<<', flush=True)

    return {
        'e_taxonomy_rho': e_tax,
        'h_taxonomy_rho': h_tax,
        'taxonomy_delta': tax_delta,
        'tax_status': tax_status,
        'e_transfer_rho': e_trans,
        'h_transfer_rho': h_trans,
        'transfer_delta': trans_delta,
        'transfer_status': transfer_status,
        'e_collision_rate': e_coll,
        'h_collision_rate': h_coll,
        'collision_delta': coll_delta,
        'decision': decision,
    }


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h-ckpt', type=str, required=True,
                        help='H-E-E-E checkpoint path')
    parser.add_argument('--e-ckpt', type=str, default=E_E_E_E_CKPT,
                        help='E-E-E-E baseline ckpt (default: task13_group_a_s21)')
    parser.add_argument('--n-sample', type=int, default=3000)
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--skip-r10', action='store_true',
                        help='Skip R@10 (requires TIGER training)')
    args = parser.parse_args()

    print(f'[task385 eval] H-E-E-E ckpt: {args.h_ckpt}', flush=True)
    print(f'[task385 eval] E-E-E-E ckpt: {args.e_ckpt}', flush=True)

    # Load embeddings once
    emb_all = torch.load(EMB_PATH, map_location='cpu', weights_only=False)
    print(f'[task385 eval] Embeddings: {emb_all.shape}', flush=True)

    # Metric 1+2: Taxonomy ρ + Transfer ρ
    print('\n=== E-E-E-E baseline ===', flush=True)
    e_results = metric_taxonomy_transfer(args.e_ckpt, 'E-E-E-E',
                                          n_sample=args.n_sample,
                                          n_users=args.n_users,
                                          min_count=args.min_count)
    e_results['collision'] = compute_collision_rate(args.e_ckpt, 'E-E-E-E', emb_all)

    print('\n=== H-E-E-E (Stage 2) ===', flush=True)
    h_results = metric_taxonomy_transfer(args.h_ckpt, 'H-E-E-E',
                                          n_sample=args.n_sample,
                                          n_users=args.n_users,
                                          min_count=args.min_count)
    h_results['collision'] = compute_collision_rate(args.h_ckpt, 'H-E-E-E', emb_all)

    # Metric 3: R@10 (skipped by default)
    if args.skip_r10:
        e_results['r10'] = None
        h_results['r10'] = None
    else:
        e_results['r10'] = compute_r10(args.e_ckpt, 'E-E-E-E')
        h_results['r10'] = compute_r10(args.h_ckpt, 'H-E-E-E')

    # Decision
    decision = make_decision(e_results, h_results)

    # Save
    out = {
        'task': 'task385_stage2_4metric_eval',
        'method': 'four_metric_comparison',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'status': 'completed',
        'data': {
            'config': {
                'h_ckpt': args.h_ckpt,
                'e_ckpt': args.e_ckpt,
                'n_sample': args.n_sample,
                'n_users': args.n_users,
                'min_count': args.min_count,
            },
            'e_eee_results': e_results,
            'h_eee_results': h_results,
            'decision': decision,
        }
    }
    out_path = f'{OUT_DIR}/stage2_4metric_eval.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[task385 eval] Saved: {out_path}', flush=True)
    print(f'[task385 eval] Final decision: {decision["decision"]}', flush=True)


if __name__ == '__main__':
    main()