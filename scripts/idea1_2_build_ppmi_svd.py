#!/usr/bin/env python3
"""Build PPMI-SVD 256-d collaborative embedding from training sequences.

Process:
  1. Load user sequences from data/amazon_data/toys/training/ (N=11924 items universe)
  2. Build sparse co-occurrence count matrix C[i,j] = #users with both items
  3. Compute PMI: PPMI[i,j] = max(0, log P(i,j) / (P(i)P(j)))
  4. TruncatedSVD to 256 dim -> f_i
  5. L2 normalize -> sim_cf(i,j) = f_i · f_j = cosine
  6. Save f_i (11924, 256) and item universe mapping
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import glob
import time
import json
import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys/training'
N_ITEMS_UNIVERSE = 11924
SVD_DIM = 256
N_USERS_LIMIT = 50000   # Load up to 50k users (3x of training set for stable PMI)


def load_user_sequences(n_users_max):
    """Load up to n_users_max user sequences from training partitions."""
    import tensorflow as tf
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'partition_*.tfrecord.gz')))
    print(f'  reading {len(files)} tfrecord files, target {n_users_max} users')
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
    user_seqs = []
    n = 0
    for raw in ds:
        if n >= n_users_max:
            break
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        seq = list(example.features.feature['sequence_data'].int64_list.value)
        if len(seq) >= 2:
            # Filter to item universe
            seq = sorted(set(x for x in seq if x < N_ITEMS_UNIVERSE))
            if len(seq) >= 2:
                user_seqs.append(seq)
        n += 1
    print(f'  loaded {len(user_seqs)} user sequences (≥2 items in universe)')
    return user_seqs


def build_co_occurrence(user_seqs, N):
    """Build co-occurrence matrix as sparse symmetric CSR.

    C[i,j] = #users with both i, j in their sequence.
    Diagonal zeroed.
    """
    rows, cols = [], []
    for seq in user_seqs:
        L = len(seq)
        for i in range(L):
            for j in range(i+1, L):
                rows.append(seq[i])
                cols.append(seq[j])
                rows.append(seq[j])
                cols.append(seq[i])
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    C = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    print(f'  co-occurrence shape={C.shape}, nnz={C.nnz}, density={C.nnz/C.size:.6f}')
    return C


def compute_ppmi(C_csr):
    """Convert co-occurrence to PPMI via shift-log formula.

    PPMI[i,j] = max(0, log P(i,j) / (P(i)P(j)))
              = max(0, log C[i,j] * N_total / (C[i,*].sum * C[*,j].sum))
    where N_total = sum of C (off-diagonal counts doubled).
    """
    row_sum = np.asarray(C_csr.sum(axis=1)).ravel()  # C[i,*]
    col_sum = np.asarray(C_csr.sum(axis=0)).ravel()  # C[*,j]
    total = row_sum.sum()                              # 2 * sum of unique pairs
    print(f'  total co-occurrences: {total:.0f}')

    # Compute PMI on nonzero entries: log(C[i,j]*total / (row_sum[i]*col_sum[j]))
    coo = C_csr.tocoo()
    r, c, v = coo.row, coo.col, coo.data
    pmi = np.log(v * total / (row_sum[r] * col_sum[c] + 1e-12) + 1e-12)
    pmi = np.maximum(pmi, 0.0).astype(np.float32)
    P = coo_matrix((pmi, (r, c)), shape=C_csr.shape).tocsr()
    P.eliminate_zeros()
    print(f'  PPMI shape={P.shape}, nnz={P.nnz}, mean={pmi.mean():.3f}, max={pmi.max():.3f}')
    return P


def svd_l2_normalize(P, k=SVD_DIM, seed=42):
    """TruncatedSVD then L2-normalize rows."""
    svd = TruncatedSVD(n_components=k, random_state=seed, n_iter=8)
    f = svd.fit_transform(P).astype(np.float32)        # (N, k)
    explained = svd.explained_variance_ratio_.sum()
    print(f'  SVD: dim={k}, explained_var_sum={explained:.4f}, f shape={f.shape}')
    norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    f_norm = f / norms
    return f_norm, f, explained


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'Step 1: Load user sequences (up to {N_USERS_LIMIT})')
    print('=' * 70)
    user_seqs = load_user_sequences(N_USERS_LIMIT)
    print(f'  time: {time.time()-t0:.1f}s')

    print('\n' + '=' * 70)
    print('Step 2: Build co-occurrence matrix')
    print('=' * 70)
    C = build_co_occurrence(user_seqs, N_ITEMS_UNIVERSE)
    print(f'  time: {time.time()-t0:.1f}s')

    print('\n' + '=' * 70)
    print('Step 3: PPMI')
    print('=' * 70)
    P = compute_ppmi(C)
    print(f'  time: {time.time()-t0:.1f}s')

    print('\n' + '=' * 70)
    print(f'Step 4: TruncatedSVD → {SVD_DIM} dim, L2 normalize')
    print('=' * 70)
    f_norm, f_raw, ev = svd_l2_normalize(P)
    print(f'  time: {time.time()-t0:.1f}s')

    print('\n' + '=' * 70)
    print('Step 5: Save')
    print('=' * 70)
    out_path = os.path.join(OUT_DIR, 'cf_ppmi_svd256.pt')
    torch.save({
        'f_normalized': torch.from_numpy(f_norm),   # (11924, 256) L2-normalized
        'f_raw':        torch.from_numpy(f_raw),     # before normalize
        'item_universe_size': N_ITEMS_UNIVERSE,
        'svd_dim': SVD_DIM,
        'explained_variance_ratio_sum': ev,
        'n_users_loaded': len(user_seqs),
        'co_occurrence_nnz': int(C.nnz),
        'ppmi_nnz': int(P.nnz),
        'ppmi_mean': float(P.data.mean()),
    }, out_path)
    print(f'  saved → {out_path}')
    print(f'  total time: {time.time()-t0:.1f}s')

    # Quick sanity: similarity distribution
    sample = np.random.default_rng(42).choice(N_ITEMS_UNIVERSE, size=2000, replace=False)
    Fs = f_norm[sample]
    sims = Fs @ Fs.T
    np.fill_diagonal(sims, 0)
    # Off-diagonal sims (flat) for pairs
    iu, ju = np.triu_indices(2000, k=1)
    pair_sims = sims[iu, ju]
    print(f'\nSanity check:')
    print(f'  random pair sim_cf: mean={pair_sims.mean():.4f}, '
          f'std={pair_sims.std():.4f}, '
          f'p99={np.quantile(pair_sims, 0.99):.4f}, '
          f'max={pair_sims.max():.4f}')
    json.dump({
        'sim_cf_random_pairs': {
            'mean': float(pair_sims.mean()),
            'std': float(pair_sims.std()),
            'p50': float(np.quantile(pair_sims, 0.5)),
            'p75': float(np.quantile(pair_sims, 0.75)),
            'p90': float(np.quantile(pair_sims, 0.9)),
            'p95': float(np.quantile(pair_sims, 0.95)),
            'p99': float(np.quantile(pair_sims, 0.99)),
            'max': float(pair_sims.max()),
        }
    }, open(os.path.join(OUT_DIR, 'cf_sim_distribution.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()