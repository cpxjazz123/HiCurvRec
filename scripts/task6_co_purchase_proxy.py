#!/usr/bin/env python3
# task18_co_purchase_proxy.py — Spearman: pairwise co-purchase frequency vs SID pairwise distance
#
# Simplified version (CPU only):
# - Load first N=3000 user sequences from training/partition_0
# - Build sparse co-purchase matrix: f_ij = #users who have both i,j in sequence
# - Restrict to items that exist in our 11924-item SID tensors
# - Sample 500 items uniformly, compute:
#     - co-purchase frequency f_ij (for i,j in sample)
#     - HRQ Poincaré L1 centroid distance d_ij (HRQ v2)
#     - RQ Euclidean L1 centroid distance d_ij (baseline)
# - Spearman correlation per algorithm
#
# Output: result/task20/task18_co_purchase.json

import sys, os, json, time, glob
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch
from scipy.stats import spearmanr

OUT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task7'
os.makedirs(OUT, exist_ok=True)

DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys/training'
EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
HRQ_SID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'
GROUP_A_CODEBOOKS = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'

N_USERS = 3000      # load first 3000 user sequences
N_ITEMS = 500       # sample 500 items for pairwise comparison


def load_user_sequences(n_users):
    """Load first n_users user-item sequences from training tfrecords."""
    import tensorflow as tf
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'partition_*.tfrecord.gz')))
    print(f'  reading {len(files)} tfrecord files, target {n_users} users')
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
    user_seqs = []
    n = 0
    for raw in ds:
        if n >= n_users:
            break
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        seq = list(example.features.feature['sequence_data'].int64_list.value)
        if len(seq) >= 2:
            user_seqs.append(seq)
            n += 1
    print(f'  loaded {len(user_seqs)} user sequences (min seq len 2)')
    return user_seqs


def build_co_purchase_freq(user_seqs, max_item_id):
    """Build sparse co-purchase count matrix using COO format."""
    rows, cols = [], []
    n_pairs = 0
    for seq in user_seqs:
        seq = sorted(set([x for x in seq if x < max_item_id]))
        L = len(seq)
        for i in range(L):
            for j in range(i+1, L):
                rows.append(seq[i])
                cols.append(seq[j])
                rows.append(seq[j])
                cols.append(seq[i])
                n_pairs += 1
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    data = np.ones(len(rows), dtype=np.float32)
    M = np.zeros((max_item_id, max_item_id), dtype=np.float32)
    # accumulate
    np.add.at(M, (rows, cols), data)
    # Zero diagonal
    np.fill_diagonal(M, 0)
    print(f'  total pairs: {n_pairs // 2}, matrix shape: {M.shape}, density: {M.nonzero()[0].size / M.size:.6f}')
    return M


def lorentz_inner(h1, h2):
    return -h1[..., 0] * h2[..., 0] + (h1[..., 1:] * h2[..., 1:]).sum(-1)

def lorentz_distance(p, q, c=1.0):
    inner = lorentz_inner(p, q)
    arg = torch.clamp(-inner / c, min=1.0 + 1e-9)
    return torch.acosh(arg)

def lift_to_lorentz(x, c=1.0):
    x_norm2 = (x * x).sum(-1, keepdim=True)
    h0 = torch.sqrt(c + x_norm2)
    return torch.cat([h0, x], dim=-1)


def compute_hrq_centroid_distances(x, sid, K_sample):
    """HRQ v2: centroids in Poincaré (Lorentz) — per-cluster Euclidean mean then lift."""
    L1_codes = sid[1].long()    # 11924
    K_total = int(L1_codes.max()) + 1
    centroids_l = torch.zeros(K_total, x.shape[1])
    for k in range(K_total):
        mask = (L1_codes == k)
        if mask.any():
            centroids_l[k] = x[mask].mean(0)
    centroids_h = lift_to_lorentz(centroids_l, c=1.0)
    d = torch.zeros(K_total, K_total)
    for i in range(K_total):
        d[i] = lorentz_distance(centroids_h[i:i+1].expand(K_total, -1), centroids_h)
    return d


def compute_rq_centroid_distances(cb, K_sample):
    """RQ baseline: L1 centroids Euclidean pairwise distance."""
    return torch.cdist(cb, cb)


def main():
    print('=' * 70)
    print('task20 co-purchase proxy — Spearman f_ij vs d_ij')
    print('=' * 70)

    # Step 1: load user sequences
    print('\nStep 1: load user sequences')
    user_seqs = load_user_sequences(N_USERS)

    # Step 2: build co-purchase freq matrix
    print('\nStep 2: build co-purchase freq matrix')
    max_item = 11924
    M = build_co_purchase_freq(user_seqs, max_item)
    M_t = torch.from_numpy(M)

    # Step 3: load SID tensors + embeddings
    print('\nStep 3: load SID tensors + embeddings')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    sid_hrq = torch.load(HRQ_SID, weights_only=False, map_location='cpu')
    bundle_a = torch.load(GROUP_A_CODEBOOKS, weights_only=False, map_location='cpu')

    # Step 4: compute centroid distances (HRQ Poincaré + RQ Euclidean)
    print('\nStep 4: compute HRQ Poincaré L1 centroid distances')
    d_hrq = compute_hrq_centroid_distances(x, sid_hrq, K_sample=N_ITEMS)
    print(f'  HRQ d shape: {d_hrq.shape}, range: [{d_hrq.min():.3f}, {d_hrq.max():.3f}]')

    print('Step 5: compute RQ Euclidean L1 centroid distances')
    cb_a = bundle_a['codebooks'][0].float()
    d_rq = compute_rq_centroid_distances(cb_a, K_sample=N_ITEMS)
    print(f'  RQ d shape: {d_rq.shape}, range: [{d_rq.min():.3f}, {d_rq.max():.3f}]')

    # Step 6: align — restrict to items in same K-clusters, sample 500 items
    print('\nStep 6: align matrices to 500 random items')
    K = min(d_hrq.shape[0], d_rq.shape[0])
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(K, size=min(N_ITEMS, K), replace=False)
    sample_idx_t = torch.from_numpy(sample_idx).long()

    # Sub-matrices: distance between centroid A and centroid B for sampled items
    d_hrq_sub = d_hrq[sample_idx_t][:, sample_idx_t].numpy()
    d_rq_sub = d_rq[sample_idx_t][:, sample_idx_t].numpy()

    # co-purchase freq restricted to sampled item IDs (sample_idx are L1 cluster ids, not item ids!)
    # Need a mapping from L1 cluster to item ids
    L1_codes = sid_hrq[1].long().numpy()
    item_to_cluster = L1_codes  # 11924 items
    # For each sampled cluster, get items assigned to it
    cluster_to_items = {k: np.where(L1_codes == k)[0] for k in sample_idx}
    # Sub-matrix: M[item_i, item_j] aggregated across items in cluster
    n = len(sample_idx)
    f_sub = np.zeros((n, n), dtype=np.float32)
    for i_idx, c_i in enumerate(sample_idx):
        items_i = cluster_to_items[c_i]
        for j_idx, c_j in enumerate(sample_idx):
            items_j = cluster_to_items[c_j]
            if i_idx == j_idx:
                f_sub[i_idx, j_idx] = M[np.ix_(items_i, items_j)].sum()
            else:
                # Symmetric off-diagonal
                f_sub[i_idx, j_idx] = M[np.ix_(items_i, items_j)].sum()

    # Flatten upper triangular
    mask = np.triu(np.ones_like(d_hrq_sub, dtype=bool), k=1)
    f_flat = f_sub[mask]
    d_hrq_flat = d_hrq_sub[mask]
    d_rq_flat = d_rq_sub[mask]
    print(f'  pairwise count: {len(f_flat)}')

    # Filter zero co-purchase
    nz = f_flat > 0
    f_nz = f_flat[nz]
    d_hrq_nz = d_hrq_flat[nz]
    d_rq_nz = d_rq_flat[nz]
    print(f'  non-zero co-purchase pairs: {nz.sum()} / {len(f_flat)} ({100*nz.mean():.1f}%)')

    # Spearman
    rho_hrq, p_hrq = spearmanr(f_nz, d_hrq_nz)
    rho_rq, p_rq = spearmanr(f_nz, d_rq_nz)
    print(f'\nSpearman(co-purchase freq, distance):')
    print(f'  HRQ Poincaré: ρ = {rho_hrq:.4f}, p = {p_hrq:.4g}, n = {len(f_nz)}')
    print(f'  RQ Euclidean:  ρ = {rho_rq:.4f}, p = {p_rq:.4g}, n = {len(f_nz)}')

    info = {
        'co_purchase': {
            'n_users': int(len(user_seqs)),
            'n_items_sampled': int(N_ITEMS),
            'n_pairs_total': int(len(f_flat)),
            'n_pairs_nonzero': int(nz.sum()),
            'frac_nonzero': float(nz.mean()),
        },
        'spearman': {
            'hrq_poincare_l1': {'rho': float(rho_hrq), 'p_value': float(p_hrq), 'n': int(len(f_nz))},
            'rq_euclidean_l1':  {'rho': float(rho_rq),  'p_value': float(p_rq),  'n': int(len(f_nz))},
        },
        'interpretation': {
            'hrq_better_than_rq': bool(abs(rho_hrq) > abs(rho_rq)),
            'hrq_sign': 'positive' if rho_hrq > 0 else 'negative',
            'rq_sign': 'positive' if rho_rq > 0 else 'negative',
            'note': 'positive Spearman = co-purchase items have LARGER centroid distance; negative = co-purchase items have SMALLER distance'
        }
    }
    out_path = os.path.join(OUT, 'task18_co_purchase.json')
    with open(out_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'\n=== JSON → {out_path} ===')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()