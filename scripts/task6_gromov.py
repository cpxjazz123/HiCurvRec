#!/usr/bin/env python3
# task18_gromov.py — Gromov δ/diam criterion on HRQ v2 codebook
#
# 测 HRQ v2 的 Poincaré 双曲空间几何是否 "足够双曲":
# - 算 L1 codebook 256 个 centroids 之间的 pairwise Poincaré 距离
# - δ = min pairwise 距离
# - diam = max pairwise 距离
# - 必须 δ/diam ∈ [1/n_layer, 1/2] = [1/3, 1/2] for L=3
#
# 也算 pairwise cos 相似度 (RQ Group A 对照)

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import torch
import numpy as np

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task7'
os.makedirs(OUT_DIR, exist_ok=True)

# HRQ v2 codebook: 任务没 cache 256 centroids per layer. Re-extract via Lloyd run from saved SID.
HRQ_SID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'
EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
GROUP_A_CODEBOOKS = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


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


def gromov_metrics(distance_matrix):
    """δ = min upper triangular distance, diam = max.
       Returns δ, diam, ratio.
    """
    n = distance_matrix.shape[0]
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    pairs = distance_matrix[mask]
    delta = pairs.min().item()
    diam = pairs.max().item()
    ratio = delta / (diam + 1e-12)
    return delta, diam, ratio


def main():
    print('=' * 70)
    print('task20 Gromov δ/diam criterion — HRQ v2 vs RQ baseline')
    print('=' * 70)

    # Re-construct HRQ L1 codebook via Lloyd on first-layer assignments
    # Load SID tensor and embedding x
    sid = torch.load(HRQ_SID, weights_only=False, map_location='cpu')    # (4, 11924)
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()  # (11924, 2048)
    L1_codes = sid[1].long()                                              # 11924
    K = int(L1_codes.max()) + 1
    print(f'HRQ SID L1 codes: {L1_codes.shape}, K={K}')

    # Lift x to Lorentz
    c = 1.0
    h_x = lift_to_lorentz(x, c=c)                                       # (11924, 2049)
    # Compute centroid in Lorentz: per cluster, take Euclidean centroid then re-lift (approximation)
    centroids_l = torch.zeros(K, x.shape[1])
    for k in range(K):
        mask = (L1_codes == k)
        if mask.any():
            centroids_l[k] = x[mask].mean(0)
    centroids_h = lift_to_lorentz(centroids_l, c=c)                      # (K, 2049)

    # Pairwise Lorentz distance
    print('Computing pairwise Lorentz distance (K=%d)...' % K)
    d_lorentz = torch.zeros(K, K)
    for i in range(K):
        d_lorentz[i] = lorentz_distance(centroids_h[i:i+1].expand(K, -1), centroids_h)
    delta_l, diam_l, ratio_l = gromov_metrics(d_lorentz)
    print(f'\nHRQ L1 (Poincaré): δ={delta_l:.4f}, diam={diam_l:.4f}, δ/diam={ratio_l:.4f}')
    print(f'  kill-line (1/L, 1/2) for L=3: [0.333, 0.500]')

    # RQ baseline L1 centroid Euclidean distances for comparison
    bundle = torch.load(GROUP_A_CODEBOOKS, weights_only=False, map_location='cpu')
    cb_a = bundle['codebooks'][0].float()                                # (256, 2048)
    d_euc = torch.cdist(cb_a, cb_a)
    delta_e, diam_e, ratio_e = gromov_metrics(d_euc)
    print(f'\nRQ A L1 (Euclidean): δ={delta_e:.4f}, diam={diam_e:.4f}, δ/diam={ratio_e:.4f}')

    # Save
    info = {
        'hrq_l1_poincare': {'delta': delta_l, 'diam': diam_l, 'ratio': ratio_l},
        'rq_a_l1_euclidean': {'delta': delta_e, 'diam': diam_e, 'ratio': ratio_e},
        'kill_lines_for_L3': {'min_ratio': 1/3, 'max_ratio': 0.5,
                              'hrq_pass': (1/3 <= ratio_l <= 0.5),
                              'rq_pass': (1/3 <= ratio_e <= 0.5)},
        'interpretation': {
            'hrq_delta_smaller_than_rq': delta_l < delta_e,
            'hrq_diam_larger_than_rq': diam_l > diam_e,
            'hrq_ratio_smaller_than_rq': ratio_l < ratio_e,
        }
    }
    out_path = os.path.join(OUT_DIR, 'task18_gromov.json')
    with open(out_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'\n=== JSON → {out_path} ===')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
