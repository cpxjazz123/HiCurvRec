#!/usr/bin/env python3
"""Centering audit — reproduce original ρ(q_1) = 0.1088 vs new ρ(q_1) = 0.6162, identify the bug.

Test 4 centering conventions on the SAME q_1 and U to isolate the source of 6x discrepancy.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import json
import numpy as np
import torch

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
RQIDX_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def build_task_subspace_fast(emb_c_np, labels_str, m=10, min_count=2):
    from collections import Counter
    d = emb_c_np.shape[1]
    cnt = Counter(labels_str)
    valid_labels = sorted([k for k, v in cnt.items() if v >= min_count])
    N = emb_c_np.shape[0]
    mu_G = emb_c_np.mean(axis=0)
    diffs = np.zeros((len(valid_labels), d))
    for i, k in enumerate(valid_labels):
        mask = np.array([l == k for l in labels_str])
        n_k = mask.sum()
        if n_k == 0:
            continue
        diffs[i] = np.sqrt(n_k / N) * (emb_c_np[mask].mean(axis=0) - mu_G)
    S_small = diffs @ diffs.T
    eigvals, V_small = np.linalg.eigh(S_small)
    idx = np.argsort(-np.abs(eigvals))[:m]
    U = diffs.T @ V_small[:, idx]
    Q, _ = np.linalg.qr(U)
    return Q[:, :m]


def rho(v, U):
    """||U^T v||² / ||v||² (v can be any shape (N, D))."""
    v_proj = v @ U
    var_task = (v_proj ** 2).sum(axis=1).mean()
    var_total = (v ** 2).sum(axis=1).mean()
    return float(var_task / var_total)


def main():
    print('=' * 70)
    print('Centering audit — ρ(q_1) under 4 centering conventions')
    print('=' * 70)

    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float().numpy()
    N, d = emb.shape
    with open(META_PATH) as f:
        md = json.load(f)
    labels_cat_sub = [md[str(i)]['cat_sub'] for i in range(N)]
    labels_brand = [md[str(i)]['brand'] for i in range(N)]

    bundle = torch.load(RQIDX_PATH, map_location='cpu', weights_only=False)
    q1 = bundle['q_lst'][0].float().numpy()

    print(f'\nshapes: emb={emb.shape}, q1={q1.shape}')
    print(f'mean norms: ‖mean(emb)‖={np.linalg.norm(emb.mean(0)):.4f}, '
          f'‖mean(q1)‖={np.linalg.norm(q1.mean(0)):.4f}')

    # Build U from cat_sub on x_centered (matches supplementary verdict)
    emb_centered = emb - emb.mean(0)
    U_cat = build_task_subspace_fast(emb_centered, labels_cat_sub, m=10)
    U_brand = build_task_subspace_fast(emb_centered, labels_brand, m=10, min_count=2)

    print(f'\n--- ρ^{{task}} on cat_sub U under 4 conventions ---')
    print(f'{"convention":<40} {"ρ(x)":<10} {"ρ(q_1)":<10} {"drop":<10}')

    # Convention A: uncentered both (matches original supplementary line 118)
    rho_x_A = rho(emb, U_cat)
    rho_q_A = rho(q1, U_cat)
    print(f'{"A: uncentered x, uncentered q_1":<40} {rho_x_A:<10.4f} {rho_q_A:<10.4f} {(rho_x_A-rho_q_A)/rho_x_A*100:<10.1f}%')

    # Convention B: x centered (x_c), q1 uncentered (matches verdict section 3.2 table)
    rho_x_B = rho(emb_centered, U_cat)
    rho_q_B = rho(q1, U_cat)
    print(f'{"B: x_c (centered), q_1 uncentered":<40} {rho_x_B:<10.4f} {rho_q_B:<10.4f} {(rho_x_B-rho_q_B)/rho_x_B*100:<10.1f}%')

    # Convention C: x centered, q_1 centered with x's mean (what I did in brand_u_rho_drop)
    q1_minus_emb_mean = q1 - emb.mean(0)
    rho_x_C = rho(emb_centered, U_cat)
    rho_q_C = rho(q1_minus_emb_mean, U_cat)
    print(f'{"C: x_c, q_1 - mean(emb) [my code]":<40} {rho_x_C:<10.4f} {rho_q_C:<10.4f} {(rho_x_C-rho_q_C)/rho_x_C*100:<10.1f}%')

    # Convention D: x centered, q_1 centered with q_1's own mean (most consistent)
    q1_minus_q1_mean = q1 - q1.mean(0)
    rho_x_D = rho(emb_centered, U_cat)
    rho_q_D = rho(q1_minus_q1_mean, U_cat)
    print(f'{"D: x_c, q_1 - mean(q_1) [own mean]":<40} {rho_x_D:<10.4f} {rho_q_D:<10.4f} {(rho_x_D-rho_q_D)/rho_x_D*100:<10.1f}%')

    print(f'\n--- ρ^{{task}} on brand U under 4 conventions ---')
    print(f'{"convention":<40} {"ρ(x)":<10} {"ρ(q_1)":<10} {"drop":<10}')
    rho_x_A_b = rho(emb, U_brand)
    rho_q_A_b = rho(q1, U_brand)
    print(f'{"A: uncentered":<40} {rho_x_A_b:<10.4f} {rho_q_A_b:<10.4f} {(rho_x_A_b-rho_q_A_b)/rho_x_A_b*100:<10.1f}%')
    rho_x_B_b = rho(emb_centered, U_brand)
    rho_q_B_b = rho(q1, U_brand)
    print(f'{"B: x_c, q_1 uncentered":<40} {rho_x_B_b:<10.4f} {rho_q_B_b:<10.4f} {(rho_x_B_b-rho_q_B_b)/rho_x_B_b*100:<10.1f}%')
    rho_q_C_b = rho(q1_minus_emb_mean, U_brand)
    print(f'{"C: x_c, q_1 - mean(emb)":<40} {rho_x_B_b:<10.4f} {rho_q_C_b:<10.4f} {(rho_x_B_b-rho_q_C_b)/rho_x_B_b*100:<10.1f}%')
    rho_q_D_b = rho(q1_minus_q1_mean, U_brand)
    print(f'{"D: x_c, q_1 - mean(q_1)":<40} {rho_x_D:<10.4f} {rho_q_D_b:<10.4f} {(rho_x_B_b-rho_q_D_b)/rho_x_B_b*100:<10.1f}%')

    # Save
    out = {
        'cat_sub': {'A': [rho_x_A, rho_q_A], 'B': [rho_x_B, rho_q_B],
                    'C': [rho_x_C, rho_q_C], 'D': [rho_x_D, rho_q_D]},
        'brand':   {'A': [rho_x_A_b, rho_q_A_b], 'B': [rho_x_B_b, rho_q_B_b],
                    'C': [rho_x_B_b, rho_q_C_b], 'D': [rho_x_B_b, rho_q_D_b]},
    }
    out_path = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/brand_u_rho_drop/centering_audit.json'
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()