#!/usr/bin/env python3
"""Idea 1 现象 2: 剥离尺度后测方向层面失配 Δ_1^{shape}

目的:
  - r̂_l = r_l / ‖r_l‖ (逐元素单位化残差, 只保留方向)
  - Ĉ_l = {c / ‖c‖, ∀c ∈ C_l} (逐码字单位化, 只保留每个码字的方向)
  - 重算 Δ_1^{shape} 和 Δ_2^{shape}
  - 比值 Δ_1^{shape}/Δ_2^{shape} 远低于原始 189x 即方向断裂不成立

Kill 线:
  - Δ_1^{shape}/Δ_2^{shape} < 3x → 判死结构性断裂
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
DATA = {
    'A_baseline (AQ, K=256^3)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'retrained WF (K=[256,64,16])': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def qerr(r, C):
    """QErr: mean per-item ‖r_i − nearest_C‖²."""
    r_norm2 = (r ** 2).sum(-1, keepdims=True)
    c_norm2 = (C ** 2).sum(-1)
    cross = r @ C.T
    d2 = np.maximum(r_norm2 + c_norm2[None, :] - 2 * cross, 0)
    nearest_idx = d2.argmin(axis=-1)
    nearest_c = C[nearest_idx]
    err = ((r - nearest_c) ** 2).sum(axis=-1)
    return float(err.mean())


def normalize_per_item(r):
    """r̂_l = r / ‖r‖ (逐元素单位化)."""
    norms = np.linalg.norm(r, axis=-1, keepdims=True).clip(1e-12)
    return r / norms


def normalize_per_codeword(C):
    """Ĉ = {c / ‖c‖} (逐码字单位化)."""
    norms = np.linalg.norm(C, axis=-1, keepdims=True).clip(1e-12)
    return C / norms


def compute_deltas_shape(r_lst_np, codebooks_np):
    """对单位化向量计算 Δ_1^{shape} 和 Δ_2^{shape}.

    公式:
      Δ_l^{shape} = [QErr(r̂_{l+1}; Ĉ_l) − QErr(r̂_{l+1}; Ĉ_{l+1})] / QErr(r̂_{l+1}; Ĉ_{l+1})
    """
    # 单位化
    r1_hat = normalize_per_item(r_lst_np[1])  # input to L1, 这是 r_1
    r2_hat = normalize_per_item(r_lst_np[2])  # input to L2, 这是 r_2
    r3_hat = normalize_per_item(r_lst_np[3]) if len(r_lst_np) > 3 else None  # input to L3, r_3
    C0_hat = normalize_per_codeword(codebooks_np[0])  # codebook L1
    C1_hat = normalize_per_codeword(codebooks_np[1])  # codebook L2
    C2_hat = normalize_per_codeword(codebooks_np[2]) if len(codebooks_np) > 2 else None  # codebook L3

    out = {}
    # Δ_1^{shape}: r_2_hat 在 Ĉ_0 vs Ĉ_1
    err_prev = qerr(r2_hat, C0_hat)
    err_curr = qerr(r2_hat, C1_hat)
    delta_1 = (err_prev - err_curr) / (err_curr + 1e-12)
    out['l=1->2'] = {
        'QErr_prev_layer_C_hat': err_prev,
        'QErr_curr_layer_C_hat': err_curr,
        'delta_l_shape': float(delta_1),
    }

    # Δ_2^{shape}: r_3_hat 在 Ĉ_1 vs Ĉ_2
    if r3_hat is not None and C2_hat is not None:
        err_prev = qerr(r3_hat, C1_hat)
        err_curr = qerr(r3_hat, C2_hat)
        delta_2 = (err_prev - err_curr) / (err_curr + 1e-12)
        out['l=2->3'] = {
            'QErr_prev_layer_C_hat': err_prev,
            'QErr_curr_layer_C_hat': err_curr,
            'delta_l_shape': float(delta_2),
        }

    return out


def main():
    print('=' * 70)
    print('Idea 1 现象 2 — 剥离尺度后方向层面失配 Δ_l^{shape}')
    print('=' * 70)
    out = {}
    for name, path in DATA.items():
        print(f'\n--- {name} ---')
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        codebooks = bundle['codebooks']
        r_lst_np = [r.float().numpy() for r in r_lst]
        cb_np = [c.float().numpy() for c in codebooks]
        deltas = compute_deltas_shape(r_lst_np, cb_np)
        print(f'  Δ_l^{{shape}} (逐码字单位化后):')
        for k, v in deltas.items():
            print(f'    {k}: QErr_C_prev_hat={v["QErr_prev_layer_C_hat"]:.6f}, '
                  f'QErr_C_curr_hat={v["QErr_curr_layer_C_hat"]:.6f}, '
                  f'Δ_l^{{shape}}={v["delta_l_shape"]:+.4f} ({v["delta_l_shape"]*100:+.1f}%)')

        # 比值 Δ_1^{shape}/Δ_2^{shape}
        d1s = abs(deltas['l=1->2']['delta_l_shape'])
        if 'l=2->3' in deltas:
            d2s = abs(deltas['l=2->3']['delta_l_shape'])
            ratio = d1s / (d2s + 1e-12)
            print(f'  |Δ_1^{{shape}}|/|Δ_2^{{shape}}| = {d1s:.4f}/{d2s:.4f} = {ratio:.2f}x')
            print(f'  原始 Δ_1/Δ_2 = 189.02x (AQ) / 2084.30x (WF)')
            if ratio < 3:
                kill = '✗ KILL — 方向断裂不成立 (比值 < 3x)'
            else:
                kill = '✓ pass — 方向断裂仍存在'
            print(f'  Kill 判定 (<3x): {kill}')
        else:
            ratio = None
            kill = '无法比较 (只有 2 层)'

        out[name] = {
            'deltas_shape': deltas,
            'abs_delta_1_shape': d1s,
            'abs_delta_2_shape': abs(deltas.get('l=2->3', {}).get('delta_l_shape', 0)),
            'shape_ratio': ratio,
            'kill_verdict': kill,
        }

    out_path = os.path.join(OUT_DIR, 'idea1_phen2_shape_only.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()