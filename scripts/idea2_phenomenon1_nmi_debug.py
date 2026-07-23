#!/usr/bin/env python3
"""Idea 2 现象 1 修复版 — NMI(c_1, c_l) 严格数值

修复：empirical_mi 改用 mask-only 求和（避免 0 项的 log(0) 误差）；
shuffle baseline 重新生成。
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

SID_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-08/00-32-52/pickle/merged_predictions_tensor.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_c_s22/pickle/merged_predictions_tensor.pt',
}
N_BOOT = 500


def mi_bits_mask(joint):
    """Robust MI bits: sum only over pxy > 0 entries.
    H(X) and H(Y) computed from marginals (also masked)."""
    N = joint.sum()
    pxy = joint / N
    px = joint.sum(axis=1) / N
    py = joint.sum(axis=0) / N
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
    mi_bits = float(mi / np.log(2))
    # Entropies
    mask_x = px > 0
    mask_y = py > 0
    h_x = float(-np.sum(px[mask_x] * np.log2(px[mask_x])))
    h_y = float(-np.sum(py[mask_y] * np.log2(py[mask_y])))
    denom = min(h_x, h_y)
    nmi = mi_bits / (denom + 1e-12)
    return float(nmi), mi_bits, h_x, h_y


def nmi_from_codes(c1, cl):
    K1 = int(max(c1.max(), cl.max())) + 1
    # Use K1 = max of both → square matrix
    K1 = max(K1, int(c1.max() + 1), int(cl.max() + 1))
    joint = np.zeros((K1, K1), dtype=np.int64)
    np.add.at(joint, (c1, cl), 1)
    return mi_bits_mask(joint)


def main():
    print('=' * 70)
    print('Loading SIDs for A/B/C (3-layer RQ code, shape (N, 3+1))')
    print('=' * 70)
    sids = {}
    for name, p in SID_PATHS.items():
        s = torch.load(p, map_location='cpu', weights_only=False).long().numpy()
        print(f'  {name}: shape={s.shape}, dtype={s.dtype}')
        # SID is (4, N_items) [layers, items] → take first 3 layers, transpose to (N_items, 3)
        sids[name] = s[:3, :].T
        print(f'    after transpose: shape={sids[name].shape}')
        print(f'    c_1 range=[{sids[name][:,0].min()},{sids[name][:,0].max()}], '
              f'c_2 range=[{sids[name][:,1].min()},{sids[name][:,1].max()}], '
              f'c_3 range=[{sids[name][:,2].min()},{sids[name][:,2].max()}]')

    nmi_results = {}
    for name, sid in sids.items():
        c1 = sid[:, 0]
        nmi_results[name] = {}
        for l in (1, 2):
            cl = sid[:, l]
            nmi_obs, mi_obs, h_x, h_y = nmi_from_codes(c1, cl)
            # Shuffle baseline
            rng = np.random.default_rng(42)
            shuf_nmis = np.empty(N_BOOT)
            shuf_mis = np.empty(N_BOOT)
            for b in range(N_BOOT):
                cl_shuf = cl[rng.permutation(len(cl))]
                nmi_s, mi_s, _, _ = nmi_from_codes(c1, cl_shuf)
                shuf_nmis[b] = nmi_s
                shuf_mis[b] = mi_s
            shuf_mean_nmi = float(shuf_nmis.mean())
            shuf_std_nmi = float(shuf_nmis.std())
            shuf_p99_nmi = float(np.quantile(shuf_nmis, 0.99))
            z_nmi = (nmi_obs - shuf_mean_nmi) / (shuf_std_nmi + 1e-12)
            # Also: MI bits vs shuffle MI
            shuf_mean_mi = float(shuf_mis.mean())
            shuf_std_mi = float(shuf_mis.std())
            shuf_p99_mi = float(np.quantile(shuf_mis, 0.99))
            z_mi = (mi_obs - shuf_mean_mi) / (shuf_std_mi + 1e-12)
            nmi_results[name][f'l{l+1}'] = {
                'nmi_obs': nmi_obs,
                'mi_obs':  mi_obs,
                'h_x':     h_x,
                'h_y':     h_y,
                'shuf_nmi_mean': shuf_mean_nmi,
                'shuf_nmi_std':  shuf_std_nmi,
                'shuf_nmi_p99':  shuf_p99_nmi,
                'z_nmi':         z_nmi,
                'shuf_mi_mean':  shuf_mean_mi,
                'shuf_mi_p99':   shuf_p99_mi,
                'z_mi':          z_mi,
                'significant':   bool(z_mi > 3),
                'n_codes_c1': int(c1.max() + 1),
                'n_codes_cl': int(cl.max() + 1),
                'N':           int(len(c1)),
            }
            print(f'  {name} NMI(c_1,c_{l+1})={nmi_obs:.4f} '
                  f'(shuf={shuf_mean_nmi:.4f}±{shuf_std_nmi:.4f}, '
                  f'p99={shuf_p99_nmi:.4f}, z={z_nmi:.1f}); '
                  f'I_bits={mi_obs:.4f} (shuf p99={shuf_p99_mi:.4f}, z={z_mi:.1f}) '
                  f'{"✓sig" if z_mi > 3 else "✗noise"}')

    out_path = os.path.join(OUT_DIR, 'idea2_phenomenon1_nmi_v2.json')
    with open(out_path, 'w') as f:
        json.dump(nmi_results, f, indent=2)
    print(f'\n=== Saved → {out_path} ===')

    print('\n' + '=' * 70)
    print('VERDICT — Idea 2 现象 1 (NMI/c_1 vs c_l)')
    print('=' * 70)
    for name, d in nmi_results.items():
        for lk, lr in d.items():
            print(f'  {name} {lk}: NMI={lr["nmi_obs"]:.4f} '
                  f'(shuf p99={lr["shuf_nmi_p99"]:.4f}); '
                  f'I_bits={lr["mi_obs"]:.4f} '
                  f'(shuf p99={lr["shuf_mi_p99"]:.4f}, z={lr["z_mi"]:.1f}) '
                  f'{"✓sig" if lr["z_mi"]>3 else "✗noise"}')


if __name__ == '__main__':
    main()