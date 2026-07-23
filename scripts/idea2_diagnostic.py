#!/usr/bin/env python3
"""Idea 2 现象 1 + 2 + 3: 互信息解耦 (L1 vs 深层)

现象 1: NMI(c_1, c_l) + shuffle baseline (bootstrap CI)
现象 2: 去冗余 r_l^⊥ = r_l - q_1·[proj] 后 V-info 变化
现象 3: 交叉判定 inject vs decouple vs both
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json, time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

CF_PATH      = os.path.join(OUT_DIR, 'cf_ppmi_svd256.pt')
EMB_PATH     = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}
SID_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-08/00-32-52/pickle/merged_predictions_tensor.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_c_s22/pickle/merged_predictions_tensor.pt',
}

N_MI_SAMPLE = 2000
N_BOOT = 200
PCA_RED_DIM = 16
ALPHA_GRID = [0.1, 0.3, 0.5]


def fast_mi_bits(x, y, n_sample=N_MI_SAMPLE, seed=42, n_neighbors=5):
    if x.ndim == 2 and x.shape[1] > 32:
        x = PCA(n_components=PCA_RED_DIM, random_state=seed).fit_transform(x)
    N = x.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    mi_nats = mutual_info_classif(x_s, y_s, n_neighbors=n_neighbors,
                                   random_state=seed, discrete_features=False, n_jobs=4)
    return float(mi_nats.mean() / np.log(2))


def empirical_mi_bits(joint_xy, marginal_x, marginal_y):
    """Compute empirical MI bits from joint and marginal counts.
    joint_xy: (Kx, Ky) matrix of counts; sums to N.
    """
    N = joint_xy.sum()
    pxy = joint_xy / N
    px = marginal_x / N
    py = marginal_y / N
    # Avoid log(0)
    outer = np.outer(px, py) + 1e-12
    pxy_ = pxy + 1e-12
    mi = np.sum(pxy * np.log(pxy_ / outer))
    return float(mi / np.log(2))


def normalized_mi(joint_xy, marginal_x, marginal_y):
    """NMI = I(X;Y) / min(H(X), H(Y)), in bits."""
    N = joint_xy.sum()
    mi = empirical_mi_bits(joint_xy, marginal_x, marginal_y)
    p_x = (marginal_x / N) + 1e-12
    p_y = (marginal_y / N) + 1e-12
    h_x = -float(np.sum(p_x * np.log2(p_x)))
    h_y = -float(np.sum(p_y * np.log2(p_y)))
    denom = min(h_x, h_y) + 1e-12
    return float(mi / denom), mi, h_x, h_y


def build_y(emb_path, K=20):
    x = torch.load(emb_path, map_location='cpu', weights_only=False).float()
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024).fit(x.numpy())
    return km.labels_.astype(np.int64)


def main():
    print('=' * 70)
    print('Loading SIDs and r_lst for A/B/C')
    print('=' * 70)
    sids = {}
    r_lst_dict = {}
    for name in RQIDX_PATHS:
        sids[name] = torch.load(SID_PATHS[name], map_location='cpu', weights_only=False).long().numpy()  # (N, 4) for A/C; (N, 4) for B
        print(f'  {name} SID shape: {sids[name].shape}')
        bundle = torch.load(RQIDX_PATHS[name], map_location='cpu', weights_only=False)
        r_lst_dict[name] = bundle['r_lst']

    # ==================== 现象 1: NMI(c_1, c_l) ====================
    print('\n' + '=' * 70)
    print('现象 1: NMI(c_1, c_l) + shuffle baseline')
    print('=' * 70)
    nmi_results = {}
    for name in sids:
        sid = sids[name]
        # Use only first 3 layers for NMI (idx_lst is 3, not 4)
        c1 = sid[:, 0]    # (N,) — layer 1 codes
        nmi_results[name] = {}
        for l in (1, 2):  # c_2 (idx 1), c_3 (idx 2)
            cl = sid[:, l]
            K1 = int(c1.max() + 1)
            Kl = int(cl.max() + 1)
            # Joint count
            joint = np.zeros((K1, Kl), dtype=np.int64)
            np.add.at(joint, (c1, cl), 1)
            marg_x = joint.sum(axis=1)
            marg_y = joint.sum(axis=0)
            nmi, mi, h_x, h_y = normalized_mi(joint, marg_x, marg_y)
            # Shuffle baseline: shuffle cl, recompute NMI
            rng = np.random.default_rng(42)
            shuf_nmis = np.empty(N_BOOT)
            for b in range(N_BOOT):
                cl_shuf = cl[rng.permutation(len(cl))]
                joint_s = np.zeros((K1, Kl), dtype=np.int64)
                np.add.at(joint_s, (c1, cl_shuf), 1)
                marg_s = joint_s.sum(axis=1)
                marg_y_s = joint_s.sum(axis=0)
                nmi_s, _, _, _ = normalized_mi(joint_s, marg_s, marg_y_s)
                shuf_nmis[b] = nmi_s
            shuf_mean = float(shuf_nmis.mean())
            shuf_std = float(shuf_nmis.std())
            shuf_p99 = float(np.quantile(shuf_nmis, 0.99))
            obs_minus_shuf_mean = nmi - shuf_mean
            obs_minus_shuf_std = (nmi - shuf_mean) / (shuf_std + 1e-12)
            nmi_results[name][f'l{l+1}'] = {
                'nmi':           nmi,
                'mi_bits':       mi,
                'h_x':           h_x,
                'h_y':           h_y,
                'shuf_mean':     shuf_mean,
                'shuf_std':      shuf_std,
                'shuf_p99':      shuf_p99,
                'z_score':       obs_minus_shuf_std,
                'significant':   bool(obs_minus_shuf_std > 3),
                'n_codes_c1':    K1,
                'n_codes_cl':    Kl,
            }
            print(f'  {name} NMI(c_1, c_{l+1}) = {nmi:.4f}, '
                  f'shuf mean={shuf_mean:.4f}, shuf p99={shuf_p99:.4f}, '
                  f'z={obs_minus_shuf_std:.1f} '
                  f'{"✓sig" if obs_minus_shuf_std>3 else "✗noise"}')

    out_path = os.path.join(OUT_DIR, 'idea2_phenomenon1_nmi.json')
    with open(out_path, 'w') as f:
        json.dump(nmi_results, f, indent=2)
    print(f'\n=== Saved → {out_path} ===')

    # ==================== 现象 2: 去冗余后 V-info ====================
    print('\n' + '=' * 70)
    print('现象 2: 去冗余 r_l^⊥ = r_l - q_1·[proj_q1] 后 V-info')
    print('=' * 70)
    print('Loading Y...')
    y = build_y(EMB_PATH, K=20)

    decomp_results = {}
    for name in r_lst_dict:
        print(f'\n--- {name} ---')
        bundle = torch.load(RQIDX_PATHS[name], map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        decomp_results[name] = {}
        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            q_1 = q_lst[0].numpy().astype(np.float32)  # (N, D) L1 codebook output
            # Linear regression: r_l = q_1 @ beta + r_l_perp
            # beta = (q_1.T @ q_1)^-1 @ q_1.T @ r_l
            # r_l_perp = r_l - q_1 @ beta
            q1T_q1_inv = np.linalg.pinv(q_1.T @ q_1 + 1e-4 * np.eye(q_1.shape[1]))
            beta = q1T_q1_inv @ q_1.T @ r_l  # (D, D)
            r_l_perp = r_l - q_1 @ beta
            mi_base = fast_mi_bits(r_l, y)
            mi_perp = fast_mi_bits(r_l_perp, y)
            delta_v = mi_perp - mi_base
            decomp_results[name][f'l{l}'] = {
                'v_info_base':  mi_base,
                'v_info_perp':  mi_perp,
                'delta_v_decouple': delta_v,
                'r_norm_mean':  float(np.linalg.norm(r_l, axis=-1).mean()),
                'perp_norm_mean': float(np.linalg.norm(r_l_perp, axis=-1).mean()),
            }
            print(f'  L{l}: I_V(r_l)={mi_base:.4f}, I_V(r_l⊥)={mi_perp:.4f}, '
                  f'Δ={delta_v:+.4f}  '
                  f'{"✓ >0" if delta_v>0 else "✗ ≤0"}')

    out_path = os.path.join(OUT_DIR, 'idea2_phenomenon2_decouple.json')
    with open(out_path, 'w') as f:
        json.dump(decomp_results, f, indent=2)
    print(f'\n=== Saved → {out_path} ===')

    # ==================== 现象 3: 交叉判定 (inject vs decouple vs both) ====================
    print('\n' + '=' * 70)
    print('现象 3: 交叉判定 inject vs decouple vs both')
    print('=' * 70)
    cf = torch.load(CF_PATH, map_location='cpu', weights_only=False)
    f = cf['f_normalized'].numpy().astype(np.float32)
    rng = np.random.default_rng(123)
    f_shuffled = f[rng.permutation(f.shape[0])]

    cross_results = {}
    # Pre-load idea 1 results
    p1 = json.load(open(os.path.join(OUT_DIR, 'idea1_phenomenon2_3.json')))
    base_vinfo = p1['baseline_vinfo']

    for name in r_lst_dict:
        bundle = torch.load(RQIDX_PATHS[name], map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        cross_results[name] = {}
        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            q_1 = q_lst[0].numpy().astype(np.float32)
            # Step 1: get r_l_perp
            q1T_q1_inv = np.linalg.pinv(q_1.T @ q_1 + 1e-4 * np.eye(q_1.shape[1]))
            beta = q1T_q1_inv @ q_1.T @ r_l
            r_l_perp = r_l - q_1 @ beta

            # Step 2: fit Proj on r_l_perp (more "available" subspace than full r_l)
            pca = PCA(n_components=64, random_state=42)
            r_perp_reduced = pca.fit_transform(r_l_perp)
            A, *_ = np.linalg.lstsq(f, r_perp_reduced, rcond=None)
            proj_lifted = (f @ A) @ pca.components_

            # Get alpha=0.3 baseline from idea 1 (it's the most stable)
            alpha = 0.3
            mi_base = base_vinfo[name][f'l{l}']
            mi_perp = fast_mi_bits(r_l_perp, y)
            mi_inj = p1['alpha_sweep'][name][f'l{l}'][f'alpha{alpha}']['v_info_inject']
            mi_both = fast_mi_bits(r_l_perp + alpha * proj_lifted, y)

            delta_inject   = mi_inj   - mi_base
            delta_decouple = mi_perp  - mi_base
            delta_both     = mi_both  - mi_base
            # additive check
            additive_pred = delta_inject + delta_decouple
            cross_results[name][f'l{l}'] = {
                'mi_base': mi_base,
                'mi_perp': mi_perp,
                'mi_inj':  mi_inj,
                'mi_both': mi_both,
                'delta_inject':   delta_inject,
                'delta_decouple': delta_decouple,
                'delta_both':     delta_both,
                'additive_pred':  additive_pred,
                'additive_error': delta_both - additive_pred,
            }
            print(f'  {name} L{l}: ΔI_V(inj)={delta_inject:+.4f}, '
                  f'ΔI_V(decouple)={delta_decouple:+.4f}, '
                  f'ΔI_V(both)={delta_both:+.4f}, '
                  f'additive_pred={additive_pred:+.4f}, '
                  f'err={delta_both-additive_pred:+.4f}')

    out_path = os.path.join(OUT_DIR, 'idea2_phenomenon3_cross.json')
    with open(out_path, 'w') as f:
        json.dump(cross_results, f, indent=2)
    print(f'\n=== Saved → {out_path} ===')

    # Final verdict
    print('\n' + '=' * 70)
    print('FINAL VERDICT — Idea 1 + Idea 2')
    print('=' * 70)
    print('\n[Idea 1]')
    for name in cross_results:
        for lk in sorted(cross_results[name].keys()):
            di = cross_results[name][lk]['delta_inject']
            print(f'  {name} {lk}: ΔI_V(inject)={di:+.4f} '
                  f'{"✓kill alive" if di>0.1 else "✗kill (Idea 1 dead)"}')
    print('\n[Idea 2 现象 1 NMI]')
    for name in nmi_results:
        for lk in nmi_results[name]:
            r = nmi_results[name][lk]
            print(f'  {name} {lk}: NMI={r["nmi"]:.4f}, shuf p99={r["shuf_p99"]:.4f}, '
                  f'z={r["z_score"]:.1f} {"✓sig" if r["significant"] else "✗noise"}')
    print('\n[Idea 2 现象 2 去冗余]')
    for name in decomp_results:
        for lk in decomp_results[name]:
            d = decomp_results[name][lk]['delta_v_decouple']
            print(f'  {name} {lk}: ΔI_V(decouple)={d:+.4f} '
                  f'{"✓kill alive" if d>0 else "✗kill (Idea 2 dead)"}')
    print('\n[Idea 2 现象 3 交叉判定]')
    for name in cross_results:
        for lk in sorted(cross_results[name].keys()):
            d = cross_results[name][lk]
            di, dd, db = d['delta_inject'], d['delta_decouple'], d['delta_both']
            print(f'  {name} {lk}: inj={di:+.4f}, decouple={dd:+.4f}, both={db:+.4f}')


if __name__ == '__main__':
    main()