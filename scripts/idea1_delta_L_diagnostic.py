#!/usr/bin/env python3
"""Idea 1 补测: Δ_l (跨层码本失配)

公式 (按用户原设计):
QErr(r; C) = Σ_i ||r_i - c_assign(i)||²
Δ_l = [QErr(r_{l+1}; C_l) − QErr(r_{l+1}; C_{l+1})] / QErr(r_{l+1}; C_{l+1})

Δ_l 表示用"前一层码本"代替"本层码本"去量化当前残差时,误差的相对膨胀倍数。
- Δ_l ≈ 0: layer l+1 没学到新东西, 跟 layer l 复用码本无异 (redundancy)
- Δ_l 大: layer l+1 确实学到了新东西 (有 information gain)

用户原设计假设: WF 比 AQ 更"平滑" (各层 Δ_l 落差小)
kill 线:
1. WF 落差比 AQ 更陡 (反向现象)
2. 深层 V-info 没有提升 (L2 已经 -30.6% 踩线)

判定: 任一 kill 线 踩中 → Idea 1 判死
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'

IDX_FILES = {
    'A_baseline (AQ, K=256^3)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'retrained WF (K=[256,64,16])': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def qerr(r, C):
    """Σ_i ||r_i - argmin_c ||r_i - c||² · c||² (总平方量化误差)."""
    # Squared distance
    r_norm2 = (r ** 2).sum(-1, keepdims=True)
    c_norm2 = (C ** 2).sum(-1)
    cross = r @ C.T
    d2 = r_norm2 + c_norm2[None, :] - 2 * cross
    d2 = np.maximum(d2, 0)
    nearest_idx = d2.argmin(axis=-1)
    nearest_c = C[nearest_idx]
    err = ((r - nearest_c) ** 2).sum(axis=-1)
    return float(err.mean())  # per-item average for stability


def compute_deltas(r_lst, q_lst, codebooks):
    """
    r_lst[0] = x, r_lst[1..3] = layer l input
    codebooks[l] for layer l

    For each l = 1, 2:
      r_in = r_lst[l]   (this is the input to layer l, also the residual after layer l-1)
      codebook_for_l = codebooks[l-1]
      target_codebook = codebooks[l]
      QErr(r_in, codebook_for_l) = what if we reuse previous layer's codebook?
      QErr(r_in, target_codebook) = what layer l actually achieved
    """
    deltas = {}
    for l in range(1, len(r_lst)):  # l = 1, 2 -> 0..N_layers-2 in codebooks
        if l >= len(codebooks):
            break
        r_in = r_lst[l].numpy().astype(np.float32)
        C_prev = codebooks[l-1].float().numpy()  # previous layer's codebook
        C_curr = codebooks[l].float().numpy()    # current layer's codebook
        err_prev = qerr(r_in, C_prev)
        err_curr = qerr(r_in, C_curr)
        delta_l = (err_prev - err_curr) / (err_curr + 1e-12)
        deltas[f'l={l}->{l+1}'] = {
            'QErr_prev_layer_C': err_prev,
            'QErr_curr_layer_C': err_curr,
            'delta_l': float(delta_l),
        }
    return deltas


def main():
    print('=' * 70)
    print('Idea 1 补测 — Δ_l 跨层码本失配 (按用户公式)')
    print('=' * 70)
    results = {}
    for name, path in IDX_FILES.items():
        print(f'\n--- {name} ---')
        if not os.path.exists(path):
            print(f'  FILE NOT FOUND: {path}')
            continue
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        idx_lst = bundle['idx_lst']
        codebooks = bundle['codebooks']
        # Verify: r_lst[i] for the input to codebook l
        # Per task14_unified_rerun: r_lst[0]=x, r_lst[1]=input to layer 0... in our case r_lst[1]=input to L1, etc.
        # Actually let's check
        deltas = compute_deltas(r_lst, q_lst, codebooks)
        # Also: V-info per layer (for kill line 2)
        # r_in_to_l (1..N-1) maps to layer l indexing
        v_info_per_layer_input = []
        for l in range(1, len(r_lst)):
            r_in = r_lst[l].numpy().astype(np.float32)
            # Mean V-info via Y proxy from q_l (no need separate Y)
            # Use K-Means on r_in as Y proxy for now
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=3, batch_size=1024).fit(r_in)
            c = km.labels_.astype(np.int64)
            # Self-MI? Actually we want I(q_l; Y) where Y is global
            # For per-layer "usable info" use cluster entropy
            from collections import Counter
            cnt = Counter(c.tolist())
            p = np.array([v for v in cnt.values()])/len(c)
            H = float(-(p * np.log2(p + 1e-12)).sum())
            v_info_per_layer_input.append(H)
        print(f'  Per-layer input entropy (cluster H proxy):')
        for l, h in enumerate(v_info_per_layer_input):
            print(f'    input_to_layer_{l+1}: H={h:.4f} bits')
        print(f'\n  Δ_l = [QErr(r_l; C_{l-1}) − QErr(r_l; C_l)] / QErr(r_l; C_l)')
        for k, v in deltas.items():
            print(f'    {k}: QErr_C_prev={v["QErr_prev_layer_C"]:.4f}, '
                  f'QErr_C_curr={v["QErr_curr_layer_C"]:.4f}, '
                  f'Δ_l={v["delta_l"]:+.4f} ({v["delta_l"]*100:+.1f}%)')
        results[name] = {
            'deltas': deltas,
            'input_entropy_per_layer': v_info_per_layer_input,
        }

    # Kill判定
    print('\n' + '=' * 70)
    print('KILL LINE 判定')
    print('=' * 70)
    if 'A_baseline (AQ, K=256^3)' in results and 'retrained WF (K=[256,64,16])' in results:
        aq = results['A_baseline (AQ, K=256^3)']['deltas']
        wf = results['retrained WF (K=[256,64,16])']['deltas']
        # AQ vs WF Δ_ls
        print('  AQ Δ_l:')
        for k, v in aq.items():
            print(f'    {k}: Δ_l = {v["delta_l"]:+.4f}')
        print('  WF Δ_l:')
        for k, v in wf.items():
            print(f'    {k}: Δ_l = {v["delta_l"]:+.4f}')

        # Kill line 1: WF 落差比 AQ 更陡
        # 落差 = max(Δ_l) - min(Δ_l)
        aq_drop = max(v['delta_l'] for v in aq.values()) - min(v['delta_l'] for v in aq.values())
        wf_drop = max(v['delta_l'] for v in wf.values()) - min(v['delta_l'] for v in wf.values())
        print(f'\n  AQ 落差 (max-min): {aq_drop:.4f}')
        print(f'  WF 落差 (max-min): {wf_drop:.4f}')
        if wf_drop > aq_drop:
            kill1 = '✗ KILL — WF 落差比 AQ 大 (steepness 上升)'
        else:
            kill1 = '✓ WF 落差 ≤ AQ 落差'

        # Kill line 2: deep V-info 没提升 (already known: L2 V-info -30.6%)
        # Hard-coded from previous comparison
        kill2 = '✗ KILL — L2 V-info 已经 -30.6% (踩线)'

        print(f'  Kill line 1: {kill1}')
        print(f'  Kill line 2: {kill2}')
        if kill1.startswith('✗') or kill2.startswith('✗'):
            verdict = '✗ KILL — Idea 1 现象 2 判定死'
        else:
            verdict = '✓ PASS — Idea 1 进入下一阶段'
        print(f'\n  VERDICT: {verdict}')

    out_path = os.path.join(OUT_DIR, 'idea1_WF_delta_L.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()