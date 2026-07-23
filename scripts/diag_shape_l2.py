#!/usr/bin/env python3
"""
#2 Δ_2^shape 反向检验: Test whether "second-to-third layer is smooth" is real or scale artifact.

For each of 6 algorithms (A_baseline, B_mmq, C_gsrq, AQ, HRQ, WF):
1. Load r_lst and codebooks.
2. Compute Δ_2_raw = (QErr(r_3, C_2) - QErr(r_3, C_3)) / QErr(r_3, C_3)
3. Compute Δ_2^shape: unit-normalize r_3 per-item and C_2/C_3 per-codeword, recompute.
4. Compute shrinkage_ratio = (Δ_2_raw - Δ_2_shape) / Δ_2_raw.

Kill line (user-specified):
- shrinkage_ratio >= 50% for most algorithms -> rephrase claim as "scale alignment artifact"
- shrinkage_ratio <= 30% for all algorithms -> original claim stands
"""

import os
import sys
import json
import csv
import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_shape_l2'

BUNDLES = {
    'A_baseline':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':       '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'idea1_WF':    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
    'HRQ':         '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/hrq_rqidx.pt',
    'AQ':          '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/aq_rqidx.pt',
}


def qerr(r, C):
    """Mean per-item squared L2 distance to nearest codeword.

    r: (N, D) numpy array
    C: (W, D) numpy array
    Returns float = mean_i ||r_i - nearest_C(r_i)||^2
    """
    r = np.ascontiguousarray(r)
    C = np.ascontiguousarray(C)
    r_norm2 = (r ** 2).sum(-1, keepdims=True)        # (N, 1)
    c_norm2 = (C ** 2).sum(-1)                       # (W,)
    cross = r @ C.T                                  # (N, W)
    d2 = np.maximum(r_norm2 + c_norm2[None, :] - 2 * cross, 0)
    nearest_idx = d2.argmin(axis=-1)
    nearest_c = C[nearest_idx]                       # (N, D)
    err = ((r - nearest_c) ** 2).sum(axis=-1)        # (N,)
    return float(err.mean()), nearest_idx


def normalize_per_item(r):
    """r̂ = r / ||r|| per row."""
    norms = np.linalg.norm(r, axis=-1, keepdims=True).clip(1e-12)
    return r / norms


def normalize_per_codeword(C):
    """Ĉ = {c / ||c||, ∀c ∈ C} per codeword."""
    norms = np.linalg.norm(C, axis=-1, keepdims=True).clip(1e-12)
    return C / norms


def compute_delta_l(r_target, C_prev, C_curr):
    """Δ_l = (QErr(r; C_prev) - QErr(r; C_curr)) / QErr(r; C_curr)."""
    err_prev, _ = qerr(r_target, C_prev)
    err_curr, _ = qerr(r_target, C_curr)
    return (err_prev - err_curr) / max(err_curr, 1e-12), err_prev, err_curr


def main():
    print('=' * 70)
    print('#2 Δ_2^shape 反向检验')
    print('=' * 70)

    results = {}

    for algo, path in BUNDLES.items():
        if not os.path.exists(path):
            print(f'\n--- {algo}: ckpt NOT FOUND ({path}), skip ---')
            continue
        print(f'\n--- {algo} ---')
        print(f'  loading {path}')
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        codebooks = bundle['codebooks']
        gains = bundle.get('gains', [None] * len(codebooks))

        # Apply gain scaling (RQ-VAE has cluster_gains that scale centroids)
        effective_codebooks = []
        for l, C in enumerate(codebooks):
            if gains[l] is not None:
                g = gains[l].view(-1, 1)
                effective_codebooks.append((C * g).float().numpy())
            else:
                effective_codebooks.append(C.float().numpy())
        r_lst_np = [r.float().numpy() for r in r_lst]

        # r_3 is residual entering L3 (index 3 in r_lst)
        r_3 = r_lst_np[3]
        C_2 = effective_codebooks[1]   # previous layer
        C_3 = effective_codebooks[2]   # current layer

        # 1) raw Δ_2
        delta_2_raw, err_2_prev_raw, err_2_curr_raw = compute_delta_l(r_3, C_2, C_3)

        # 2) shape-only Δ_2: unit-normalize r_3 per-item and C_2/C_3 per-codeword
        r_3_hat = normalize_per_item(r_3)
        C_2_hat = normalize_per_codeword(C_2)
        C_3_hat = normalize_per_codeword(C_3)

        delta_2_shape, err_2_prev_shape, err_2_curr_shape = compute_delta_l(r_3_hat, C_2_hat, C_3_hat)

        # 3) shrinkage ratio
        if abs(delta_2_raw) > 1e-9:
            shrinkage_ratio = (delta_2_raw - delta_2_shape) / delta_2_raw
        else:
            shrinkage_ratio = float('nan')

        print(f'  RAW:        QErr(r_3, C_2) = {err_2_prev_raw:.4f}, QErr(r_3, C_3) = {err_2_curr_raw:.4f}, '
              f'Δ_2_raw = {delta_2_raw:+.4f} ({delta_2_raw*100:+.2f}%)')
        print(f'  SHAPE:      QErr(r̂_3, Ĉ_2) = {err_2_prev_shape:.4f}, QErr(r̂_3, Ĉ_3) = {err_2_curr_shape:.4f}, '
              f'Δ_2_shape = {delta_2_shape:+.4f} ({delta_2_shape*100:+.2f}%)')
        print(f'  SHRINKAGE:  (Δ_2_raw - Δ_2_shape) / Δ_2_raw = {shrinkage_ratio:.4f} '
              f'({shrinkage_ratio*100:.1f}%)')

        # Verify per-item ||r̂|| = 1 and per-codeword ||ĉ|| = 1
        r_3_norms = np.linalg.norm(r_3_hat, axis=-1)
        C_2_norms = np.linalg.norm(C_2_hat, axis=-1)
        C_3_norms = np.linalg.norm(C_3_hat, axis=-1)
        print(f'  norms check: r̂_3 ∈ [{r_3_norms.min():.4f}, {r_3_norms.max():.4f}], '
              f'Ĉ_2 ∈ [{C_2_norms.min():.4f}, {C_2_norms.max():.4f}], '
              f'Ĉ_3 ∈ [{C_3_norms.min():.4f}, {C_3_norms.max():.4f}]')

        results[algo] = {
            'raw': {
                'QErr_prev_layer_C': err_2_prev_raw,
                'QErr_curr_layer_C': err_2_curr_raw,
                'delta_2_raw': delta_2_raw,
            },
            'shape': {
                'QErr_prev_layer_C_hat': err_2_prev_shape,
                'QErr_curr_layer_C_hat': err_2_curr_shape,
                'delta_2_shape': delta_2_shape,
            },
            'shrinkage_ratio': shrinkage_ratio,
        }

    # ---------- CSV output ----------
    csv_path = os.path.join(OUT_DIR, 'delta2_compare_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algo', 'delta_2_raw', 'delta_2_shape', 'shrinkage_ratio'])
        for algo, r in results.items():
            writer.writerow([
                algo,
                f"{r['raw']['delta_2_raw']:.6f}",
                f"{r['shape']['delta_2_shape']:.6f}",
                f"{r['shrinkage_ratio']:.6f}",
            ])
    print(f'\nCSV → {csv_path}')

    # ---------- JSON output ----------
    json_path = os.path.join(OUT_DIR, 'delta2_compare_table.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'JSON → {json_path}')

    # ---------- Verdict ----------
    # Compute aggregate
    shrinkages = [abs(r['shrinkage_ratio']) for r in results.values() if not np.isnan(r['shrinkage_ratio'])]
    n_above_50 = sum(1 for s in shrinkages if s >= 0.50)
    n_below_30 = sum(1 for s in shrinkages if s <= 0.30)
    median_shrink = float(np.median(shrinkages))
    mean_shrink = float(np.mean(shrinkages))

    # Kill line
    if n_above_50 >= len(shrinkages) // 2:
        verdict_kind = 'SCALE_ALIGNMENT_ARTIFACT'
        verdict_msg = (
            f'shrinkage_ratio >= 50% for {n_above_50}/{len(shrinkages)} algorithms — '
            'the claim "second-to-third layer is smooth" must be rephrased as '
            '"appears smooth due to scale alignment, not necessarily shape match"'
        )
    elif n_below_30 == len(shrinkages):
        verdict_kind = 'ORIGINAL_CLAIM_STANDS'
        verdict_msg = (
            f'shrinkage_ratio <= 30% for all {len(shrinkages)} algorithms — '
            'original claim stands: the smoothness is shape-real, not scale-driven'
        )
    else:
        verdict_kind = 'MIXED'
        verdict_msg = (
            f'shrinkage_ratio is mixed ({n_above_50} above 50%, {n_below_30} below 30%, '
            f'median={median_shrink:.3f}) — claim should be qualified per-algorithm'
        )

    # ---------- Build verdict.md ----------
    verdict_lines = [
        '# #2 Δ_2^shape 反向检验 — 判定',
        '',
        '**问题**：第二层到第三层的"平滑过渡"是真实的方向匹配，还是因为 codeword 幅度对齐造成的伪影？',
        '',
        '**方法**：',
        '1. Δ_2^raw = (QErr(r_3, C_2) − QErr(r_3, C_3)) / QErr(r_3, C_3) — 不做归一',
        '2. Δ_2^shape = (QErr(r̂_3, Ĉ_2) − QErr(r̂_3, Ĉ_3)) / QErr(r̂_3, Ĉ_3) — r̂ = r/‖r‖ (per-item), Ĉ = c/‖c‖ (per-codeword)',
        '3. shrinkage_ratio = (Δ_2^raw − Δ_2^shape) / Δ_2^raw — 比例越大说明越依赖尺度',
        '',
        '**Kill line**：',
        '- 若 shrinkage_ratio ≥ 50% 占大多数算法 → claim 必须改写为"尺度对齐造成的平滑感"',
        '- 若 shrinkage_ratio ≤ 30% 对所有算法 → 原始 claim 成立',
        '',
        '## Per-algorithm table',
        '',
        '| 算法 | Δ_2^raw | Δ_2^shape | shrinkage_ratio | 解读 |',
        '|------|---------|-----------|-----------------|------|',
    ]
    for algo, r in results.items():
        d_raw = r['raw']['delta_2_raw']
        d_shape = r['shape']['delta_2_shape']
        sr = r['shrinkage_ratio']
        if abs(sr) >= 0.50:
            interp = '尺度主导 (>50% shrinkage)'
        elif abs(sr) <= 0.30:
            interp = '形状主导 (<30% shrinkage)'
        else:
            interp = '尺度+形状混合'
        verdict_lines.append(
            f'| {algo} | {d_raw:+.4f} ({d_raw*100:+.2f}%) | {d_shape:+.4f} ({d_shape*100:+.2f}%) | '
            f'{sr:.4f} ({sr*100:+.1f}%) | {interp} |'
        )

    verdict_lines.extend([
        '',
        '## Aggregate',
        '',
        f'- 6 个算法的 shrinkage_ratio 绝对值：{[f"{s:.3f}" for s in shrinkages]}',
        f'- 计数 ≥ 50% (尺度主导): {n_above_50} / {len(shrinkages)}',
        f'- 计数 ≤ 30% (形状主导): {n_below_30} / {len(shrinkages)}',
        f'- 中位数: {median_shrink:.4f} ({median_shrink*100:.1f}%)',
        f'- 平均: {mean_shrink:.4f} ({mean_shrink*100:.1f}%)',
        '',
        '## 判定结果',
        '',
        f'**类别**: {verdict_kind}',
        '',
        f'**说明**: {verdict_msg}',
        '',
    ])

    verdict_path = os.path.join(OUT_DIR, 'verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_lines))
    print(f'\nVerdict → {verdict_path}')

    # ---------- Print summary ----------
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Verdict kind: {verdict_kind}')
    print(f'Median shrinkage: {median_shrink*100:.1f}%')
    print(f'# >=50%: {n_above_50} / {len(shrinkages)}')
    print(f'# <=30%: {n_below_30} / {len(shrinkages)}')
    print(f'\n{verdict_msg}')

    return {
        'results': results,
        'verdict_kind': verdict_kind,
        'median_shrinkage': median_shrink,
        'n_above_50': n_above_50,
        'n_below_30': n_below_30,
    }


if __name__ == '__main__':
    main()