#!/usr/bin/env python3
"""Idea 1 现象 1: 纯尺度模型 — 验证尺度差异本身能解释多少 Δ_1=15.9

目的:
  - 算出 E[‖r_l‖], E[‖c^(l)‖]
  - 构造纯尺度蒙特卡洛模型: 码字方向完全相同 (单位球面均匀), 仅整体缩放不同
  - 在该假设下, 预测 Δ_1^{scale-only}
  - 对比 Δ_1 实测 = 15.8963 (AQ) / 14.43 (WF)

Kill 线:
  - 若 Δ_1^{scale-only} 占 Δ_1 实测 > 80%, "非平稳性"叙事站不住
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


def compute_residual_and_codebook_norms(bundle):
    """从 bundle 取:
      r_lst: list of tensors, r_lst[0]=x, r_lst[1]=input to L1, r_lst[2]=input to L2
      codebooks: list of codebook tensors, codebooks[l] = layer l's codebook
    返回:
      E[‖r_l‖] for l in {0,1} (即 input to layer 1, input to layer 2)
      E[‖c^(l)‖] for l in {0,1} (codebook of layer 1, layer 2)
    """
    r_lst = bundle['r_lst']
    codebooks = bundle['codebooks']

    r_norms = {}
    c_norms = {}
    for l in range(1, len(r_lst)):
        r = r_lst[l].float().numpy()
        r_norms[f'l={l}'] = float(np.linalg.norm(r, axis=-1).mean())

    for l, C in enumerate(codebooks):
        Cnp = C.float().numpy()
        c_norms[f'l={l}'] = float(np.linalg.norm(Cnp, axis=-1).mean())

    return r_norms, c_norms


def scale_only_monte_carlo(r_norms, c_norms, r_lst_np, codebooks_np, n_seeds=5, n_samples=10000):
    """构造纯尺度模型:
      - r_lst[l]: 保持方向分布 (实际单位化方向), 仅整体缩放成 target 范数 E[‖r_l‖]
      - codebooks[l]: 保持方向分布 (实际单位化方向), 仅整体缩放成 target 范数 E[‖c^(l)‖]
      - 重算 Δ_1^{scale-only}

    假设: 码字方向分布完全一样 → 我们用真实的单位化方向, 但替换范数.
    """
    rng = np.random.default_rng(42)

    # 真实单位化方向
    r1 = r_lst_np[1]
    r1_unit = r1 / np.linalg.norm(r1, axis=-1, keepdims=True).clip(1e-12)
    r2 = r_lst_np[2]
    r2_unit = r2 / np.linalg.norm(r2, axis=-1, keepdims=True).clip(1e-12)

    C1 = codebooks_np[0]  # codebook layer 1
    C1_unit = C1 / np.linalg.norm(C1, axis=-1, keepdims=True).clip(1e-12)
    C2 = codebooks_np[1]  # codebook layer 2
    C2_unit = C2 / np.linalg.norm(C2, axis=-1, keepdims=True).clip(1e-12)

    # 目标范数 (实测)
    target_r1 = r_norms['l=1']
    target_r2 = r_norms['l=2']
    target_C1 = c_norms['l=0']
    target_C2 = c_norms['l=1']

    # 用真实方向 + 目标范数构造纯尺度向量
    r1_scale = r1_unit * target_r1
    r2_scale = r2_unit * target_r2
    C1_scale = C1_unit * target_C1
    C2_scale = C2_unit * target_C2

    # 计算 Δ_1^{scale-only} 用 r_2 (输入 layer 2) 在 C_1 vs C_2 上的 QErr
    # 公式: Δ_l = [QErr(r_{l+1}; C_l) − QErr(r_{l+1}; C_{l+1})] / QErr(r_{l+1}; C_{l+1})
    # l=1: r_2 + C_1 vs C_2
    err_prev = qerr(r2_scale, C1_scale)  # 用 layer 1 码本去量化 r_2
    err_curr = qerr(r2_scale, C2_scale)  # 用 layer 2 码本去量化 r_2
    delta_scale_only = (err_prev - err_curr) / (err_curr + 1e-12)
    return {
        'QErr_scale_prev': err_prev,
        'QErr_scale_curr': err_curr,
        'delta_scale_only': float(delta_scale_only),
    }


def main():
    print('=' * 70)
    print('Idea 1 现象 1 — 纯尺度模型验证')
    print('=' * 70)
    out = {}
    for name, path in DATA.items():
        print(f'\n--- {name} ---')
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        codebooks = bundle['codebooks']
        r_norms, c_norms = compute_residual_and_codebook_norms(bundle)
        r_lst_np = [r.float().numpy() for r in r_lst]
        cb_np = [c.float().numpy() for c in codebooks]
        print(f'  范数实测:')
        print(f'    E[‖r_l‖] for l=1,2 = {r_norms}')
        print(f'    E[‖c^(l)‖] for l=0,1 = {c_norms}')

        # 实测 Δ (用真实数据)
        from idea1_delta_L_diagnostic import compute_deltas
        deltas = compute_deltas(r_lst, None, codebooks)
        delta_1_real = deltas['l=1->2']['delta_l']
        delta_2_real = deltas['l=2->3']['delta_l']
        print(f'  实测 Δ_1 = {delta_1_real:.4f}, Δ_2 = {delta_2_real:.4f}')
        print(f'  实测 Δ_1/Δ_2 = {delta_1_real/delta_2_real:.2f}x')

        # 纯尺度模型预测
        scale_result = scale_only_monte_carlo(r_norms, c_norms, r_lst_np, cb_np)
        delta_scale_only = scale_result['delta_scale_only']
        print(f'  纯尺度预测 Δ_1^scale-only = {delta_scale_only:.4f}')
        ratio = delta_scale_only / (delta_1_real + 1e-12) * 100
        print(f'  占比 = {ratio:.1f}% (实测 Δ_1={delta_1_real:.4f})')

        kill = '✗ KILL — 非平稳性叙事站不住' if ratio > 80 else '✓ pass — 尺度不足以解释'
        print(f'  Kill 判定 (>80%): {kill}')

        out[name] = {
            'r_norms': r_norms,
            'c_norms': c_norms,
            'delta_1_real': delta_1_real,
            'delta_2_real': delta_2_real,
            'scale_only': scale_result,
            'scale_only_pct': ratio,
            'kill_verdict': kill,
        }

    out_path = os.path.join(OUT_DIR, 'idea1_phen1_scale_only.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()