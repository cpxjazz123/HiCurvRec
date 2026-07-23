#!/usr/bin/env python3
# task6_q11.py — 跨层码本失配 Δ_l
# 量 11: Δ_l = (QErr_C_l - QErr_C_{l+1}) / QErr_C_{l+1}
# 数据：task16 cache r_lst / codebooks
#
# 判 A idea: Δ_l > 0.1 表明跨层形状变化真有量化代价

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

OUT18 = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6'
TASK20_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task18'
os.makedirs(TASK20_DIR, exist_ok=True)

ALGORITHMS = ['A_baseline', 'B_mmq', 'C_gsrq']
L = 3

def quantize_err(r, codebook, has_gains, gain=None):
    """QErr(r, C): 用码本 C 量化 r 的平均重构误差
       r: (N, D)
       codebook: (W, D) or list if has_gains
       返回标量 L2 误差均值
    """
    if has_gains and gain is not None:
        eff_C = codebook * gain.unsqueeze(-1)
    else:
        eff_C = codebook
    # Nearest neighbor (chunked for memory)
    r_norm2 = (r ** 2).sum(-1, keepdim=True)
    C_norm2 = (eff_C ** 2).sum(-1)
    d2 = r_norm2 - 2 * (r @ eff_C.T) + C_norm2
    idx = d2.argmin(dim=1)
    q = eff_C[idx]
    err = ((r - q) ** 2).sum(-1).mean().item()
    return err

def main():
    print('=' * 60)
    print('task16_q11 — 跨层码本失配 Δ_l')
    print('=' * 60)

    all_results = {}
    for name in ALGORITHMS:
        cache_path = os.path.join(OUT18, f'{name}_rqidx.pt')
        print(f'\n=== {name}: loading {cache_path} ===')
        if not os.path.exists(cache_path):
            print(f'  cache missing, skipping')
            continue
        bundle = torch.load(cache_path, weights_only=False)
        r_lst = bundle['r_lst']                          # 4 个 r: r_0 = x, r_1, r_2, r_3
        codebooks = bundle['codebooks']                  # 3 个码本
        gains = bundle['gains']                          # 3 个 gain (C 才有)
        has_gains = any(g is not None for g in gains)

        # 量 11: 对每个 l ∈ [0, 1, ..., L-2]
        # 用 C_l 量化 r_{l+1} 与用 C_{l+1} 量化 r_{l+1} 比较
        delta_l_list = []
        err_with_C_l_list = []
        err_with_C_lp1_list = []
        for l in range(L - 1):
            r_lp1 = r_lst[l + 1]                         # 用 r_{l+1} 量化
            # 用第 l 层码本
            err_with_l = quantize_err(r_lp1, codebooks[l],
                                       has_gains, gains[l] if has_gains else None)
            # 用第 l+1 层码本
            err_with_lp1 = quantize_err(r_lp1, codebooks[l + 1],
                                         has_gains, gains[l + 1] if has_gains else None)
            delta_l = (err_with_l - err_with_lp1) / (err_with_lp1 + 1e-12)
            delta_l_list.append(delta_l)
            err_with_C_l_list.append(err_with_l)
            err_with_C_lp1_list.append(err_with_lp1)
            print(f'  l={l+1}→{l+2}: QErr_C{l+1}={err_with_l:.4f}, '
                  f'QErr_C{l+2}={err_with_lp1:.4f}, Δ={delta_l:.4f}')

        all_results[name] = {
            'algorithm': name,
            'has_gains': has_gains,
            'delta_l': delta_l_list,
            'QErr_C_l': err_with_C_l_list,
            'QErr_C_lp1': err_with_C_lp1_list,
        }
        print(f'  Δ_l: {[round(v, 4) for v in delta_l_list]}')

    # Save
    combined_path = os.path.join(TASK20_DIR, 'task6_q11_all.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n=== JSON → {combined_path} ===')

    for name, result in all_results.items():
        p = os.path.join(TASK20_DIR, f'task16_q11_{name}.json')
        with open(p, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  → {p}')

    # ---------- Plot ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        COLORS = {'A_baseline': '#1f77b4', 'B_mmq': '#ff7f0e', 'C_gsrq': '#2ca02c'}
        LABELS = {
            'A_baseline': 'A baseline (Euclid, no normalize)',
            'B_mmq':      'B MMQ (cosine, normalize=True)',
            'C_gsrq':     'C GSRQ (gain-shape, no normalize)',
        }

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：QErr_C_l vs QErr_C_l+1
        for name, result in all_results.items():
            ls = list(range(1, L))   # l = 1, 2 (L-1 个)
            ax[0].plot(ls, result['QErr_C_l'], marker='o', linestyle='--',
                       color=COLORS[name], alpha=0.6, label=f'{LABELS[name]} (C_l)')
            ax[0].plot(ls, result['QErr_C_lp1'], marker='o', linestyle='-',
                       color=COLORS[name], label=f'{LABELS[name]} (C_l+1)')
        ax[0].set_xlabel(r'transition $\ell \to \ell+1$')
        ax[0].set_ylabel(r'QErr (mean $\|r-q\|^2$)')
        ax[0].set_title('Cross-layer quant error')
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)

        # 右图：Δ_l 柱状
        x = np.arange(L - 1)
        width = 0.25
        for i, (name, result) in enumerate(all_results.items()):
            ax[1].bar(x + (i - 1) * width, result['delta_l'], width,
                       color=COLORS[name], label=LABELS[name], alpha=0.85)
        ax[1].axhline(0.1, color='red', linestyle=':', linewidth=1, label='A-matters threshold (0.1)')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f'l={l+1}→{l+2}' for l in range(L - 1)])
        ax[1].set_ylabel(r'$\Delta_\ell = (QErr_{C_\ell} - QErr_{C_{\ell+1}}) / QErr_{C_{\ell+1}}$')
        ax[1].set_title('Quantity 11: cross-layer codebook mismatch  $\\Delta_\\ell$')
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(TASK20_DIR, 'task16_q11_cross_layer_mismatch.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f'\n=== PNG → {TASK20_DIR}/task16_q11_cross_layer_mismatch.png ===')
    except Exception as e:
        print(f'plot failed: {e}')

    # ---------- Verdict ----------
    verdict_lines = ['# Task 20 量 11 — 跨层码本失配 Δ_l\n']
    verdict_lines.append('\n## 量化结果\n')
    verdict_lines.append('| Algorithm | Δ_1 (l=1→2) | Δ_2 (l=2→3) | A matters? |')
    verdict_lines.append('|-----------|--------------|--------------|------------|')
    for name, result in all_results.items():
        d1 = result['delta_l'][0] if len(result['delta_l']) > 0 else None
        d2 = result['delta_l'][1] if len(result['delta_l']) > 1 else None
        d1_str = f'{d1:.4f}' if d1 is not None else '-'
        d2_str = f'{d2:.4f}' if d2 is not None else '-'
        # A matters: Δ > 0.1 任意层
        matters = 'YES' if (d1 is not None and d1 > 0.1) or (d2 is not None and d2 > 0.1) else 'NO'
        verdict_lines.append(f'| {name} | {d1_str} | {d2_str} | {matters} |')

    verdict_lines.append('\n## 判 A idea\n')
    for name, result in all_results.items():
        deltas = result['delta_l']
        max_d = max(deltas)
        if max_d > 0.1:
            verdict_lines.append(f'- **{name}**: Δ_max = {max_d:.4f} > 0.1 → **A idea matters** (跨层码本失配有量化代价)')
        else:
            verdict_lines.append(f'- **{name}**: Δ_max = {max_d:.4f} < 0.1 → A idea 弱证据（变化对量化代价小）')

    verdict_lines.append('\n## 整体判定\n')
    verdict_lines.append('（基于 task16+task17+task18 数据综合）')
    verdict_path = os.path.join(TASK20_DIR, 'task6_q11_verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_lines))
    print(f'\n=== Verdict → {verdict_path} ===')

if __name__ == '__main__':
    main()