#!/usr/bin/env python3
# task17_mp_spike.py — task19 #6: 逐层 Marchenko-Pastur spike 判据 (v2 鲁棒版)
#
# 测什么: 3 算法 × 4 层 — 数显著超出 bulk 的 spike 数
# 数据: task16 cache (rqidx.pt), 3 算法
#
# 鲁棒方法:
#   - 算 G = r r^T / D 的所有 N'=2000 个 eigenvalues
#   - bulk = eigenvalues rank [100, 1900] (中间 90%)
#   - mp_upper (经验) = bulk_max × 1.2  (避免 σ² clamp 崩)
#   - n_spike = #{eigvals > mp_upper}
#   - 也算 MP 理论公式作交叉验证 (但只供打印)
#
# 倾向: 浅层 (L0) 应该有 spike, 深层 (L3) 应该没 spike (各向同性噪声)

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6'
TASK21_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task19'
os.makedirs(TASK21_DIR, exist_ok=True)

ALGORITHMS = ['A_baseline', 'B_mmq', 'C_gsrq']
SUBSAMPLE_N = 2000


def compute_mp_metrics(r, dim, n_subsample=SUBSAMPLE_N, seed=42):
    """鲁棒版 MP 判据.

    returns: dict with
      - n_spike_bulk5: count of eigvals > 5x bulk_max
      - n_spike_bulk3: count of eigvals > 3x bulk_max
      - bulk_max: bulk region max
      - bulk_median: bulk region median
      - eigval_max: top eigval
      - mp_upper_theory: σ²(1+√γ)² (理论, 仅参考)
      - n_spike_theory: count of eigvals > mp_upper_theory
      - top_eigval_to_bulk_ratio
    """
    N, D = r.shape
    rng = np.random.default_rng(seed)
    if N > n_subsample:
        idx_sub = rng.choice(N, size=n_subsample, replace=False)
        r_sub = r[idx_sub]
    else:
        r_sub = r
    N_eff = r_sub.shape[0]

    # Empirical covariance Σ = r^T r / N
    Sigma = (r_sub.T @ r_sub) / N_eff                  # (D, D)
    sigma2 = Sigma.diagonal().sum().item() / D          # trace/D

    # γ = D / N_eff (Wishart convention)
    gamma = D / N_eff
    mp_upper_theory = sigma2 * (1 + np.sqrt(gamma)) ** 2

    # Gram G = r r^T / D (smaller: N_eff x N_eff)
    G = (r_sub @ r_sub.T) / D                          # (N_eff, N_eff)
    G_sym = 0.5 * (G + G.T)
    eigvals = torch.linalg.eigvalsh(G_sym).cpu().numpy()  # ascending

    # bulk region: middle 90%
    bulk_lo = int(0.05 * N_eff)
    bulk_hi = int(0.95 * N_eff)
    bulk = eigvals[bulk_lo:bulk_hi]
    bulk_max = bulk.max()
    bulk_median = np.median(bulk)

    # Spike counts using bulk_max as threshold
    n_spike_5 = int((eigvals > 5 * bulk_max).sum())
    n_spike_3 = int((eigvals > 3 * bulk_max).sum())
    n_spike_theory = int((eigvals > mp_upper_theory).sum()) if sigma2 > 1e-10 else -1

    top10 = eigvals[-10:][::-1].tolist()               # descending
    eigval_max = top10[0]

    return {
        'sigma2': float(sigma2),
        'gamma': float(gamma),
        'mp_upper_theory': float(mp_upper_theory),
        'bulk_max': float(bulk_max),
        'bulk_median': float(bulk_median),
        'n_spike_bulk5': n_spike_5,
        'n_spike_bulk3': n_spike_3,
        'n_spike_theory': n_spike_theory,
        'top10_eigvals': top10,
        'eigval_max': float(eigval_max),
        'top_to_bulk_ratio': float(eigval_max / (bulk_max + 1e-12)),
    }


def main():
    print('=' * 70)
    print('task19 #6 v2 — MP spike 鲁棒判据 (bulk-relative spike count)')
    print('=' * 70)

    all_results = {}
    for name in ALGORITHMS:
        cache_path = os.path.join(OUT_DIR, f'{name}_rqidx.pt')
        print(f'\n=== {name}: loading {cache_path} ===')
        if not os.path.exists(cache_path):
            print(f'  cache missing, skipping')
            continue
        bundle = torch.load(cache_path, weights_only=False)
        r_lst = bundle['r_lst']
        L = len(r_lst)

        per_layer = {}
        for l in range(L):
            r = r_lst[l]
            D = r.shape[-1]
            mp = compute_mp_metrics(r, D)
            per_layer[l] = mp
            print(f'  Layer {l}: '
                  f'sigma2={mp["sigma2"]:.5f}, '
                  f'bulk_max={mp["bulk_max"]:.5f}, '
                  f'eigval_max={mp["eigval_max"]:.5f}, '
                  f'top/bulk={mp["top_to_bulk_ratio"]:.1f}, '
                  f'spike_5x={mp["n_spike_bulk5"]}, '
                  f'spike_3x={mp["n_spike_bulk3"]}, '
                  f'spike_theory={mp["n_spike_theory"]}')

        n_spike_seq = [per_layer[l]['n_spike_bulk5'] for l in range(L)]
        ratio_seq = [per_layer[l]['top_to_bulk_ratio'] for l in range(L)]
        print(f'  4 层 N_spike_5x 序列: {n_spike_seq}')
        print(f'  4 层 top/bulk_max 序列: {[round(v,1) for v in ratio_seq]}')

        all_results[name] = {
            'algorithm': name,
            'per_layer': per_layer,
            'n_spike_5x_sequence': n_spike_seq,
            'top_to_bulk_ratio_sequence': ratio_seq,
            'L': L,
        }

    # Save
    combined_path = os.path.join(TASK21_DIR, 'task17_mp_spike_per_layer.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n=== JSON → {combined_path} ===')

    for name, result in all_results.items():
        p = os.path.join(TASK21_DIR, f'task17_mp_spike_{name}.json')
        with open(p, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  → {p}')

    # ---------- Verdict ----------
    print('\n' + '=' * 70)
    print('task19 #6 v2 verdict')
    print('=' * 70)
    pass_count = 0
    fail_count = 0
    for name, result in all_results.items():
        spike_seq = result['n_spike_5x_sequence']
        ratio_seq = result['top_to_bulk_ratio_sequence']
        # Pass: deep spike 0 (or near 0) AND shallow spike > 0
        shallow_spike = spike_seq[0] if spike_seq else 0
        deep_spike = spike_seq[-1] if spike_seq else 0
        deep_ratio = ratio_seq[-1] if ratio_seq else 1.0
        shallow_ratio = ratio_seq[0] if ratio_seq else 1.0

        if deep_spike <= 2 and shallow_spike >= 1 and deep_ratio < 0.5 * shallow_ratio:
            tag = '✅ PASS'
            pass_count += 1
        elif deep_spike > shallow_spike:
            tag = '❌ FAIL (反噬)'
            fail_count += 1
        elif deep_spike >= 3:
            tag = '⚠️ WEAK (深层仍有 spike)'
            fail_count += 1
        else:
            tag = '○ NEUTRAL'
        print(f'  {name}: spike_5x={spike_seq}, top/bulk_max={[round(v,1) for v in ratio_seq]} → {tag}')

    print(f'\n=== pass={pass_count} fail={fail_count} ===')

    # ---------- Plot ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Row 1 col 0: N_spike_5x per layer
        ax = axes[0, 0]
        for name in ALGORITHMS:
            if name in all_results:
                seq = all_results[name]['n_spike_5x_sequence']
                ax.plot(range(len(seq)), seq, marker='o', label=name)
        ax.set_xlabel('Layer')
        ax.set_ylabel('N_spike (eigvals > 5× bulk_max)')
        ax.set_title('MP spike count (5×bulk) per layer')
        ax.set_xticks(range(4))
        ax.legend()
        ax.grid(alpha=0.3)

        # Row 1 col 1, 2: top10 eigenvalues per algorithm
        for idx, name in enumerate(ALGORITHMS):
            if idx + 1 > 2:
                break
            ax = axes[0, idx + 1]
            if name in all_results:
                for l in range(4):
                    top10 = all_results[name]['per_layer'][l]['top10_eigvals']
                    ax.semilogy(range(len(top10)), top10, marker='o',
                                label=f'L{l}', alpha=0.7)
                bulk_max = all_results[name]['per_layer'][3]['bulk_max']
                ax.axhline(y=bulk_max * 5, color='red', linestyle='--',
                           label=f'5×bulk_max L3')
            ax.set_xlabel('top eigenvalue index')
            ax.set_ylabel('eigenvalue (log)')
            ax.set_title(f'{name}: top-10 eigvals per layer')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # Row 2 col 0: σ² per layer
        ax = axes[1, 0]
        for name in ALGORITHMS:
            if name in all_results:
                s2_seq = [all_results[name]['per_layer'][l]['sigma2'] for l in range(4)]
                ax.plot(range(len(s2_seq)), s2_seq, marker='o', label=name)
        ax.set_xlabel('Layer')
        ax.set_ylabel('σ² = trace(Σ)/D')
        ax.set_title('Residual noise variance')
        ax.set_xticks(range(4))
        ax.legend()
        ax.grid(alpha=0.3)

        # Row 2 col 1: top_to_bulk_ratio per layer
        ax = axes[1, 1]
        for name in ALGORITHMS:
            if name in all_results:
                r_seq = all_results[name]['top_to_bulk_ratio_sequence']
                ax.semilogy(range(len(r_seq)), r_seq, marker='o', label=name)
        ax.set_xlabel('Layer')
        ax.set_ylabel('top_eigval / bulk_max (log)')
        ax.set_title('Top eigenvalue dominance')
        ax.set_xticks(range(4))
        ax.legend()
        ax.grid(alpha=0.3)

        # Row 2 col 2: bulk_max per layer per algorithm
        ax = axes[1, 2]
        for name in ALGORITHMS:
            if name in all_results:
                b_seq = [all_results[name]['per_layer'][l]['bulk_max'] for l in range(4)]
                ax.semilogy(range(len(b_seq)), b_seq, marker='o', label=name)
        ax.set_xlabel('Layer')
        ax.set_ylabel('bulk_max (log)')
        ax.set_title('Bulk upper edge per layer')
        ax.set_xticks(range(4))
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(TASK21_DIR, 'task17_mp_spike_fit.png')
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f'\n=== PNG → {png_path} ===')
    except Exception as e:
        print(f'Plot error: {e}')

    # ---------- Verdict.md ----------
    verdict_md = [
        '# task19 #6 v2 — MP spike 鲁棒判据 verdict',
        '',
        '## 测什么',
        '逐层数"显著超出 bulk (中间 90% eigvals) 的 spike 数"——验证深层残差谱退化为各向同性噪声',
        '',
        '## 3 算法结果',
        '',
        '| 算法 | N_spike_5x (4 层序列) | top_eigval/bulk_max (4 层) | verdict |',
        '|------|---------------------|--------------------------|---------|',
    ]
    for name in ALGORITHMS:
        if name in all_results:
            spike_seq = all_results[name]['n_spike_5x_sequence']
            ratio_seq = [round(v, 1) for v in all_results[name]['top_to_bulk_ratio_sequence']]
            shallow_spike = spike_seq[0] if spike_seq else 0
            deep_spike = spike_seq[-1] if spike_seq else 0
            deep_ratio = ratio_seq[-1] if ratio_seq else 1.0
            shallow_ratio = ratio_seq[0] if ratio_seq else 1.0

            if deep_spike <= 2 and shallow_spike >= 1 and deep_ratio < 0.5 * shallow_ratio:
                tag = '✅ PASS (各向同性噪声)'
            elif deep_spike > shallow_spike:
                tag = '❌ FAIL (反噬)'
            elif deep_spike >= 3:
                tag = '⚠️ WEAK (深层仍有 spike)'
            else:
                tag = '○ NEUTRAL'
            verdict_md.append(f'| {name} | {spike_seq} | {ratio_seq} | {tag} |')

    verdict_md.extend([
        '',
        '## kill 线',
        '- **pass**: 深层 L3 N_spike ≤ 2 且 L3 top/bulk < 0.5 × L0 → "深层是各向同性噪声" 成立',
        '- **fail**: 深层 L3 N_spike > L0 → 反噬, 深层残留信号强',
        '- **中间**: 深层 N_spike 在 [1, 3] 或 ratio 不衰减 → 待解释',
        '',
        '## 鲁棒性',
        '- 用 bulk_max × 5 替代 σ²(1+√γ)² 公式: 避免 σ² clamp 到 0 导致 MP upper 接近 0',
        '- bulk = rank [5%, 95%] 的 eigenvalues (中间 90%)',
        '- subsample N=2000 / D=2048 → γ≈0.97, 期望 spike 出现在 top-50',
        '',
        '## 文件',
        '- JSON: `result/task19/task17_mp_spike_per_layer.json`',
        '- PNG:  `result/task19/task17_mp_spike_fit.png`',
        '- 单算法: `result/task19/task17_mp_spike_{A,B,C}_*.json`',
    ])
    verdict_path = os.path.join(TASK21_DIR, 'task17_mp_spike_verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_md))
    print(f'\n=== Verdict → {verdict_path} ===')


if __name__ == '__main__':
    main()
