#!/usr/bin/env python3
# task5_batch2.py — 11 诊断量批 2：3 个跨层一致性诊断量
# 量 5: D_l (cluster 内 distance profile)
# 量 6: η_l (local intrinsic dimension)
# 量 7: SW(残差分布跨层 Sliced Wasserstein)
#
# 数据：task16 cache (rqidx.pt)，3 算法 (A_baseline / B_mmq / C_gsrq)
#       复用前向 r_lst, q_lst, idx_lst（避免重做 forward）

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6'   # task17 复用 task16 cache
TASK19_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task17'
os.makedirs(TASK19_DIR, exist_ok=True)

ALGORITHMS = ['A_baseline', 'B_mmq', 'C_gsrq']
L = 3

# ---------- Quantity 5: cluster 内 distance profile ----------
def compute_cluster_distance_profile(r_lst, q_lst, idx_lst):
    """D_l = {mean(||r_l - q_l||) per cluster}: 反映 cluster 内部的拟合质量"""
    mean_dists = []
    std_dists = []
    max_dists = []
    for l in range(L):
        r = r_lst[l]                                 # (N, D)
        q = q_lst[l]                                 # (N, D)
        idx = idx_lst[l]                             # (N,)
        dists = (r - q).norm(dim=-1)                 # (N,)
        # cluster-wise stats
        cluster_means = []
        cluster_stds = []
        cluster_max = []
        for k in idx.unique():
            mask = (idx == k)
            d_in = dists[mask]
            if d_in.numel() > 0:
                cluster_means.append(d_in.mean().item())
                cluster_stds.append(d_in.std().item())
                cluster_max.append(d_in.max().item())
        mean_dists.append(float(np.mean(cluster_means)))
        std_dists.append(float(np.mean(cluster_stds)))
        max_dists.append(float(np.mean(cluster_max)))
    return {
        'mean_dist_l': mean_dists,
        'std_dist_l':  std_dists,
        'max_dist_l':  max_dists,
    }

# ---------- Quantity 6: local intrinsic dimension η_l ----------
def compute_local_intrinsic_dim(r_lst, k=10):
    """η_l = MLE intrinsic dim estimator (Levina-Bickel)
       η̂ = [1/k * Σ log(T_k/T_i)]^{-1}  (T_k 是第 k 邻居距离，T_i 是第 i 邻居)
       反映残差流形的本征维度
       clip 到 [1, D] 防止除零爆
    """
    etas = []
    rng = np.random.default_rng(42)
    sample_n = 2000
    D_max = 2048   # 上界 = embedding 维度
    for l in range(L):
        r = r_lst[l]
        if r.shape[0] > sample_n:
            idx = rng.choice(r.shape[0], size=sample_n, replace=False)
            r_sub = r[idx]
        else:
            r_sub = r
        a_norm2 = (r_sub ** 2).sum(-1, keepdim=True)             # (n, 1)
        b_norm2 = (r_sub ** 2).sum(-1)                           # (n,)
        d2 = a_norm2 - 2 * (r_sub @ r_sub.T) + b_norm2           # (n, n)
        d2.clamp_(min=0)
        d_topk, _ = d2.topk(k + 1, dim=1, largest=False)        # (n, k+1)
        d_topk = d_topk[:, 1:].clamp(min=1e-12).sqrt()            # 去掉自己 (n, k)
        # MLE: η̂_i = (k-1) / Σ log(T_k/T_j)
        # clip log ratio 防止除零爆（log(1+1e-6) ≈ 1e-6）
        T_k = d_topk[:, -1:].clamp(min=1e-12)
        T_j = d_topk[:, :-1].clamp(min=1e-12)
        log_ratios = torch.log(T_k / T_j).clamp(min=1e-6)        # 下限保护
        eta_i = (k - 1) / log_ratios.sum(-1).clamp(min=1e-6)
        eta_i = eta_i.clamp(min=1.0, max=D_max)                  # clip 到 [1, D]
        eta_l = eta_i.median().item()                            # median 比 mean 鲁棒
        etas.append(eta_l)
    return {'eta_l': etas}

# ---------- Quantity 7: Sliced Wasserstein r̂_l → r̂_{l+1} ----------
def sliced_wasserstein(a, b, n_proj=64, seed=42):
    """SW_2(a, b): 1D Wasserstein on random projections, averaged
       a, b: (N, D) tensors
       Returns scalar SW_2
    """
    rng = np.random.default_rng(seed)
    D = a.shape[-1]
    sw_vals = []
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    for _ in range(n_proj):
        proj = rng.standard_normal(D).astype(np.float32)
        proj /= np.linalg.norm(proj) + 1e-12
        pa = (a_np @ proj).flatten()                # (N,)
        pb = (b_np @ proj).flatten()
        # Sort both
        pa_sorted = np.sort(pa)
        pb_sorted = np.sort(pb)
        # 1D Wasserstein-2 (squared)
        w2 = ((pa_sorted - pb_sorted) ** 2).mean()
        sw_vals.append(w2 ** 0.5)
    return float(np.mean(sw_vals))

def compute_cross_layer_sw(r_lst, sample_n=1500):
    """SW_l = SW2(r̂_l, r̂_{l+1}) 反映相邻层残差分布的差异（漂移速度）"""
    rng = np.random.default_rng(42)
    sw_list = []
    for l in range(L - 1):
        r_l = r_lst[l]
        r_lp = r_lst[l + 1]
        # Subsample for speed
        if r_l.shape[0] > sample_n:
            idx = rng.choice(r_l.shape[0], size=sample_n, replace=False)
            r_l_sub = r_l[idx]
            r_lp_sub = r_lp[idx]
        else:
            r_l_sub = r_l
            r_lp_sub = r_lp
        sw_l = sliced_wasserstein(r_l_sub, r_lp_sub, n_proj=64)
        sw_list.append(sw_l)
    return {'sw_l': sw_list}  # length L-1

# ---------- Main ----------
def main():
    print('=' * 60)
    print('task15_batch2 — 3 个跨层诊断量')
    print('=' * 60)

    all_results = {}
    for name in ALGORITHMS:
        cache_path = os.path.join(OUT_DIR, f'{name}_rqidx.pt')
        print(f'\n=== {name}: loading {cache_path} ===')
        if not os.path.exists(cache_path):
            print(f'  cache missing, skipping')
            continue
        bundle = torch.load(cache_path, weights_only=False)
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        idx_lst = bundle['idx_lst']
        print(f'  loaded {len(r_lst)} r, {len(q_lst)} q, {len(idx_lst)} idx')

        # Quantity 5
        q5 = compute_cluster_distance_profile(r_lst, q_lst, idx_lst)
        # Quantity 6
        q6 = compute_local_intrinsic_dim(r_lst)
        # Quantity 7
        q7 = compute_cross_layer_sw(r_lst)

        result = {
            'algorithm': name,
            'cache': cache_path,
            'layers': []
        }
        for l in range(L):
            entry = {
                'l': l + 1,
                'mean_dist_l': q5['mean_dist_l'][l],
                'std_dist_l':  q5['std_dist_l'][l],
                'max_dist_l':  q5['max_dist_l'][l],
                'eta_l':       q6['eta_l'][l],
            }
            if l < L - 1:
                entry[f'sw_l_to_{l+2}'] = q7['sw_l'][l]
            result['layers'].append(entry)
        # SW cross-layer
        result['cross_layer_sw'] = q7['sw_l']

        all_results[name] = result
        print(f'  q5 mean_dist: {[round(v, 4) for v in q5["mean_dist_l"]]}')
        print(f'  q6 eta:       {[round(v, 2) for v in q6["eta_l"]]}')
        print(f'  q7 SW_l:      {[round(v, 4) for v in q7["sw_l"]]}')

    # Save
    combined_path = os.path.join(TASK19_DIR, 'task5_all_algorithms.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n=== Combined JSON → {combined_path} ===')

    for name, result in all_results.items():
        p = os.path.join(TASK19_DIR, f'task15_{name}.json')
        with open(p, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  → {p}')

    # ---------- Plots ----------
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

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        # 5: cluster mean dist
        for name, result in all_results.items():
            ls = [r['l'] for r in result['layers']]
            md = [r['mean_dist_l'] for r in result['layers']]
            ax[0].plot(ls, md, marker='o', color=COLORS[name], label=LABELS[name], linewidth=2)
        ax[0].set_xlabel('layer l')
        ax[0].set_ylabel(r'$D_l$ (mean $\|r_l - q_l\|$ per cluster)')
        ax[0].set_title('Quantity 5: cluster-fit distance  $D_l$')
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        # 6: eta_l
        for name, result in all_results.items():
            ls = [r['l'] for r in result['layers']]
            et = [r['eta_l'] for r in result['layers']]
            ax[1].plot(ls, et, marker='o', color=COLORS[name], label=LABELS[name], linewidth=2)
        ax[1].set_xlabel('layer l')
        ax[1].set_ylabel(r'$\eta_l$ (Levina-Bickel MLE)')
        ax[1].set_title('Quantity 6: local intrinsic dim  $\eta_l$')
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

        # 7: SW cross-layer
        for name, result in all_results.items():
            ls = list(range(1, L))   # SW_l is defined for l=1,2 (between layers)
            sw = result['cross_layer_sw']
            ax[2].plot(ls, sw, marker='o', color=COLORS[name], label=LABELS[name], linewidth=2)
        ax[2].set_xlabel(r'transition $\ell \to \ell+1$')
        ax[2].set_ylabel(r'$SW_2(\hat r_l, \hat r_{l+1})$')
        ax[2].set_title('Quantity 7: cross-layer SW')
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(TASK19_DIR, 'task15_batch2_quantities.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f'\n=== PNG → {TASK19_DIR}/task15_batch2_quantities.png ===')
    except Exception as e:
        print(f'plot failed: {e}')

    # Verdict
    verdict_lines = ['# Task 19 (Diag11 批 2) 一段话判据\n']
    for name, result in all_results.items():
        verdict_lines.append(f'\n## {name}\n')
        for r in result['layers']:
            l = r['l']
            sw_str = ''
            if f'sw_l_to_{l+1}' in r:
                sw_str = f", SW_l→{l+1}={r[f'sw_l_to_{l+1}']:.4f}"
            elif f'sw_l_to_{l+2}' in r:
                sw_str = f", SW_l→{l+1}={r[f'sw_l_to_{l+2}']:.4f}"
            verdict_lines.append(
                f"- Layer {l}: mean_D={r['mean_dist_l']:.4f}±{r['std_dist_l']:.4f}, "
                f"max_D={r['max_dist_l']:.4f}, η={r['eta_l']:.2f}{sw_str}"
            )

    verdict_lines.append('\n## 整体判定（task16 + task17 联合）\n')
    verdict_lines.append('（待后续分析基于 task16 + task17 数据写出）')

    verdict_path = os.path.join(TASK19_DIR, 'task5_verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_lines))
    print(f'\n=== Verdict → {verdict_path} ===')

if __name__ == '__main__':
    main()