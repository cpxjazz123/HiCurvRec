#!/usr/bin/env python3
# task4_plot_only.py — 从已有 JSON 数据生成 PNG 图
# 避免重算 embedding/ckpt（前向 + SVD 90s）

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6'

# Load combined JSON
with open(os.path.join(OUT_DIR, 'task4_all_algorithms.json')) as f:
    all_results = json.load(f)

# Colors for consistency
COLORS = {
    'A_baseline': '#1f77b4',  # blue
    'B_mmq':      '#ff7f0e',  # orange
    'C_gsrq':     '#2ca02c',  # green
}
LABELS = {
    'A_baseline': 'A baseline (Euclid, no normalize)',
    'B_mmq':      'B MMQ (cosine, normalize=True)',
    'C_gsrq':     'C GSRQ (gain-shape, no normalize)',
}

# ---------- Plot 1: 残差幅度剖面 + erank + m_l ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

# 1a: mu_l ± sigma_l
for name, result in all_results.items():
    ls = [r['l'] for r in result['layers']]
    mus = [r['mu_l'] for r in result['layers']]
    sigmas = [r['sigma_l'] for r in result['layers']]
    ax[0].plot(ls, mus, marker='o', color=COLORS[name], label=LABELS[name])
    ax[0].fill_between(ls, [m - s for m, s in zip(mus, sigmas)],
                       [m + s for m, s in zip(mus, sigmas)],
                       alpha=0.15, color=COLORS[name])
ax[0].set_xlabel('layer l')
ax[0].set_ylabel(r'$\|r_l\|$ (residual norm)')
ax[0].set_title('量 1: 残差幅度剖面  $\mu_l \pm \sigma_l$')
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

# 1b: erank_l
for name, result in all_results.items():
    ls = [r['l'] for r in result['layers']]
    eranks = [r['erank_l'] for r in result['layers']]
    ax[1].plot(ls, eranks, marker='o', color=COLORS[name], label=LABELS[name], linewidth=2)
ax[1].set_xlabel('layer l')
ax[1].set_ylabel('effective rank  $\mathrm{erank}_l = \exp H(\hat\lambda)$')
ax[1].set_title('量 2: 谱有效秩  $\mathrm{erank}_l$')
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

# 1c: m_l
for name, result in all_results.items():
    ls = [r['l'] for r in result['layers']]
    ms = [r['m_l'] for r in result['layers']]
    ax[2].bar([l + (0.27 * list(all_results.keys()).index(name) - 0.27) for l in ls],
              ms, width=0.27, color=COLORS[name], label=LABELS[name], alpha=0.85)
ax[2].set_xlabel('layer l')
ax[2].set_ylabel(r'$m_l$ (above threshold)')
ax[2].set_title('量 3: 水填充维度  $m_l$')
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'task14_layer_norm_and_rank.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f'  → task14_layer_norm_and_rank.png')

# ---------- Plot 2: f_radial + cell CV (gain-shape decomp) ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

for name, result in all_results.items():
    ls = [r['l'] for r in result['layers']]
    fr = [r['f_radial_l'] for r in result['layers']]
    ax[0].plot(ls, fr, marker='o', color=COLORS[name], label=LABELS[name], linewidth=2)
ax[0].set_xlabel('layer l')
ax[0].set_ylabel(r'$f_{\mathrm{radial},l}$ (radial error / total)')
ax[0].set_title('量 4a: 量化误差径向占比  $f_{\mathrm{radial},l}$')
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_ylim([0, 1])

for name, result in all_results.items():
    ls = [r['l'] for r in result['layers']]
    cv = [r['cell_cv_l'] for r in result['layers']]
    ax[1].plot(ls, cv, marker='o', color=COLORS[name], label=LABELS[name], linewidth=2)
ax[1].set_xlabel('layer l')
ax[1].set_ylabel(r'cell CV  $(\sigma(r) / \mu(r))$ within cluster')
ax[1].set_title('量 4b: cluster 内幅度变异  cell CV')
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'task14_gain_shape_decomp.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f'  → task14_gain_shape_decomp.png')

# ---------- Plot 3: 跨算法对比横截面（每层）----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

quantities = [
    ('mu_l', '$\mu_l$', 0, 0),
    ('erank_l', '$\mathrm{erank}_l$', 0, 1),
    ('m_l', '$m_l$', 0, 2),
    ('sigma_l', '$\sigma_l$', 1, 0),
    ('f_radial_l', '$f_{\mathrm{radial},l}$', 1, 1),
    ('cell_cv_l', 'cell CV', 1, 2),
]

for qkey, qlabel, row, col in quantities:
    ax = axes[row][col]
    x = np.arange(len(list(all_results.values())[0]['layers']))
    width = 0.25
    for i, (name, result) in enumerate(all_results.items()):
        vals = [r[qkey] for r in result['layers']]
        ax.bar(x + (i - 1) * width, vals, width, color=COLORS[name], label=LABELS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([f'l={r["l"]}' for r in list(all_results.values())[0]['layers']])
    ax.set_ylabel(qlabel)
    ax.set_title(qlabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'task14_cross_algorithm.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f'  → task14_cross_algorithm.png')

print('\n所有 PNG 出图完成。')