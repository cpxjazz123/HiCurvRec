#!/usr/bin/env python3
# diag_joint_summary_chart.py — 配套脚本: 画 misleading_frac 横条图 + corr 热图

import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial'
CSV = os.path.join(OUT, 'misleading_frac_table.csv')

rows = []
with open(CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

algos = sorted(set(r['algo'] for r in rows))
layers = [1, 2, 3]

# Plot 1: misleading_frac by algo and layer
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(algos))
width = 0.25
for i, L in enumerate(layers):
    vals = [float(next(r['misleading_frac'] for r in rows if r['algo'] == a and int(r['layer']) == L)) * 100
            for a in algos]
    ax.bar(x + (i - 1) * width, vals, width, label=f'L{L}')
ax.set_xticks(x)
ax.set_xticklabels(algos, rotation=20)
ax.set_ylabel('misleading fraction (%)')
ax.set_title('Misleading fraction per algo and layer (f_r > 0.5 ∧ |cos| < 0.2)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUT, 'misleading_frac_bars.png')
plt.savefig(out1, dpi=120)
plt.close()
print(f'saved → {out1}')

# Plot 2: corr(cos, f_radial) heatmap
mat = np.zeros((len(algos), len(layers)))
for i, a in enumerate(algos):
    for j, L in enumerate(layers):
        mat[i, j] = float(next(r['corr_cos_f_radial'] for r in rows if r['algo'] == a and int(r['layer']) == L))

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(layers))); ax.set_xticklabels([f'L{l}' for l in layers])
ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos)
for i in range(len(algos)):
    for j in range(len(layers)):
        ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('corr(cos_theta, f_radial) per layer')
fig.colorbar(im, ax=ax)
plt.tight_layout()
out2 = os.path.join(OUT, 'corr_cos_f_radial_heatmap.png')
plt.savefig(out2, dpi=120)
plt.close()
print(f'saved → {out2}')

# Plot 3: mean_f_radial per layer
fig, ax = plt.subplots(figsize=(10, 5))
for i, L in enumerate(layers):
    vals = [float(next(r['mean_f_radial'] for r in rows if r['algo'] == a and int(r['layer']) == L))
            for a in algos]
    ax.plot(algos, vals, marker='o', label=f'L{L}')
ax.set_ylabel('mean f_radial')
ax.set_title('mean f_radial per algo per layer')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=20)
plt.tight_layout()
out3 = os.path.join(OUT, 'mean_f_radial_lines.png')
plt.savefig(out3, dpi=120)
plt.close()
print(f'saved → {out3}')

# Plot 4: mean_cos per layer
fig, ax = plt.subplots(figsize=(10, 5))
for i, L in enumerate(layers):
    vals = [float(next(r['mean_cos'] for r in rows if r['algo'] == a and int(r['layer']) == L))
            for a in algos]
    ax.plot(algos, vals, marker='o', label=f'L{L}')
ax.set_ylabel('mean cos_theta')
ax.set_title('mean cos_theta per algo per layer')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=20)
plt.tight_layout()
out4 = os.path.join(OUT, 'mean_cos_lines.png')
plt.savefig(out4, dpi=120)
plt.close()
print(f'saved → {out4}')
