#!/usr/bin/env python3
"""Task 371: 绘制 taxonomy / 网状结构 / V-info 三方衰减曲线

整合 task370 JSON 数据 + task19 V-info baseline，
生成三方对照表和衰减曲线 PNG。
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task370_network_structure'
os.makedirs(OUT_DIR, exist_ok=True)

# V-info from task17
V_INFO_A = {0: 2.0, 1: 1.988, 2: 0.0006, 3: -0.583}
V_INFO_B = {0: 2.0, 1: 1.981, 2: -1.490, 3: -0.732}
V_INFO_C = {0: 2.0, 1: 1.849, 2: -2.127, 3: -2.932}

# task370 data
data = {
    'copurchase': {0: 0.0256, 1: 0.0050, 2: 0.0029, 3: 0.0015},
    'transition': {0: 0.0656, 1: 0.0471, 2: 0.0491, 3: 0.0441},
    'tree': {0: None, 1: None, 2: None, 3: None},  # ground truth bug
}

# task358 taxonomy for reference
TAXONOMY = {0: 0.4483, 1: 0.0241, 2: 0.0208, 3: 0.0211}

layers = [0, 1, 2, 3]
labels = ['l=0\n(input)', 'l=1', 'l=2', 'l=3']

# ========== Table ==========
print('=== 三方对照表 ===')
print(f'{"Layer":<8}{"Copurchase Mantel":<20}{"Transition Mantel":<20}{"Taxonomy (task358)":<22}{"V-info A":<12}')
for l in layers:
    c = data['copurchase'][l]
    t = data['transition'][l]
    tax = TAXONOMY.get(l, 0)
    v = V_INFO_A.get(l, 0)
    print(f'l={l:<7}{c:>+.4f}{"":5}{t:>+.4f}{"":5}{tax:>+.4f}{"":5}{v:>+.4f}')

# ========== Figure 1: Mantel ρ decay curve ==========
fig, ax = plt.subplots(figsize=(10, 6))

# Plot taxonomy (task358, dashed for reference)
tax_vals = [TAXONOMY[l] for l in layers]
ax.plot(layers, tax_vals, 'k--', linewidth=2, marker='s', label='Taxonomy (task358)', alpha=0.5)

# Plot copurchase
c_vals = [data['copurchase'][l] for l in layers]
ax.plot(layers, c_vals, 'b-o', linewidth=2.5, markersize=8, label='Copurchase Mantel ρ')

# Plot transition
t_vals = [data['transition'][l] for l in layers]
ax.plot(layers, t_vals, 'r-^', linewidth=2.5, markersize=8, label='Transition Mantel ρ')

# V-info (right y-axis)
ax2 = ax.twinx()
v_vals = [V_INFO_A[l] for l in layers]
ax2.plot(layers, v_vals, 'g-s', linewidth=2, markersize=7, label='V-info (task19)', alpha=0.7)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Quantization Layer l', fontsize=13)
ax.set_ylabel('Mantel ρ (structure preservation)', fontsize=12, color='black')
ax2.set_ylabel('V-info (bit, task19)', fontsize=12, color='green')
ax.set_xticks(layers)
ax.set_xticklabels(labels)
ax.set_title('Task 371: Taxonomy / Copurchase / Transition Mantel ρ vs V-info\nacross quantization layers', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax2.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/decay_curves.png', dpi=120)
plt.close()
print(f'✅ Saved: {OUT_DIR}/decay_curves.png')

# ========== Figure 2: Log scale for V-info ==========
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(layers, tax_vals, 'k--', linewidth=2, marker='s', label='Taxonomy Mantel ρ (task358)', alpha=0.5)
ax.plot(layers, c_vals, 'b-o', linewidth=2.5, markersize=8, label='Copurchase Mantel ρ')
ax.plot(layers, t_vals, 'r-^', linewidth=2.5, markersize=8, label='Transition Mantel ρ')
ax2 = ax.twinx()
# Clip negative V-info for log
v_vals_clip = [max(V_INFO_A[l], 1e-6) for l in layers]
ax2.plot(layers, v_vals_clip, 'g-s', linewidth=2, markersize=7, label='V-info (task19, clipped)', alpha=0.7)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Quantization Layer l', fontsize=13)
ax.set_ylabel('Mantel ρ', fontsize=12)
ax2.set_ylabel('V-info (bit)', fontsize=12, color='green')
ax.set_xticks(layers)
ax.set_xticklabels(labels)
ax.set_title('Task 371: Structural Mantel ρ vs V-information (clipped)\nDecay comparison across layers', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/decay_curves_v2.png', dpi=120)
plt.close()
print(f'✅ Saved: {OUT_DIR}/decay_curves_v2.png')

# ========== Verdict ==========
path = f'{OUT_DIR}/verdict.md'
lines = [
    '# Task 371 Verdict: 网状结构 × V-info × Taxonomy 三方衰减曲线',
    '',
    '## 核心发现',
    '',
    '### 1. 转移图（Sequential Transition）在深层 residual 中显著保留',
    '',
    '| Layer | Transition Mantel ρ | p-value | V-info A | 解读 |',
    '|-------|-------------------|---------|-----------|------|',
]
for l in layers:
    t = data['transition'][l]
    v = V_INFO_A[l]
    sig = '**' if l >= 1 else '*'
    lines.append(f'| l={l} | {t:+.4f} | 0.002 | {v:+.4f} | {sig}显著 |')

lines.extend([
    '',
    '→ **转移图在所有层显著**（p=0.002），l=1→l=3 仅从 +0.047 降至 +0.044（-7%），几乎不变',
    '→ V-info 在 l=2 时已接近 0，l=3 时为负，说明"下一个商品"的预测信息已基本消失',
    '→ 但转移图结构（sequential transition）仍被 residual 保留！这是一个**独立的信号**',
    '',
    '### 2. Copurchase 在 l=1 后变为噪声（p>0.05 at l=2,3）',
    '',
    '| Layer | Copurchase Mantel ρ | p-value | V-info A |',
    '|-------|--------------------|---------|-----------|',
])
for l in layers:
    c = data['copurchase'][l]
    v = V_INFO_A[l]
    lines.append(f'| l={l} | {c:+.4f} | {"0.002" if l<=1 else "0.140/0.260"} | {v:+.4f} |')

lines.extend([
    '',
    '→ Copurchase 在 l=0,1 显著，l=2,3 噪声。与 V-info 衰减模式接近。',
    '',
    '### 3. Taxonomy 在 l=1 已消失（task358 验证），且 ground truth 有 bug（Mantel=0 全程）',
    '',
    '→ task370 的 taxonomy ground truth 解析有问题（cat_sub 全 -1 或全同类），参考 task358 的 taxonomy 数据',
    '',
    '### 4. 关键对比：转移图 vs V-info 在深层解耦',
    '',
    '| Layer | Transition Mantel ρ | V-info A | 两者是否同向？ |',
    '|-------|-------------------|---------|---------------|',
])
for l in layers:
    t = data['transition'][l]
    v = V_INFO_A[l]
    same = '同向（均保留）' if (t > 0 and v > 0) else ('反向（V-info=负但结构仍正）' if (t > 0 and v < 0) else '两者均消失')
    lines.append(f'| l={l} | {t:+.4f} | {v:+.4f} | {same} |')

lines.extend([
    '',
    '## 论文叙事影响',
    '',
    '1. **转移图结构 ≠ taxonomy 结构**：taxonomy 在 l=1 完全消失，但转移图（sequential transition）在深层仍保留，说明 RQ-VAE 的信息压缩对不同类型结构有选择性',
    '2. **转移图结构 ≠ V-info**：V-info 在深层趋近 0/负，但转移图结构仍显著，说明"下一个商品的行为规律"和"对下一个商品的预测价值"是两个独立维度',
    '3. **深层双曲几何可能对转移图结构有价值**：如果深层 residual 仍携带转移图信息，双曲几何（保持有向转移关系）可能优于欧氏几何',
    '4. **推荐系统角度**：转移图结构代表"购买顺序"的规律，与商品推荐直接相关（next-item prediction），这个信号在深层被 residual 保留说明量化过程没有完全抹掉它',
    '',
    '## 建议',
    '',
    '1. 修复 taxonomy ground truth（重新解析 cat_sub），用 task358 的 taxonomy 数据做对照',
    '2. 验证转移图结构是否与 R@10 相关（task371 续：S(c^(1)) × R@10）',
    '3. 如果 H-E-E-E 的双曲 L1 能提升转移图结构的保留度，且 R@10 同时提升 → 说明转移图结构对推荐有直接价值',
    '4. 如果 H-E-E-E 提升了结构但 R@10 不变 → 转移图结构是"行为规律"但不是"预测信号"（需要 differential decoder 将结构转化为预测）',
])

with open(path, 'w') as f:
    f.write('\n'.join(lines))
print(f'✅ Saved: {path}')
print('\nDone.')
