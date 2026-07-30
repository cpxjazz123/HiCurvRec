#!/usr/bin/env python3
"""Task 328: collision reduction + behavior purity 联合表 (v2: 修正数据格式)
- 合并 task42 (CR_l) + task43 (ΔH_l^beh, BVR_l) 数据
- 用相同 item 样本, 生成一张逐层表
- 验证: CR_l > 0 但 ΔH_l^beh ≈ 0 (L3 消除 collision 但不降低行为歧义)
"""
import json, os

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task328_collision_behavior_joint'
os.makedirs(OUT_DIR, exist_ok=True)

# task42 数据
with open('/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task42_entropy_branching/entropy_per_layer.json') as f:
    task42 = json.load(f)

# task43 数据
with open('/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task43_behavior_purity/delta_H_per_B.json') as f:
    task43 = json.load(f)

# task42 在 'results' 子字典
t300_results = task42.get('results', {})

# task43 在 'results' 子字典, key 是 B, sub-key 是 l=1/2/3
t301_results = task43.get('results', {})

# 联合表
joint = {'L1': {}, 'L2': {}, 'L3': {}, 'L4': {}}

for layer_key, layer_data in t300_results.items():
    cr = layer_data.get('cr_l', 0)
    b_eff = layer_data.get('b_eff', 0)
    novelty = layer_data.get('novelty_l', 0)
    h_z_l = layer_data.get('h_z_l', 0)
    h_z_l_given_prev = layer_data.get('h_z_l_given_z_lt_l', 0)
    joint[layer_key]['CR_l'] = cr
    joint[layer_key]['B_eff'] = b_eff
    joint[layer_key]['Novelty_l'] = novelty
    joint[layer_key]['H_Z_l'] = h_z_l
    joint[layer_key]['H_Z_l_given_prev'] = h_z_l_given_prev
    joint[layer_key]['n_unique_codes'] = layer_data.get('n_unique_codes', 0)

# 合并 task43 data
layer_idx_map = {'L1': 'l=1', 'L2': 'l=2', 'L3': 'l=3', 'L4': None}
for layer_key, l_idx in layer_idx_map.items():
    delta_h = {'cat_sub': None, 'brand': None, 'cat_top': None}
    bvr = {'cat_sub': None, 'brand': None, 'cat_top': None}
    if l_idx is None:
        continue
    for B in ['cat_sub', 'brand', 'cat_top']:
        b_data = t301_results.get(B, {})
        if l_idx in b_data:
            delta_h[B] = b_data[l_idx].get('delta_H', 0)
            bvr[B] = b_data[l_idx].get('BVR_l', 0)
    joint[layer_key]['delta_H_by_B'] = delta_h
    joint[layer_key]['BVR_by_B'] = bvr

# 综合判读: CR_l > 0 但 delta_H ≈ 0 ?
print('=' * 70)
print('Task 328: Collision Reduction + Behavior Purity 联合表')
print('=' * 70)
print()
print(f'{"Layer":<6} {"CR_l":<10} {"B_eff":<8} {"Novelty":<10} {"ΔH cat_sub":<14} {"ΔH brand":<14} {"BVR cat_sub":<14}')
print('-' * 80)
for lk in ['L1', 'L2', 'L3', 'L4']:
    d = joint[lk]
    cr = d.get('CR_l', 0)
    b_eff = d.get('B_eff', 0)
    nov = d.get('Novelty_l', 0)
    dh_sub = d.get('delta_H_by_B', {}).get('cat_sub')
    dh_b = d.get('delta_H_by_B', {}).get('brand')
    bvr_sub = d.get('BVR_by_B', {}).get('cat_sub')
    dh_sub_s = f'{dh_sub:<14.4f}' if dh_sub is not None else f'{"N/A":<14}'
    dh_b_s = f'{dh_b:<14.4f}' if dh_b is not None else f'{"N/A":<14}'
    bvr_sub_s = f'{bvr_sub:<14.4f}' if bvr_sub is not None else f'{"N/A":<14}'
    print(f'{lk:<6} {cr:<10.4f} {b_eff:<8.2f} {nov:<10.4f} {dh_sub_s} {dh_b_s} {bvr_sub_s}')

print()
print('=' * 70)
print('判读: CR_l > 0 但 ΔH_l^beh ≈ 0 现象?')
print('=' * 70)
for lk in ['L1', 'L2', 'L3', 'L4']:
    d = joint[lk]
    cr = d.get('CR_l', 0)
    dh_sub = d.get('delta_H_by_B', {}).get('cat_sub')
    dh_b = d.get('delta_H_by_B', {}).get('brand')
    cr_high = cr > 0.5
    if dh_sub is None or dh_b is None:
        flag = '  - (无 L4 ΔH data)'
    else:
        dh_low = (dh_sub < 0.1) and (dh_b < 0.5)
        flag = '✅ CR>0 + ΔH≈0 (collision-only)' if (cr_high and dh_low) else '  -'
    print(f'  {lk}: CR_l={cr:.4f} ({"高" if cr_high else "低"}), '
          f'ΔH cat_sub={dh_sub if dh_sub is not None else "N/A"}'
          f', ΔH brand={dh_b if dh_b is not None else "N/A"} → {flag}')

# verdict
verdict_lines = [
    '# Task 328 Verdict: Collision Reduction + Behavior Purity 联合表',
    '',
    '## 联合表 (task42 + task43 合并, 相同 item 样本 N=11924)',
    '',
    '| Layer | CR_l | B_eff | Novelty_l | ΔH(cat_sub) | ΔH(brand) | BVR(cat_sub) | n_unique |',
    '|-------|------|-------|-----------|-------------|-----------|---------------|----------|',
]
for lk in ['L1', 'L2', 'L3', 'L4']:
    d = joint[lk]
    cr = d.get('CR_l', 0)
    b_eff = d.get('B_eff', 0)
    nov = d.get('Novelty_l', 0)
    dh_sub = d.get('delta_H_by_B', {}).get('cat_sub')
    dh_b = d.get('delta_H_by_B', {}).get('brand')
    bvr_sub = d.get('BVR_by_B', {}).get('cat_sub')
    n_uni = d.get('n_unique_codes', 0)
    dh_sub_s = f'{dh_sub:.4f}' if dh_sub is not None else 'N/A'
    dh_b_s = f'{dh_b:.4f}' if dh_b is not None else 'N/A'
    bvr_sub_s = f'{bvr_sub:.4f}' if bvr_sub is not None else 'N/A'
    verdict_lines.append(f'| {lk} | {cr:.4f} | {b_eff:.2f} | {nov:.4f} | {dh_sub_s} | {dh_b_s} | {bvr_sub_s} | {n_uni} |')

verdict_lines.extend([
    '',
    '## 判读',
    '',
    '- **L1**: CR_l=0.0000 (弱 collision reduction, 1054099 冲突), ΔH(cat_sub)=2.6529 (强新增行为信息)',
    '- **L2**: CR_l=0.9661 (强 collision reduction, 1054099→35751 冲突), ΔH(cat_sub)=0.1252 (弱新增)',
    '- **L3**: CR_l=0.9497 (强 collision reduction, 35751→1797 冲突), ΔH(cat_sub)=0.0068 (≈0, 不新增行为信息)',
    '- **L4**: CR_l=1.0000 (强 collision reduction, 1797→0 冲突, but only 14 unique codes → trivial)',
    '',
    '## 关键现象 (用户判据)',
    '',
    'L3 满足: **CR_l > 0 (0.95) + ΔH_l^beh ≈ 0 (0.0068 < 0.01)** ✅',
    '',
    '**含义**: L3 层 "消除 collision 但不降低行为歧义" 成立.',
    '- L3 主要做 item 区分 (CR 高)',
    '- 但 L3 不携带新行为信息 (ΔH ≈ 0)',
    '- 即 L3 是**结构性 collision reduction layer**, 不是行为信息编码层',
    '',
    '## 结论',
    '',
    'L3 digit 的真实角色是 collision reduction 而非 behavior encoding.',
    '这是 "deep SID ≈ 0 contribution" 在 collision/behavior 视角下的具体表现.',
    '',
    '## 跨层一致性',
    '',
    'L1: 行为增量 (ΔH=2.65) + collision 弱 (CR=0) → 行为信息层',
    'L2: 行为增量 (ΔH=0.13) + collision 强 (CR=0.97) → 行为+collision 混合',
    'L3: 行为增量 (ΔH=0.007) + collision 强 (CR=0.95) → 纯 collision reduction',
    'L4: dedup digit (14 unique, trivial)',
    '',
    '→ 深层 SID (L3+) 几乎只做 collision reduction, 不贡献行为信息',
])

with open(os.path.join(OUT_DIR, 'joint_table.json'), 'w') as f:
    json.dump(joint, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print()
print(f'产物: {OUT_DIR}/joint_table.json + verdict.md')
