#!/usr/bin/env python3
# task17_dissociation.py — task19 现象 B: I(Z;Y) vs H(c_l|C_<l) dissociation check
#
# 跨算法 (A/B/C) × 跨层 (L1, L2, L3)
# - X axis: I(Z;Y) bits (from task19 #3 InfoTok)
# - Y axis: H(c_l|C_<l) entropy bits (from task18 q10)
# - Dissociation: 两序列方向不同 (一个减 一个涨)
# - ReSID 镜像: 两序列同向 (都减 或 都涨) → I 是 H 的 mirror
#
# 输出: 3 算法 dissociation score table + verdict

import os, json, numpy as np

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task19'
INFO_TOK_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task19/task17_info_tok_plane.json'
PREFIX_H_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task18/task6_q10_all.json'


def main():
    print('=' * 70)
    print('task19 现象 B — I(Z;Y) (InfoTok) vs H(c_l|C_<l) (prefix entropy)')
    print('=' * 70)

    info_tok = json.load(open(INFO_TOK_PATH))
    prefix_h = json.load(open(PREFIX_H_PATH))

    ALGORITHMS = ['A_baseline', 'B_mmq', 'C_gsrq']

    results = {}
    for algo in ALGORITHMS:
        if algo not in info_tok or algo not in prefix_h:
            print(f'  {algo}: skip (missing)')
            continue
        # I(Z;Y) sequence at L1, L2, L3 (3 layer values)
        i_seq = info_tok[algo]['mi_zy_sequence']
        # prefix_h entropy at L0..L3 (4 layers; L0 is uniform at 5.549, so meaningful from L1)
        h_seq = prefix_h[algo]['entropy_per_l'][1:4]      # L1, L2, L3

        # Compute linear correlation & direction
        i_arr = np.array(i_seq)
        h_arr = np.array(h_seq)
        diff_i = i_arr - i_arr[0]                          # Δ from L1
        diff_h = h_arr - h_arr[0]                          # Δ from L1

        # Pearson
        if i_arr.std() > 1e-6 and h_arr.std() > 1e-6:
            pearson = float(np.corrcoef(i_arr, h_arr)[0, 1])
        else:
            pearson = 0.0

        # Dissociation: 在某些层 diff_i 和 diff_h 符号相反
        signs_i = np.sign(diff_i)
        signs_h = np.sign(diff_h)
        opposite_layer_count = int(((signs_i * signs_h) < 0).sum())
        total_layers = len(diff_i)

        # ReSID mirror: 全同号
        same_sign_layer_count = int(((signs_i * signs_h) > 0).sum())

        results[algo] = {
            'i_zy_bits_per_layer': i_seq,
            'h_prefix_entropy_per_layer': h_seq,
            'delta_i_from_l1': diff_i.tolist(),
            'delta_h_from_l1': diff_h.tolist(),
            'pearson_corr': pearson,
            'opposite_direction_layer_count': opposite_layer_count,
            'same_direction_layer_count': same_sign_layer_count,
            'kill_lines': {
                'dissociation_should_have_opposite_sign_count_le_0': opposite_layer_count >= 1,
                'mirror_should_have_pearson_eq_1': abs(pearson - 1.0) < 0.10,
            }
        }
        print(f'\n  {algo}:')
        print(f'    I(Z;Y): {i_seq}, Δ = {[round(v, 3) for v in diff_i.tolist()]}')
        print(f'    H(c_l|C_<l): {[round(v, 3) for v in h_seq]}, Δ = {[round(v, 3) for v in diff_h.tolist()]}')
        print(f'    Pearson: {pearson:.4f}')
        print(f'    Opposite direction layer count: {opposite_layer_count}/{total_layers}')

    # Save
    out_json = os.path.join(OUT_DIR, 'task17_dissociation.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n=== JSON → {out_json} ===')

    # MD summary
    summary = ['# task19 现象 B — dissociation', '',
               'Dissociation check: I(Z;Y) (InfoTok) vs H(c_l|C_<l) (prefix entropy) — 跨算法 × 层',
               '',
               '| Algo | I(Z;Y) L1→L3 | Δ from L1 | H(c_l\\|C_<l) L1→L3 | Δ from L1 | Pearson | Opposite layers | Verdict |',
               '|------|------------|-----------|--------------|-----------|---------|------------------|---------|']
    for algo in ALGORITHMS:
        if algo in results:
            r = results[algo]
            i_seq = r['i_zy_bits_per_layer']
            h_seq = r['h_prefix_entropy_per_layer']
            di = r['delta_i_from_l1']
            dh = r['delta_h_from_l1']
            pearson = r['pearson_corr']
            opp = r['opposite_direction_layer_count']
            if opp >= 1:
                verdict = '✅ Dissociated'
            elif abs(pearson - 1.0) < 0.10:
                verdict = '❌ Mirror'
            else:
                verdict = '○ Partial'
            summary.append(f'| {algo} | {[round(v, 3) for v in i_seq]} | {[round(v, 3) for v in di]} | {[round(v, 3) for v in h_seq]} | {[round(v, 3) for v in dh]} | {pearson:.3f} | {opp} | {verdict} |')

    summary.append('')
    summary.append('## kill lines')
    summary.append('- **dissociation**: 至少 1 层 I(Z;Y) 与 H(c_l|C_<l) 方向相反 → I 是 **真正信息**, 不只 code entropy 的镜像')
    summary.append('- **mirror**: 两序列完全同向 (Pearson ≈ 1.0) → I 是 H 的镜像, "V-info 没新东西"')
    summary.append('- **partial**: 同向主导但有例外 → 需细看')

    md_path = os.path.join(OUT_DIR, 'task17_dissociation_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(summary))
    print(f'\n=== Summary → {md_path} ===')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
