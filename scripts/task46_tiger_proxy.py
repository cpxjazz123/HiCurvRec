#!/usr/bin/env python3
"""Task 304 真版 proxy: 用 task21 inference tensor (2026-07-07) 验证 L3 digit 贡献.

策略:
1. 加载 task21 inference tensor (19412, 10, 4) - 真实 TIGER 生成的 top-10 SID
2. 加载 SID table (4, 11924) - 所有 item 的 4-digit SID
3. 对每个预测的 SID, 检查它在 SID table 中是否存在 → 命中真实 item
4. 对每个预测的 SID, 强制把 L3=0, 看是否还能匹配真实 item → L3 贡献诊断
5. ABCD buckets based on:
   - full_hit: 原始 SID ∈ SID table
   - ablate_L3_hit: 把 SID 的 L3 digit 设为 0 后 ∈ SID table (且原始 ≠ 改后)

更精确: 计算 L3 digit 对 SID 唯一性的影响
"""
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task46_prob_vs_rank'
os.makedirs(OUT_DIR, exist_ok=True)

INFER_TENSOR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'
SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'

N_LAYERS = 4
K = 10


def main():
    print('=' * 70)
    print('Task 304 真版 proxy: task21 真实推断 tensor + SID mapping 分析')
    print('=' * 70)

    # Load predictions tensor (task21 ckpt 推断)
    print(f'\n[1] 加载 inference tensor {INFER_TENSOR}...')
    preds = torch.load(INFER_TENSOR, map_location='cpu', weights_only=False)
    print(f'  shape: {preds.shape}, dtype: {preds.dtype}')

    # Load SID table
    print(f'\n[2] 加载 SID table {SID_PATH}...')
    sid_table = torch.load(SID_PATH, map_location='cpu', weights_only=False)
    if sid_table.dim() == 3:
        sid_table = sid_table.squeeze(0)
    print(f'  shape: {sid_table.shape}')

    # Build lookup: (L1, L2, L3, L4) -> item_idx
    print(f'\n[3] 构建 SID 查找表 (item_id 集合)...')
    sid_np = sid_table.numpy().T  # (N, 4)
    # Tuple-based set for fast lookup
    sid_set = set(tuple(row) for row in sid_np.tolist())
    sid_lookup = {}
    for i, row in enumerate(sid_np):
        sid_lookup[tuple(row)] = i

    print(f'  total items: {len(sid_lookup)}')
    print(f'  unique SIDs: {len(sid_set)}')

    # Statistics: how many items have unique L3?
    print(f'\n[4] 统计每层 unique digit 数...')
    for l in range(N_LAYERS):
        unique_digit = len(set(sid_np[:, l].tolist()))
        print(f'  L{l+1}: {unique_digit} unique digits')

    # L3 collapse: how many items per (L1, L2, L4) combo?
    print(f'\n[5] L3 collapse analysis...')
    from collections import Counter
    l3_per_l1l2l4 = {}
    for tup in sid_np:
        key = (tup[0], tup[1], tup[3])
        l3_per_l1l2l4.setdefault(key, []).append(tup[2])

    avg_l3_per_key = np.array([len(v) for v in l3_per_l1l2l4.values()])
    print(f'  (L1,L2,L4) unique combos: {len(l3_per_l1l2l4)}')
    print(f'  avg items per combo: {avg_l3_per_key.mean():.2f}')
    print(f'  max items per combo: {avg_l3_per_key.max()}')
    print(f'  median: {np.median(avg_l3_per_key)}')
    single_l3 = (avg_l3_per_key == 1).sum()
    print(f'  combos with single L3 (L3 fully determines item): {single_l3} ({single_l3/len(l3_per_l1l2l4)*100:.1f}%)')

    # ABCD analysis: for each predicted SID, does L3 matter?
    print(f'\n[6] ABCD analysis on task21 predicted SIDs (19412 × 10 = {preds.shape[0] * preds.shape[1]} preds)...')
    preds_np = preds.numpy().astype(int)  # (19412, 10, 4)

    A, B, C, D = 0, 0, 0, 0
    n_total = 0
    n_full_hit = 0
    n_ablate_hit = 0
    rank_shift_records = []

    for u_idx in range(preds_np.shape[0]):
        user_preds = preds_np[u_idx]  # (10, 4)
        for k_idx, pred_sid in enumerate(user_preds):
            pred_tup = tuple(int(x) for x in pred_sid.tolist())
            full_hit = pred_tup in sid_lookup

            # Ablate L3: replace L3 with the most common L3 for (L1,L2,L4)
            key = (pred_tup[0], pred_tup[1], pred_tup[3])
            if key in l3_per_l1l2l4:
                # Use most common L3 for this key
                from collections import Counter
                l3_candidates = l3_per_l1l2l4[key]
                if l3_candidates:
                    # Use the modal L3
                    cnt = Counter(l3_candidates)
                    modal_l3 = cnt.most_common(1)[0][0]
                    ablate_tup = (pred_tup[0], pred_tup[1], modal_l3, pred_tup[3])
                else:
                    ablate_tup = pred_tup
            else:
                # No item matches (L1,L2,L4) at all → L3 is the ONLY differentiator
                ablate_tup = pred_tup

            ablate_hit = ablate_tup in sid_lookup

            if full_hit and ablate_hit:
                A += 1
            elif full_hit and not ablate_hit:
                B += 1
            elif not full_hit and ablate_hit:
                C += 1
            else:
                D += 1

            if full_hit:
                n_full_hit += 1
            if ablate_hit:
                n_ablate_hit += 1
            n_total += 1

        if u_idx % 5000 == 0:
            print(f'  [{u_idx:5d}] full_hit={n_full_hit}/{n_total}={n_full_hit/n_total:.4f}, '
                  f'ablate_hit={n_ablate_hit/n_total:.4f}, ABCD=({A},{B},{C},{D})')

    print(f'\n[7] 完成:')
    print(f'  total preds: {n_total}')
    print(f'  full_hit: {n_full_hit} ({n_full_hit/n_total:.4f})')
    print(f'  ablate_hit (L3 modal): {n_ablate_hit} ({n_ablate_hit/n_total:.4f})')
    print(f'  ABCD: A={A}, B={B}, C={C}, D={D}')
    pct = {k: v/n_total for k, v in zip('ABCD', [A,B,C,D])}
    print(f'  ABCD %: A={pct["A"]:.4f}, B={pct["B"]:.4f}, C={pct["C"]:.4f}, D={pct["D"]:.4f}')

    # Save results
    out = {
        'infer_tensor': INFER_TENSOR,
        'sid_path': SID_PATH,
        'n_users': int(preds.shape[0]),
        'K': K,
        'n_predictions_total': n_total,
        'full_hit_rate': n_full_hit / n_total,
        'ablate_L3_modal_hit_rate': n_ablate_hit / n_total,
        'ABCD_counts': {'A': A, 'B': B, 'C': C, 'D': D},
        'ABCD_pct': pct,
        'L3_collapse': {
            'n_combos_L1L2L4': len(l3_per_l1l2l4),
            'avg_items_per_combo': float(avg_l3_per_key.mean()),
            'single_l3_combos': int(single_l3),
            'single_l3_pct': single_l3 / len(l3_per_l1l2l4),
        },
        'note': 'task21 inference tensor (real TIGER) + SID mapping. Ablate = use modal L3 for (L1,L2,L4).',
    }
    with open(os.path.join(OUT_DIR, 'abcd_real_tiger.json'), 'w') as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(OUT_DIR, 'verdict_real_tiger.md'), 'w') as f:
        f.write('# Task 304 Verdict (真 TIGER proxy)\n\n')
        f.write(f'数据集: Toys, K=10, n_users={preds.shape[0]}, 模型: task19_aq_s3/best_tiger_step3900.ckpt\n')
        f.write(f'预测来源: {INFER_TENSOR} (task21 ckpt 真实推断)\n\n')
        f.write('## L3 collapse 结构分析\n\n')
        f.write(f'- (L1,L2,L4) unique combos: **{len(l3_per_l1l2l4)}** (远少于 N={len(sid_lookup)})\n')
        f.write(f'- avg items per combo: **{avg_l3_per_key.mean():.2f}**\n')
        f.write(f'- single L3 combos (L3 完全决定 item): **{single_l3} ({single_l3/len(l3_per_l1l2l4)*100:.1f}%)**\n\n')
        f.write('## ABCD buckets (L3 digit 真实贡献)\n\n')
        f.write('| Bucket | Count | % | 含义 |\n')
        f.write('|--------|-------|---|------|\n')
        f.write(f'| A | {A} | {pct["A"]:.4f} | full ∩ ablate 命中 → L3 不影响 |\n')
        f.write(f'| **B** | **{B}** | **{pct["B"]:.4f}** | **full ∩ ¬ablate → L3 是必要 digit** |\n')
        f.write(f'| **C** | **{C}** | **{pct["C"]:.4f}** | **¬full ∩ ablate → L3 是噪声/不一致** |\n')
        f.write(f'| D | {D} | {pct["D"]:.4f} | 都未命中 |\n\n')
        f.write(f'## 命中率\n\n')
        f.write(f'- 原始 SID 命中率: **{n_full_hit/n_total:.4f}** ({n_full_hit}/{n_total})\n')
        f.write(f'- Ablate-L3 (modal) 命中率: {n_ablate_hit/n_total:.4f}\n')
        f.write(f'- 差异: {n_full_hit/n_total - n_ablate_hit/n_total:+.4f}\n\n')
        f.write('## 判读\n\n')
        b_pct, c_pct = pct['B'], pct['C']
        if b_pct > c_pct:
            f.write(f'✅ L3 digit 提供**正向贡献** (B={b_pct:.4f} > C={c_pct:.4f})\n')
        elif c_pct > b_pct:
            f.write(f'⚠️ L3 digit 主要作为**不一致/噪声** (C={c_pct:.4f} > B={b_pct:.4f})\n')
        f.write(f'\n→ L3 真正决定命中的样本 (B桶) = {b_pct*100:.2f}%\n')
        f.write(f'→ L3 噪声/不一致样本 (C桶) = {c_pct*100:.2f}%\n')
        f.write(f'\n**结论**: {("L3 digit 提供正向贡献" if b_pct > c_pct else "L3 digit 是结构性 noise (C>B)")}\n')

    print(f'\n[产物] {OUT_DIR}/abcd_real_tiger.json')
    print(f'[产物] {OUT_DIR}/verdict_real_tiger.md')


if __name__ == '__main__':
    main()