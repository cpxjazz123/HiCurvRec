#!/usr/bin/env python3
"""Idea 3 方向调整: 测试 L2/L3 加 brand 监督 vs L1 加

核心问题: STACodec L1 监督看上去 L1 已饱和, 改测 L2/L3 监督

测: 对每个 layer, 看 q_l 的 per-dim MI with brand_label
- L1 high MI (2.4 bits) → 已饱和, 加监督 redundant
- L2/L3 ~0.5 bits → 是否达到 brand 信息上限?

用 150 维 (Top-150 brand 后熵 H ≈ 4.5 bits) 替代 50 brand, 更细粒度
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea3_L1_aux'
os.makedirs(OUT_DIR, exist_ok=True)

META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}


def empirical_mi_bits(joint):
    N = joint.sum()
    pxy = joint / N
    px = joint.sum(axis=1) / N
    py = joint.sum(axis=0) / N
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
    return float(mi / np.log(2))


def main():
    print('=' * 70)
    print('Idea 3 方向调整: L1 vs L2/L3 brand 监督')
    print('=' * 70)
    metadata = json.load(open(META_PATH))
    brand_arr = np.array([metadata.get(str(i), {}).get('brand', 'NO_BRAND') for i in
                          range(len(metadata))])
    TOP_K = 150
    cnt = Counter(brand_arr.tolist())
    top_brands = [b for b, _ in cnt.most_common(TOP_K) if b != 'NO_BRAND']
    brand_labels = np.array([b if b in top_brands else 'OTHER' for b in brand_arr])
    brand_subs = sorted(set(brand_labels.tolist()))
    brand_to_id = {b: i for i, b in enumerate(brand_subs)}
    brand_ids = np.array([brand_to_id[b] for b in brand_labels])
    marg = np.bincount(brand_ids)
    p = marg / marg.sum()
    H_brand = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    print(f'  H(brand, Top-{TOP_K}+OTHER) = {H_brand:.4f} bits, '
          f'unique={len(brand_subs)}')

    # === Per-algo L2/L3 brand saturation test ===
    results = {}
    for algo, p in RQIDX_PATHS.items():
        print(f'\n--- {algo} ---')
        bundle = torch.load(p, map_location='cpu', weights_only=False)
        idx_lst = bundle['idx_lst']
        q_lst = bundle['q_lst']
        algo_res = {}
        for l in range(3):
            idx = idx_lst[l].numpy()
            n_active = int(idx.max() + 1)  # 实际有效 code 数
            q_l = q_lst[l].numpy().astype(np.float32)
            # K-Means cluster q_l, 用实际 K (=256 即 AQ baseline 的 K_l)
            K_use = min(n_active, 256)
            km = MiniBatchKMeans(n_clusters=K_use, random_state=42, n_init=3,
                                 batch_size=1024).fit(q_l)
            c_q = km.labels_.astype(np.int64)
            # Joint (c_q, brand)
            joint = np.zeros((K_use, len(brand_subs)), dtype=np.int64)
            np.add.at(joint, (c_q, brand_ids), 1)
            mi = empirical_mi_bits(joint)
            nmi = float(normalized_mutual_info_score(c_q, brand_ids))
            # Random baseline
            rng = np.random.default_rng(42)
            mi_baseline = empirical_mi_bits(np.zeros((K_use, len(brand_subs))))
            # 1 label per cluster prediction
            cluster_major = joint.argmax(axis=1)
            cluster_pred_acc = float((cluster_major[brand_ids] == brand_ids).mean())  # wrong; use majority predict
            cluster_pred = []
            for c in range(K_use):
                if joint[c].sum() > 0:
                    cluster_pred.append(joint[c].argmax())
                else:
                    cluster_pred.append(-1)
            cluster_pred = np.array(cluster_pred)
            pred_acc = float((cluster_pred[c_q] == brand_ids).mean())
            top_acc = float(np.bincount(brand_ids).max() / len(brand_ids))  # majority class baseline
            print(f'  L{l+1} (K={K_use}, n_active={n_active}): '
                  f'I(q_l→brand)={mi:.4f} bits ({mi/H_brand:.1%} of H), NMI={nmi:.4f}, '
                  f'pred_acc={pred_acc:.4f} (vs top-class={top_acc:.4f})')
            algo_res[f'l{l+1}'] = {
                'K_use': K_use,
                'n_active_codes': n_active,
                'mi_q_brand_bits': mi,
                'frac_H_brand': mi / H_brand,
                'nmi': nmi,
                'pred_acc': pred_acc,
                'top_class_acc': top_acc,
            }
        # Room-to-improve (potential supervision gain)
        L1_acc = algo_res['l1']['pred_acc']
        L3_acc = algo_res['l3']['pred_acc']
        top = algo_res['l1']['top_class_acc']
        L1_room = L1_acc - top  # L1 已经超过 baseline top-class
        L3_room = L3_acc - top  # L3 还有多少提升空间
        print(f'  L1 acc-top = {L1_room:+.4f}, L3 acc-top = {L3_room:+.4f}')
        algo_res['L1_room'] = L1_room
        algo_res['L3_room'] = L3_room
        results[algo] = algo_res

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea 3 方向调整')
    print('=' * 70)
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        a = results[algo]
        print(f'\n  {algo}:')
        print(f'    L1: I={a["l1"]["mi_q_brand_bits"]:.4f} bits, '
              f'pred_acc={a["l1"]["pred_acc"]:.4f}, room(top-diff)={a["L1_room"]:+.4f}')
        print(f'    L3: I={a["l3"]["mi_q_brand_bits"]:.4f} bits, '
              f'pred_acc={a["l3"]["pred_acc"]:.4f}, room(top-diff)={a["L3_room"]:+.4f}')
    # Decision rule
    L1_room_avg = np.mean([results[a]['L1_room'] for a in results])
    L3_room_avg = np.mean([results[a]['L3_room'] for a in results])
    print(f'\n  Average top-diff: L1={L1_room_avg:+.4f}, L3={L3_room_avg:+.4f}')
    if L3_room_avg > 0.05:
        verdict = '✓ L3 仍有提升空间, 监督放 L2/L3 而非 L1'
    elif L1_room_avg > 0.05 and L3_room_avg < 0.03:
        verdict = '✓ L1 已抓绝大部分 brand 信号, 监督 L1 是 trivial '
    else:
        verdict = '△ brand 信号已被 RQ 充分吸收, 监督 ROI 低'

    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    print(f'\n  Verdict: {verdict}')
    out_path = os.path.join(OUT_DIR, 'idea3_layer_target_brand.json')
    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'H_brand': H_brand,
            'top_k': TOP_K,
            'results': results,
            'L1_room_avg': L1_room_avg,
            'L3_room_avg': L3_room_avg,
            'verdict': verdict,
        }), f, indent=2)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()