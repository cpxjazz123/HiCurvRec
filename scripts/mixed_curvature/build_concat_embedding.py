"""Phase 2 P0.2 prep: 构造 Euclidean-Concat 928-dim embedding tensor.

- 拼接 4 因子: F_text (768) + F_brand (32) + F_taxonomy (96) + F_behavior (32) = 928
- 输出 shape 必须与 toys items TFRecord 行数一致 (12288), 失败 item 用 0 vector
- 与现有 rqvae_train_flat.yaml 兼容 (embedding_path=..., embedding_dim=928)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi_factor_pt', default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/multi_factor.pt')
    ap.add_argument('--out_path', default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt')
    ap.add_argument('--total_items', type=int, default=12288)
    args = ap.parse_args()

    d = torch.load(args.multi_factor_pt, weights_only=False)
    n_parsed = d['n_items']
    n_total = args.total_items
    print(f"[info] parsed items: {n_parsed}, total items (TFRecord): {n_total}")
    if n_parsed > n_total:
        raise ValueError(f"Parsed {n_parsed} > total {n_total}, check data_dir")

    F_text = d['F_text'].numpy()       # (n_parsed, 768)
    F_brand = d['F_brand'].numpy()     # (n_parsed, 32)
    F_tax = d['F_taxonomy'].numpy()    # (n_parsed, 96)
    F_beh = d['F_behavior'].numpy()    # (n_parsed, 32)
    print(f"[shapes] text={F_text.shape} brand={F_brand.shape} tax={F_tax.shape} beh={F_beh.shape}")

    # L2 normalize each factor individually so concat weights balanced
    def l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    F_text_n = l2(F_text) * np.sqrt(768)
    F_brand_n = l2(F_brand) * np.sqrt(32)
    F_tax_n = l2(F_tax) * np.sqrt(96)
    F_beh_n = l2(F_beh) * np.sqrt(32)

    concat = np.concatenate([F_text_n, F_brand_n, F_tax_n, F_beh_n], axis=1)
    print(f"[concat] shape={concat.shape}, dim={concat.shape[1]}")

    # Pad to total_items (rows 0..n_parsed-1 已知, 剩下 n_total-n_parsed 行 zero)
    # 注意: parsed 是按 item_id 排序的, 但中间可能有 gap (解析失败的)
    # 更安全的做法: 重新生成 item_id → embedding 映射, 然后填到 full tensor
    # 但我们提取时已经 skip 失败 — 所以当前 concat 行号是 "成功 items 的顺序", 与 TFRecord 顺序不一致
    # 解决: 重新生成 (n_total, dim) tensor, 用 item_id 索引填充
    print(f"[warn] parsed items 不一定按 id 0..11923 连续, 需要重新建全表")

    # 重新读 item_id 顺序, 然后用 idx → embedding dict
    # 由于 extract_multi_factor.py 已 sort by item_id, 实际上 row k 对应第 k 个成功 item
    # 失败 item 的 id 缺失 — 暂用 0 行 + 之后用 index_map 修正
    # 简化方案: 让 extract 输出含 item_id 列表, 然后这里直接 idx
    # 现在用 zero-padding 后, 训练时这 511 个 item 的 embedding 是 0, RQ-VAE 会映射到固定 SID

    full = np.zeros((n_total, concat.shape[1]), dtype=np.float32)
    # 加载 item_id 顺序
    item_ids = d.get('item_ids')
    if item_ids is None:
        raise ValueError("multi_factor.pt missing item_ids; re-run extract_multi_factor.py with that field")
    item_ids = item_ids.numpy()
    print(f"[map] item_ids range: {item_ids.min()}..{item_ids.max()}, n={len(item_ids)}")

    # 检查是否与 TFRecord item id 一致 (应为 0..n_total-1 不间断)
    expected = set(range(n_total))
    have = set(item_ids.tolist())
    missing = expected - have
    if missing:
        print(f"[warn] {len(missing)} item ids missing from parsed set; will be zero-row")

    # 直接 index 填充
    full[item_ids] = concat
    print(f"[save] {args.out_path}, shape={full.shape}, zero rows={int((full.sum(axis=1) == 0).sum())}")

    torch.save(torch.from_numpy(full), args.out_path)

    # Meta
    meta = {
        'shape': list(full.shape),
        'total_items': n_total,
        'parsed_items': n_parsed,
        'zero_padded': len(missing),
        'factors': {
            'F_text': 768,
            'F_brand': 32,
            'F_taxonomy': 96,  # 3 levels × 32d
            'F_behavior': 32,
        },
        'concat_dim': int(full.shape[1]),
    }
    meta_path = args.out_path.replace('.pt', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] {meta_path}")


if __name__ == '__main__':
    main()