#!/usr/bin/env python3
"""工具: 算 item popularity (次数) 从 tfrecord 训练序列

复用 toys 训练分区 (152 文件), 数每个 item 在 user 序列中出现的次数.
输出: np.array, shape (11924,), 是 popularity.
"""
import sys, os, glob, gzip, time
import numpy as np
import tensorflow as tf

DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys/training'
OUT_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag2_ollivier_ricci/toys_item_popularity.npy'


def read_tfrecord(path):
    """Read records via TFRecordDataset with gzip."""
    ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
    return list(ds.as_numpy_iterator())


def parse_user_sequence(record_bytes):
    """Extract the items list (sequence_data) from a user record."""
    example = tf.train.Example()
    example.ParseFromString(record_bytes)
    feats = example.features.feature
    # 寻找 sequence_data (int64 list of items)
    if 'sequence_data' in feats:
        items = feats['sequence_data'].int64_list.value
        if items:
            return list(items)
    # 其它可能的字段
    for k in feats:
        v = feats[k]
        if v.int64_list.value and len(v.int64_list.value) > 1:
            return list(v.int64_list.value)
    return []


def main():
    print('=' * 70)
    print('计算 item popularity (从 toys/training/*.tfrecord.gz)')
    print('=' * 70)
    if os.path.exists(OUT_PATH):
        pop = np.load(OUT_PATH)
        print(f'  Cached popularity found: shape {pop.shape}, '
              f'min {pop.min()}, max {pop.max()}, median {np.median(pop)}')
        return pop

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'partition_*.tfrecord.gz')))
    print(f'  Found {len(files)} training files')

    # Item popularity counter (assume max item_id < 200000)
    N_ITEMS = 200000
    pop = np.zeros(N_ITEMS, dtype=np.int64)
    t0 = time.time()
    total_seq_count = 0
    for fi, path in enumerate(files):
        records = read_tfrecord(path)
        for rec in records:
            items = parse_user_sequence(rec)
            total_seq_count += 1
            for it in items:
                if 0 <= it < N_ITEMS:
                    pop[it] += 1
        if fi % 20 == 0:
            print(f'    [{fi}/{len(files)}] '
                  f'elapsed={time.time()-t0:.1f}s, '
                  f'users={total_seq_count}, '
                  f'max_pop={pop.max()}, '
                  f'mean={pop[pop > 0].mean():.1f}')
    print(f'\n  Total users: {total_seq_count}')
    # Only keep first 11924 items (Toys catalog size)
    pop_final = pop[:11924].astype(np.int32)
    print(f'  pop[:11924] stats: '
          f'min={pop_final.min()}, max={pop_final.max()}, '
          f'mean={pop_final.mean():.2f}, median={np.median(pop_final):.0f}')
    print(f'  nonzero count: {(pop_final > 0).sum()}/{len(pop_final)}')

    np.save(OUT_PATH, pop_final)
    print(f'  Saved → {OUT_PATH}')
    return pop_final


if __name__ == '__main__':
    main()
