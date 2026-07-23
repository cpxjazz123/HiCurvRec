#!/usr/bin/env python3
"""P3a: 三路切分脚本 — 按用户切训练/验证/测试

目的:
- 训练集: 70% 用户 (约 13588) — 用来训 codebook 和 T5
- 验证集: 15% 用户 (约 2912) — 只用来做 val_R@10_Generative 监控和选 checkpoint
- 测试集: 15% 用户 (约 2912) — 全程封存, 只在最后拿出来跑一次

切分规则 (实验设计文档, 训练前写死):
  user_id hash → bucket[0, 9]
  bucket in [0..6] → train
  bucket in [7..8] → val
  bucket in [9]   → test
  (每个用户只在三路之一)

输出:
- data/amazon_data/toys/diag_train/  (按用户过滤的 TFRecord)
- data/amazon_data/toys/diag_val/
- data/amazon_data/toys/diag_test/
- 验证: 三路用户集合互不重叠 (hashed set intersection = ∅)
"""
import os
import sys
import gzip
import json
import hashlib
import glob
import numpy as np
import tensorflow as tf

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys'
OUT_BASE = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys'

# Bucket config (100-bucket, 写死, 不再改):
#   bucket [0..69]  → train   (70%)
#   bucket [70..84] → val     (15%)
#   bucket [85..99] → test    (15%)


def user_bucket(uid: int) -> int:
    """Deterministic user_id → bucket [0,99].  Use 100 buckets for precise 70/15/15 split."""
    h = hashlib.sha256(str(uid).encode()).hexdigest()
    return int(h, 16) % 100


def assign_split(uid: int) -> str:
    """70/15/15 split via 100-bucket assignment (写死, see 选点规则.md)."""
    b = user_bucket(uid)
    if b < 70:
        return 'train'    # 70% users
    if b < 85:
        return 'val'      # 15% users (bucket 70..84)
    return 'test'         # 15% users (bucket 85..99)


# Legacy 10-bucket config (kept for reference / not used)
LEGACY_TRAIN_BUCKETS = set(range(0, 7))
LEGACY_VAL_BUCKETS = set(range(7, 9))
LEGACY_TEST_BUCKETS = set(range(9, 10))


def read_tfrecord(path):
    """Read all records from a gzipped TFRecord."""
    out = []
    with tf.io.gfile.GFile(path, 'rb') as f:
        raw = f.read()
    # Decompress
    try:
        decompressed = gzip.decompress(raw)
    except Exception:
        decompressed = raw
    for record in tf.compat.v1.python_io.tf_record_iterator(
        decompressed.decode() if isinstance(decompressed, bytes) else None
    ) if False else []:
        out.append(record)
    # Use tf.data.TFRecordDataset
    ds = tf.data.TFRecordDataset(decompressed if isinstance(decompressed, bytes) else raw)
    return list(ds.as_numpy_iterator())


def parse_user_id(record_bytes):
    """Extract user_id from a TFRecord record."""
    example = tf.train.Example()
    example.ParseFromString(record_bytes)
    feats = example.features.feature
    if 'user_id' in feats:
        uid_list = feats['user_id'].int64_list.value
        if uid_list:
            return int(uid_list[0])
    # Fallback: parse all features and look
    return None


def write_tfrecord(records, path):
    """Write records to gzipped TFRecord file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tf.io.TFRecordWriter(path + '.tmp', options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for r in records:
            writer.write(r)
    # Actually the compression is built into the writer. Use plain writer:
    # Remove tmp
    if os.path.exists(path + '.tmp'):
        os.remove(path + '.tmp')


def main():
    """Process each source partition and split by user."""
    import collections

    summary = collections.Counter()
    user_set_train = set()
    user_set_val = set()
    user_set_test = set()

    source_splits = ['training', 'evaluation', 'testing']
    # For each source partition (which contains users), split users by bucket

    for split in source_splits:
        src_dir = os.path.join(DATA_DIR, split)
        files = sorted(glob.glob(os.path.join(src_dir, 'partition_*.tfrecord.gz')))
        print(f'\n=== {split}: {len(files)} files ===')

        # Split into 3 output buckets
        out_records = {'train': [], 'val': [], 'test': []}

        for fi, fpath in enumerate(files):
            # Read records
            ds = tf.data.TFRecordDataset(fpath, compression_type='GZIP')
            for record in ds.as_numpy_iterator():
                uid = parse_user_id(record)
                if uid is None:
                    continue
                target = assign_split(uid)
                out_records[target].append(record)
                if target == 'train':
                    user_set_train.add(uid)
                elif target == 'val':
                    user_set_val.add(uid)
                else:
                    user_set_test.add(uid)
            if fi % 30 == 0:
                print(f'  {fi}/{len(files)} files, '
                      f'train_users={len(user_set_train)}, val={len(user_set_val)}, test={len(user_set_test)}')

        # Write out the 3 split directories
        for target, records in out_records.items():
            out_dir = os.path.join(OUT_BASE, f'diag_{target}')
            os.makedirs(out_dir, exist_ok=True)
            # Write as one combined file for simplicity (or per-source-partition files)
            out_path = os.path.join(out_dir, f'from_{split}.tfrecord.gz')
            with tf.io.TFRecordWriter(out_path, options=tf.io.TFRecordOptions(compression_type='GZIP')) as w:
                for r in records:
                    w.write(r)
            summary[f'{split}_to_{target}'] = len(records)
            print(f'  → {target}: {len(records)} records → {out_path}')

    # Verify disjointness
    inter_tv = user_set_train & user_set_val
    inter_tt = user_set_train & user_set_test
    inter_vt = user_set_val & user_set_test
    print(f'\n=== USER SET DISJOINTNESS ===')
    print(f'  train: {len(user_set_train)} users')
    print(f'  val:   {len(user_set_val)} users')
    print(f'  test:  {len(user_set_test)} users')
    print(f'  train ∩ val:   {len(inter_tv)}  (should be 0)')
    print(f'  train ∩ test:  {len(inter_tt)}  (should be 0)')
    print(f'  val   ∩ test:  {len(inter_vt)}  (should be 0)')

    # Save user sets for later use
    out_meta = {
        'summary': dict(summary),
        'n_users': {
            'train': len(user_set_train),
            'val': len(user_set_val),
            'test': len(user_set_test),
        },
        'disjointness': {
            'train_val': len(inter_tv),
            'train_test': len(inter_tt),
            'val_test': len(inter_vt),
        },
        'bucket_config': {
            'train_buckets': '0..69 (100-bucket modulo)',
            'val_buckets': '70..84',
            'test_buckets': '85..99',
            'note': '100 buckets for precise 70/15/15 split',
        },
    }
    out_path = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/p3_split/p3_user_split_summary.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_meta, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()