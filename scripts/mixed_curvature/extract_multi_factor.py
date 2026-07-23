"""Phase 2 P0.1: 多因素提取 pipeline.

从 Toys items TFRecord 解析 (Title, Brand, Categories, Price) → 4 因子表示:
- F_text (768-dim):   TFRecord 现成 embedding (Stage 0 fused text representation)
- F_brand (32-dim):   Brand id → Embedding (learned at RQ-VAE input)
- F_taxonomy (32-dim): Categories path → Poincaré 层级位置 (固定 depth-based)
- F_behavior (32-dim): 用户训练序列 Co-occurrence SVD

输出: per-factor tensor (12288, dim) + meta.json
"""
import json
import os
import re
import sys
import glob
import argparse
import numpy as np
import torch
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ITEM_PATTERN = re.compile(
    r"Title:\s*(?P<title>[^;]+?);\s*"
    r"Brand:\s*(?P<brand>[^;]+?);\s*"
    r"Categories:\s*(?P<cats>\[[^\]]*?\]);\s*"
    r"Price:\s*(?P<price>[0-9.nan]+)\s*;",
    re.IGNORECASE
)

CAT_SPLIT = re.compile(r"'\s*([^']+?)\s*'")

# HTML entity unmask
HTML_ENTITIES = {
    '&amp;': '&',
    '&reg;': '®',
    '&copy;': '©',
    '&quot;': '"',
    '&lt;': '<',
    '&gt;': '>',
    '&trade;': '™',
}


def unmask(s: str) -> str:
    for k, v in HTML_ENTITIES.items():
        s = s.replace(k, v)
    return s


def parse_item_text(text: str):
    """解析 item text → (title, brand, categories[3], price)"""
    text = unmask(text)
    m = ITEM_PATTERN.search(text)
    if not m:
        raise ValueError(f"Cannot parse: {text[:200]}")
    cats_raw = m.group('cats')
    cats = CAT_SPLIT.findall(cats_raw)
    # 取最多 3 层
    cats = cats[:3]
    while len(cats) < 3:
        cats.append('<PAD>')
    brand = m.group('brand').strip()
    if brand.lower() == 'unknown':
        brand = '<UNKNOWN>'
    price_raw = m.group('price').strip().lower()
    if price_raw in ('nan', ''):
        price = float('nan')
    else:
        try:
            price = float(price_raw)
        except ValueError:
            price = float('nan')
    return m.group('title').strip(), brand, cats, price


def load_items(data_dir: str):
    """遍历 items TFRecord, 返回 list[(id, title, brand, cats, price, embedding)]"""
    files = sorted(glob.glob(os.path.join(data_dir, 'items', '*.tfrecord.gz')))
    if not files:
        raise FileNotFoundError(f"No .tfrecord.gz under {data_dir}/items/")
    print(f"[items] {len(files)} files, parsing...", flush=True)
    out = []
    for fi, f in enumerate(files):
        ds = tf.data.TFRecordDataset([f], compression_type='GZIP')
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            text = ex.features.feature['text'].bytes_list.value[0].decode('utf-8')
            item_id = int(ex.features.feature['id'].int64_list.value[0])
            emb = np.array(ex.features.feature['embedding'].float_list.value, dtype=np.float32)
            try:
                title, brand, cats, price = parse_item_text(text)
            except ValueError:
                continue  # 跳过无法解析的
            out.append((item_id, title, brand, cats, price, emb))
        if (fi + 1) % 5 == 0:
            print(f"  {fi+1}/{len(files)} files, {len(out)} items so far", flush=True)
    # 按 item_id 排序
    out.sort(key=lambda x: x[0])
    return out


def build_brand_table(items):
    """Brand → id 映射 (保留频次顺序, 0 = PAD/UNKNOWN)"""
    from collections import Counter
    cnt = Counter(b for _, _, b, _, _, _ in items)
    sorted_brands = [b for b, _ in cnt.most_common()]
    b2i = {'<PAD>': 0}
    for b in sorted_brands:
        if b == '<UNKNOWN>':
            continue
        b2i[b] = len(b2i)
    b2i['<UNKNOWN>'] = 0  # 未知/缺失都映射到 0
    print(f"[brand] unique brands: {len(b2i)-1} (excl PAD)")
    return b2i


def build_taxonomy_table(items):
    """构造 taxonomy 层级: Path → depth (0/1/2).
    用 level-based embedding: 每个 category 的 Poincaré 球坐标由 depth 决定。
    返回: cat_path → tuple id (per-level position)
    """
    from collections import defaultdict
    # 三层: level0 / level1 / level2
    lvl0 = {}
    lvl1 = defaultdict(dict)  # lvl0 -> {lvl1_name: idx}
    lvl2 = defaultdict(lambda: defaultdict(dict))  # (l0, l1) -> {l2_name: idx}

    for _, _, _, cats, _, _ in items:
        c0, c1, c2 = cats
        if c0 not in lvl0:
            lvl0[c0] = len(lvl0)
        if c0 not in lvl1:
            lvl1[c0] = {}
        if c1 not in lvl1[c0]:
            lvl1[c0][c1] = len(lvl1[c0])
        key = (c0, c1)
        if key not in lvl2:
            lvl2[key] = {}
        if c2 not in lvl2[key]:
            lvl2[key][c2] = len(lvl2[key])
    print(f"[taxonomy] L0={len(lvl0)} L1_max={max(len(d) for d in lvl1.values())} L2_max={max(len(d) for d in lvl2.values())}")
    return lvl0, dict(lvl1), {k: dict(v) for k, v in lvl2.items()}


def build_taxonomy_polar_embedding(cats, lvl0, lvl1, lvl2, dim=32):
    """把 3 层 categories 编码为 Poincaré ball 坐标 (3 × dim).
    设计: 顶层在球心附近 (radius ~ 0.1), 中层半径 ~ 0.4, 底层半径 ~ 0.7。
    角度由 lvl 索引生成。
    返回 numpy (3*dim,)
    """
    out = np.zeros(3 * dim, dtype=np.float32)
    c0, c1, c2 = cats
    # 半径按 depth 分配
    radii = [0.1, 0.4, 0.7]
    levels = [
        (lvl0.get(c0, 0), len(lvl0)),
        (lvl1.get(c0, {}).get(c1, 0), max(1, len(lvl1.get(c0, {})))),
        (lvl2.get((c0, c1), {}).get(c2, 0), max(1, len(lvl2.get((c0, c1), {})))),
    ]
    for li, (idx, n_total) in enumerate(levels):
        r = radii[li]
        # 角度由 idx 均匀分
        if n_total == 0:
            n_total = 1
        # 用 dim-1 维单位球面点 (skip one dim for radius)
        angles = np.linspace(0, 2 * np.pi, n_total, endpoint=False)
        a = angles[idx % n_total]
        # 在 dim 维里填充, 用前 dim-1 维放 cos/sin 调制, 第 dim 维为 r
        start = li * dim
        out[start] = r * np.cos(a)
        out[start + 1] = r * np.sin(a)
        # 剩余维度填 0 + 小噪声 (避免完全共线)
        if dim > 2:
            for k in range(2, dim):
                out[start + k] = r * np.cos(a * (k + 1)) * 0.1
    return out


def load_user_sequences(data_dir: str, split: str = 'training'):
    """加载用户训练序列, 返回 list[list[int]]
    Toys training TFRecord schema: user_id, text (n=5), sequence_data (n=5 int), embedding (n=5*768)
    每行对应一个 user, sequence_data 是该 user 的 5-item 序列(可能因 sliding window 多行)。
    """
    files = sorted(glob.glob(os.path.join(data_dir, split, '*.tfrecord.gz')))
    if not files:
        raise FileNotFoundError(f"No tfrecord.gz under {data_dir}/{split}/")
    print(f"[users:{split}] {len(files)} files, parsing...", flush=True)
    seqs = []
    for fi, f in enumerate(files):
        ds = tf.data.TFRecordDataset([f], compression_type='GZIP')
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            # Toys 字段名: sequence_data (每行 5 个 item id)
            for field in ['sequence_data', 'sequence', 'item_ids', 'items']:
                if field in ex.features.feature:
                    feat = ex.features.feature[field]
                    if feat.WhichOneof('kind') == 'int64_list' and len(feat.int64_list.value):
                        seqs.append(list(feat.int64_list.value))
                        break
    print(f"[users:{split}] {len(seqs)} sequences loaded")
    return seqs


def build_behavior_svd(seqs, n_items, dim=32, window=1):
    """构造 item co-occurrence matrix → SVD → (n_items, dim)"""
    print(f"[behavior] building co-occurrence (window={window}) from {len(seqs)} sequences...")
    coo = np.zeros((n_items, n_items), dtype=np.float32)
    for seq in seqs:
        for i in range(len(seq)):
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                a, b = seq[i], seq[j]
                if 0 <= a < n_items and 0 <= b < n_items:
                    coo[a, b] += 1
    # 对称化
    coo = coo + coo.T
    # 行归一化 (PMI 替代: log(1 + coo) 后减去 log 行和)
    coo_log = np.log1p(coo)
    row_sum = coo_log.sum(axis=1, keepdims=True) + 1e-12
    coo_norm = coo_log / row_sum
    print(f"[behavior] SVD on {n_items}x{n_items}...")
    U, S, Vt = np.linalg.svd(coo_norm, full_matrices=False)
    F_beh = U[:, :dim].astype(np.float32) * np.sqrt(S[:dim]).astype(np.float32)
    print(f"[behavior] F_beh shape: {F_beh.shape}, sample norm: {np.linalg.norm(F_beh[0]):.3f}")
    return F_beh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys')
    ap.add_argument('--out_dir', default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature')
    ap.add_argument('--brand_dim', type=int, default=32)
    ap.add_argument('--taxonomy_dim', type=int, default=32)
    ap.add_argument('--behavior_dim', type=int, default=32)
    ap.add_argument('--co_window', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) items
    items = load_items(args.data_dir)
    n_items = len(items)
    print(f"[items] {n_items} parsed")

    # 2) Brand table
    b2i = build_brand_table(items)
    brand_ids = np.zeros(n_items, dtype=np.int64)
    for k, (_, _, b, _, _, _) in enumerate(items):
        brand_ids[k] = b2i.get(b, 0)

    # 3) Taxonomy Poincaré embedding
    lvl0, lvl1, lvl2 = build_taxonomy_table(items)
    F_tax = np.stack([
        build_taxonomy_polar_embedding(cats, lvl0, lvl1, lvl2, dim=args.taxonomy_dim)
        for _, _, _, cats, _, _ in items
    ], axis=0)
    print(f"[taxonomy] F_tax shape: {F_tax.shape}")

    # 4) F_text: 直接复用 TFRecord.embedding (768-dim fused)
    F_text = np.stack([emb for _, _, _, _, _, emb in items], axis=0)
    print(f"[text] F_text shape: {F_text.shape}")

    # 5) Behavior SVD
    seqs = load_user_sequences(args.data_dir, split='training')
    F_beh = build_behavior_svd(seqs, n_items=n_items, dim=args.behavior_dim, window=args.co_window)

    # 6) F_brand: 简单 random init (之后在 RQ-VAE 中可 fine-tune 或作为静态输入)
    n_brands = len(b2i)
    rng = np.random.default_rng(42)
    brand_emb_table = rng.normal(0, 0.1, size=(n_brands, args.brand_dim)).astype(np.float32)
    F_brand = brand_emb_table[brand_ids]
    print(f"[brand] F_brand shape: {F_brand.shape} (from {n_brands} brands)")

    # 7) Save
    item_ids_arr = np.array([it[0] for it in items], dtype=np.int64)
    out = {
        'F_text': torch.from_numpy(F_text),       # (N, 768)
        'F_brand': torch.from_numpy(F_brand),     # (N, brand_dim)
        'F_taxonomy': torch.from_numpy(F_tax),    # (N, 3*taxonomy_dim) (Poincaré 3-level)
        'F_behavior': torch.from_numpy(F_beh),    # (N, behavior_dim)
        'item_ids': torch.from_numpy(item_ids_arr),  # (N,) int64
        'brand_ids': torch.from_numpy(brand_ids), # (N,) int64
        'n_brands': n_brands,
        'n_items': n_items,
        'brand_table': b2i,
        'lvl0_size': len(lvl0),
        'lvl1_max': max(len(d) for d in lvl1.values()) if lvl1 else 0,
        'lvl2_max': max(len(d) for d in lvl2.values()) if lvl2 else 0,
    }
    save_path = os.path.join(args.out_dir, 'multi_factor.pt')
    torch.save(out, save_path)
    print(f"\n[save] {save_path}")
    print(f"  F_text:    {out['F_text'].shape}")
    print(f"  F_brand:   {out['F_brand'].shape}")
    print(f"  F_taxonomy:{out['F_taxonomy'].shape} (3 levels × {args.taxonomy_dim}d)")
    print(f"  F_behavior:{out['F_behavior'].shape}")
    print(f"  total dim: {F_text.shape[1] + args.brand_dim + 3*args.taxonomy_dim + args.behavior_dim}")

    meta = {
        'n_items': n_items,
        'F_text_dim': int(F_text.shape[1]),
        'F_brand_dim': args.brand_dim,
        'F_taxonomy_dim': args.taxonomy_dim,
        'F_taxonomy_levels': 3,
        'F_behavior_dim': args.behavior_dim,
        'total_concat_dim': int(F_text.shape[1] + args.brand_dim + 3 * args.taxonomy_dim + args.behavior_dim),
        'n_brands': n_brands,
        'co_window': args.co_window,
    }
    with open(os.path.join(args.out_dir, 'multi_factor_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] {os.path.join(args.out_dir, 'multi_factor_meta.json')}")


if __name__ == '__main__':
    main()