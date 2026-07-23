#!/usr/bin/env python3
"""解析 Toys items tfrecord 提取 brand / category / title → 用于 Idea B 概念方向
Output: result/ideaB_causal/toys_metadata.json
{
  item_id: {title, brand, category_top, category_full: [...]}
}
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json, glob, re, time
import tensorflow as tf

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys/items'
N_ITEMS = 11924


def parse_text(text_bytes):
    """Parse text field: 'Title: ...; Brand: ...; Categories: [...]' """
    try:
        text = text_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return None
    # Brand
    brand = 'Unknown'
    m = re.search(r'Brand:\s*([^;]+)', text)
    if m:
        brand = m.group(1).strip()
    # Categories
    cats = []
    m = re.search(r'Categories:\s*\[([^\]]*)\]', text)
    if m:
        raw = m.group(1)
        cats = [c.strip().strip("'").strip('"') for c in raw.split(',') if c.strip()]
    # Title
    title = ''
    m = re.search(r'Title:\s*([^;]+)', text)
    if m:
        title = m.group(1).strip()
    return {
        'title':  title,
        'brand':  brand,
        'cats':   cats,
        'cat_top': cats[0] if cats else 'Unknown',
        'cat_sub': cats[1] if len(cats) > 1 else 'Unknown',
    }


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.tfrecord.gz')))
    print(f'Loading {len(files)} files...')
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')

    metadata = {}
    n = 0
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        item_id = int(ex.features.feature['id'].int64_list.value[0])
        text_b = ex.features.feature['text'].bytes_list.value[0]
        meta = parse_text(text_b)
        if meta is not None:
            metadata[item_id] = meta
        n += 1
        if n % 2000 == 0:
            print(f'  {n} items parsed ({time.time()-t0:.1f}s)')
        if n >= N_ITEMS + 1000:  # padding for safety
            break

    print(f'\nTotal: {n} items, {len(metadata)} with metadata')
    print(f'time: {time.time()-t0:.1f}s')

    # Stats
    from collections import Counter
    cat_top_counter = Counter(m['cat_top'] for m in metadata.values())
    brand_counter   = Counter(m['brand']   for m in metadata.values())
    print(f'\nTop 10 categories (level 1):')
    for k, v in cat_top_counter.most_common(10):
        print(f'  {k}: {v}')
    print(f'\nTop 10 brands:')
    for k, v in brand_counter.most_common(10):
        print(f'  {k}: {v}')
    print(f'\nUnique brand count: {len(brand_counter)}')
    print(f'Unique cat_top count: {len(cat_top_counter)}')
    print(f'Items with brand != Unknown: {sum(1 for m in metadata.values() if m["brand"] != "Unknown")}')

    # Save
    out_path = os.path.join(OUT_DIR, 'toys_metadata.json')
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nSaved → {out_path}')

    # Save index by brand and category
    by_brand = {}
    by_cat_top = {}
    for iid, m in metadata.items():
        if m['brand'] not in ('Unknown', ''):
            by_brand.setdefault(m['brand'], []).append(iid)
        if m['cat_top'] not in ('Unknown', '', 'Toys & Games'):
            by_cat_top.setdefault(m['cat_top'], []).append(iid)
    print(f'\nUsable brand groups: {len(by_brand)} (≥5 items: {sum(1 for v in by_brand.values() if len(v)>=5)})')
    print(f'Usable cat_top groups: {len(by_cat_top)} (≥5 items: {sum(1 for v in by_cat_top.values() if len(v)>=5)})')

    with open(os.path.join(OUT_DIR, 'by_brand.json'), 'w') as f:
        json.dump({k: v for k, v in by_brand.items() if len(v) >= 5}, f)
    with open(os.path.join(OUT_DIR, 'by_cat_top.json'), 'w') as f:
        json.dump({k: v for k, v in by_cat_top.items() if len(v) >= 5}, f)
    print(f'Saved by_brand and by_cat_top indices.')


if __name__ == '__main__':
    main()