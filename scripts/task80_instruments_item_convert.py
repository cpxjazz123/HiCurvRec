#!/usr/bin/env python3
"""
Task #80/#81 — Convert Musical_Instruments meta → RecBole .item file (FDSA/S³Rec 需)

输入: data/amazon_data/musical_instruments_meta/meta_Musical_Instruments.jsonl.gz
     (213,593 lines, 每行有 parent_asin + categories)

输出: RecBole/dataset/Musical_Instruments/Musical_Instruments.item
     格式: item_id:token \t class:token_seq
     (FDSA 需要 class 字段做 feature embedding)

策略 (R11.3 自主决策):
- 使用 categories[-1] (最具体的 sub-category, 如 "Drum Sets" 而非 "Musical Instruments")
- categories[0] 几乎都是 'Musical Instruments' 本身, 没有区分度
- categories 列表用 " > " 拼接成 token_seq 以保留 hierarchy 信息
- 没有 categories 的 item 给 class="unknown" (R2: 这是数据缺失, 不掩盖)
"""
import gzip
import json
import os
from pathlib import Path

META_GZ = "/home/wlia0047/ar57/wenyu/data/amazon_data/musical_instruments_meta/meta_Musical_Instruments.jsonl.gz"
OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/dataset/Musical_Instruments"
OUT_ITEM = f"{OUT_DIR}/Musical_Instruments.item"

# 与 task80_instruments_inter_convert.py 一致的映射
INTER_FILE = f"{OUT_DIR}/Musical_Instruments.inter"
ITEM_MAP_FILE = f"{OUT_DIR}/item_id_map.json"


def load_item_map():
    """从 .inter 反向构造 asin → int 映射 (因 inter_convert 用了 dict 顺序)."""
    import json as _json
    if not Path(ITEM_MAP_FILE).exists():
        raise FileNotFoundError(
            f"{ITEM_MAP_FILE} 不存在. 必须先跑 task80_instruments_inter_convert.py."
        )
    with open(ITEM_MAP_FILE) as f:
        asin_to_iid = _json.load(f)
    return asin_to_iid


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading item_id_map ...")
    asin_to_iid = load_item_map()
    print(f"[INFO] {len(asin_to_iid)} items in map")

    print(f"[INFO] Reading {META_GZ} ...")
    rows = []
    missing = 0
    with gzip.open(META_GZ, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = d.get("parent_asin")
            if asin not in asin_to_iid:
                continue  # 不是 .inter 中的 item
            cats = d.get("categories", [])
            if not cats or not isinstance(cats, list):
                cls = "unknown"
                missing += 1
            else:
                # R11.3 决策: categories[-1] (最具体的 sub-category)
                cls = str(cats[-1]).replace(" ", "_").replace(",", "_")
            rows.append((asin_to_iid[asin], cls))

    print(f"[INFO] {len(rows)} items have category, {missing} items missing/empty")

    # 必须填满 .inter 中所有 item (RecBole 要求 1:1)
    seen = {r[0] for r in rows}
    for iid in range(24587):
        if iid not in seen:
            rows.append((iid, "unknown"))
    rows.sort(key=lambda x: x[0])

    print(f"[INFO] Writing {OUT_ITEM} ...")
    with open(OUT_ITEM, "w") as f:
        f.write("item_id:token\tclass:token_seq\n")
        for iid, cls in rows:
            f.write(f"{iid}\t{cls}\n")

    print(f"[INFO] File size: {os.path.getsize(OUT_ITEM) / 1024**2:.1f} MB")
    print(f"[OK] RecBole .item ready: {OUT_ITEM}")
    print()
    print("===== Sample categories distribution =====")
    from collections import Counter
    cls_counter = Counter([r[1] for r in rows])
    for cls, n in cls_counter.most_common(15):
        print(f"  {cls}: {n}")
    print(f"  (total {len(cls_counter)} unique categories)")


if __name__ == "__main__":
    main()
