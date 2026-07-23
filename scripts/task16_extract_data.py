"""
Task 62 — 从 Toys TFRecords 提取 item embeddings (768-dim) 和元数据。
输出:
  - products/task16/from_task62/item_embeddings.pt   : Tensor (11924, 768)
  - products/task16/from_task62/item_metadata.pt     : dict with brand/category/price per item
"""

import os, re, json
import tensorflow as tf
import torch

ITEMS_DIR = "data/amazon_data/toys/items"
OUT_DIR = "products/task16/from_task62"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(os.listdir(ITEMS_DIR))
print(f"Found {len(files)} TFRecord files")

all_embeddings = []
all_metadata = []

for fname in files:
    path = os.path.join(ITEMS_DIR, fname)
    ds = tf.data.TFRecordDataset(path, compression_type="GZIP")
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        feat = ex.features.feature

        # Embedding (768-dim)
        emb = list(feat["embedding"].float_list.value)
        all_embeddings.append(emb)

        # Metadata from text field
        text = feat["text"].bytes_list.value[0].decode("utf-8")
        meta = {"title": "", "brand": "", "categories": [], "price": None}

        m = re.search(r"Title:\s*(.*?);\s*Brand:", text)
        if m:
            meta["title"] = m.group(1).strip()

        m = re.search(r"Brand:\s*([^;]+)", text)
        if m:
            meta["brand"] = m.group(1).strip()

        m = re.search(r"Categories:\s*\[(.*?)\]", text)
        if m:
            cats = []
            for c in m.group(1).split(","):
                c = c.strip().strip("'").strip()
                if c:
                    cats.append(c)
            meta["categories"] = cats

        m = re.search(r"Price:\s*([0-9.]+)", text)
        if m:
            meta["price"] = float(m.group(1))

        all_metadata.append(meta)

        if len(all_embeddings) % 2000 == 0:
            print(f"  ... {len(all_embeddings)} items processed")

# Save embeddings
emb_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
torch.save(emb_tensor, os.path.join(OUT_DIR, "item_embeddings.pt"))
print(f"\nEmbeddings saved: {emb_tensor.shape}")

# Save metadata
torch.save(all_metadata, os.path.join(OUT_DIR, "item_metadata.pt"))

# Also save a human-readable summary
brands = set(m["brand"] for m in all_metadata)
cats = set()
for m in all_metadata:
    cats.update(m["categories"])
prices = [m["price"] for m in all_metadata if m["price"] is not None]

summary = {
    "total_items": len(all_metadata),
    "embedding_dim": 768,
    "unique_brands": len(brands),
    "unique_categories": len(cats),
    "price_range": [min(prices), max(prices)] if prices else None,
    "sample_brands": list(brands)[:10],
    "sample_categories": list(cats)[:10],
}
with open(os.path.join(OUT_DIR, "data_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Metadata saved: {len(all_metadata)} items")
print(f"  Brands: {len(brands)} unique")
print(f"  Categories: {len(cats)} unique")
print(f"  Prices: [{min(prices):.2f}, {max(prices):.2f}]" if prices else "  No prices")
print("Done.")
