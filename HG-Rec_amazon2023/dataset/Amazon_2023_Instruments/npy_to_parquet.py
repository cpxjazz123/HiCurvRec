"""R53 v3.8: item_emb.npy → item_emb.parquet (HG-Rec HRQ-VAE 训练入口只读 .parquet).

输入: dataset/Amazon_2023_Instruments/item_emb.npy (24587, 768) float32
      dataset/Amazon_2023_Instruments/item_ids.json (list[str] 长度 24587)
输出: dataset/Amazon_2023_Instruments/item_emb.parquet (24587, 2)
      列: ItemID (str) | embedding (list[float32])
"""
import json

import numpy as np
import pandas as pd

DATASET_DIR = "./dataset/Amazon_2023_Instruments"

item_emb = np.load(f"{DATASET_DIR}/item_emb.npy").astype(np.float32)
with open(f"{DATASET_DIR}/item_ids.json", 'r') as f:
    item_ids = json.load(f)

assert item_emb.shape[0] == len(item_ids), (
    f"item_emb rows={item_emb.shape[0]} != item_ids len={len(item_ids)}"
)

df = pd.DataFrame({
    'ItemID': list(item_ids),
    'embedding': [row.tolist() for row in item_emb],
})
df.to_parquet(f"{DATASET_DIR}/item_emb.parquet", index=False)

print(f"[npy_to_parquet] {item_emb.shape} -> item_emb.parquet ({df.shape[0]} rows)")