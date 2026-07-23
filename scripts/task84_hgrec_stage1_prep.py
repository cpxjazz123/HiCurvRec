#!/usr/bin/env python3
"""Task #84 Stage 1 prep — encode HG-Rec InAverageReviews item.json → item_emb.parquet.

R11.3 决策: paper 用 sentence-t5-xl (~2.4GB 没下载), 我们用 sentence-t5-base (GTR-T5-base 等价,
~250M params, 已缓存). 输出 dim 跟 XL 一致 (768d), 偏差在参数量. Loss-function 对比不受影响.

Runs in genrec_env (sentence_transformers + torch + CUDA ready).
GPU 1 (R7 强制空闲 — S3Rec 训练占 GPU 0, P5-SID 占 GPU 2).
"""

import sys
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET = "Instruments"
MODEL_NAME = "sentence-transformers/sentence-t5-base"  # 250M, downloaded; XL would be 1.2B
INPUT_JSON = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{DATASET}/{DATASET}.item.json"
OUTPUT_PARQUET = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{DATASET}/item_emb.parquet"
LOG_FILE = "/home/wlia0047/ar57/wenyu/GeneRec/logs/task84_hgrec_stage1_prep.log"

print("[Stage 1 prep] Loading model...")
model = SentenceTransformer(MODEL_NAME)

print(f"[Stage 1 prep] Reading {INPUT_JSON}...")
with open(INPUT_JSON, 'r') as f:
    item_info = json.load(f)
print(f"[Stage 1 prep] N items = {len(item_info)}")

print("[Stage 1 prep] Encoding items (this takes ~5-15 min on GPU)...")
item_embeddings = []
for item_id, info in item_info.items():
    semantics = (
        f"'title': {info['title']}, "
        f"'description': {info['description']}, "
        f"'brand': {info['brand']}, "
        f"'categories': {info['categories']}"
    )
    embedding = model.encode(
        semantics, show_progress_bar=False, convert_to_numpy=True
    )
    item_embeddings.append({"ItemID": item_id, "embedding": embedding.tolist()})

item_emb_df = pd.DataFrame(item_embeddings)
print(f"[Stage 1 prep] DataFrame shape = {item_emb_df.shape}")
print(item_emb_df.head(3))

print(f"[Stage 1 prep] Saving to {OUTPUT_PARQUET}...")
item_emb_df.to_parquet(OUTPUT_PARQUET, index=False)
print("[Stage 1 prep] DONE ✓")
