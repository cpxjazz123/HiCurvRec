#!/usr/bin/env python3
"""
Task #73 — Generate text embedding substitute for LETTER (Instruments.emb-llama-td.npy)

LETTER paper uses LLaMA-3.1-8B text-description embedding (4096-d).
Original embedding missing. Use BAAI/bge-base-en-v1.5 (768-d) as substitute.

Output: external/LETTER/data/Instruments/Instruments.emb-bge.npy
Shape: (9922, 768) [matches original Items count, BGE dim]
"""
import sys, json, os
os.environ['HF_HOME'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'
os.environ['HF_HUB_CACHE'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'

import numpy as np
from sentence_transformers import SentenceTransformer

# Load items
ITEMS = '/home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data/Instruments/Instruments.item.json'
with open(ITEMS) as f:
    items = json.load(f)

print(f"Total items: {len(items)}")

# Build text in same order as items dict (Python 3.7+ dict preserves insertion order)
texts = []
item_ids = []
for iid, info in items.items():
    title = info.get('title', '')
    desc = info.get('description', '')
    # Concatenate title + description (truncate desc to 256 chars to avoid OOM)
    text = f"{title}. {desc[:256]}" if desc else title
    texts.append(text)
    item_ids.append(iid)

print(f"First text: {texts[0][:200]}")
print(f"Avg text length: {np.mean([len(t) for t in texts]):.0f}")

# Load BGE model (768-d, smaller than LLaMA but well-tested)
print("\nLoading BGE base model...")
model = SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder='/home/wlia0047/ar57_scratch/wenyu/hf_models')

# Encode all items
print(f"Encoding {len(texts)} items...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,  # L2 normalize for stable training
    convert_to_numpy=True,
)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Norm range: {np.linalg.norm(embeddings, axis=-1).min():.4f} - {np.linalg.norm(embeddings, axis=-1).max():.4f}")
print(f"Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.4f}")

# Save
out_path = '/home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data/Instruments/Instruments.emb-bge.npy'
np.save(out_path, embeddings.astype(np.float32))
print(f"\nSaved: {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")