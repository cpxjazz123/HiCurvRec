#!/usr/bin/env python3
"""
Task #73 — Extract sentence-t5 embeddings for Musical_Instruments items.

Input:  DECOR/cache/AmazonReviews2023/Musical_Instruments/processed/metadata.sentence.json
        Format: {item_id (raw): text_description}

Output: external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy
        Format: (N_items, 768) float32 array, ordered by item_id (RemappedSequential id)
        Plus: sentence-t5-id-map.json mapping remapped_id -> raw_id

Pipeline:
1. Read metadata.sentence.json (raw item_id -> text)
2. Map raw_id to remapped sequential id using id_mapping.json
3. Use sentence-t5-base (768-d) to encode text
4. Save as .npy
"""
import os
import sys
import json
import numpy as np
import torch
from transformers import T5EncoderModel, AutoTokenizer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # Use GPU 1 (ETEGRec uses GPU 0)

INPUT_META = "/home/wlia0047/ar57/wenyu/GeneRec/DECOR/cache/AmazonReviews2023/Musical_Instruments/processed/metadata.sentence.json"
INPUT_IDMAP = "/home/wlia0047/ar57/wenyu/GeneRec/DECOR/cache/AmazonReviews2023/Musical_Instruments/processed/id_mapping.json"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments"
OUTPUT_NPY = os.path.join(OUTPUT_DIR, "sentence-t5-768d.npy")
OUTPUT_MAP = os.path.join(OUTPUT_DIR, "sentence-t5-id-map.json")


def main():
    if not os.path.exists(INPUT_META):
        print(f"❌ Metadata not found: {INPUT_META}")
        sys.exit(1)
    if not os.path.exists(INPUT_IDMAP):
        print(f"❌ ID mapping not found: {INPUT_IDMAP}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/4] Loading metadata from {INPUT_META}...")
    with open(INPUT_META) as f:
        meta = json.load(f)
    print(f"  Items with text: {len(meta)}")

    print(f"[2/4] Loading id_mapping from {INPUT_IDMAP}...")
    with open(INPUT_IDMAP) as f:
        idmap = json.load(f)
    # idmap structure: {"user2id": {...}, "item2id": {raw_item -> remapped_id}, "id2user": {...}, ...}
    if isinstance(idmap, dict) and "item2id" in idmap:
        raw_to_remapped = idmap["item2id"]
        print(f"  idmap[item2id] entries: {len(raw_to_remapped)}")
    else:
        # fallback: idmap is raw->remapped directly
        raw_to_remapped = idmap
    print(f"  id_map sample: {list(raw_to_remapped.items())[:3]}")

    remapped_to_text = {}
    missing = 0
    for raw_id, text in meta.items():
        if raw_id in raw_to_remapped:
            rem_id = raw_to_remapped[raw_id]
            if isinstance(rem_id, int):
                remapped_to_text[rem_id] = text
            elif isinstance(rem_id, list) and len(rem_id) == 1:
                remapped_to_text[rem_id[0]] = text
        else:
            missing += 1
    print(f"  Matched: {len(remapped_to_text)}, missing: {missing}")

    # Find max remapped id
    max_id = max(remapped_to_text.keys())
    print(f"  Max remapped id: {max_id}")
    print(f"  Items to encode: {len(remapped_to_text)}")

    print(f"[3/4] Loading sentence-t5-base (768-d)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = T5EncoderModel.from_pretrained("sentence-transformers/sentence-t5-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    model.eval()

    print(f"[4/4] Encoding items...")
    emb_dim = 768
    embeddings = np.zeros((max_id + 1, emb_dim), dtype=np.float32)

    # Batch encode
    batch_size = 64
    items = sorted(remapped_to_text.keys())
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch_ids = items[i:i + batch_size]
            batch_texts = [remapped_to_text[bid][:512] for bid in batch_ids]  # truncate long
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            # Mean pool over attention mask
            mask = inputs.attention_mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = pooled.cpu().numpy()
            for j, bid in enumerate(batch_ids):
                embeddings[bid] = pooled[j]
            if (i // batch_size) % 20 == 0:
                print(f"  Encoded {i + len(batch_ids)}/{len(items)}")

    # Save
    print(f"Saving {OUTPUT_NPY} shape={embeddings.shape}")
    np.save(OUTPUT_NPY, embeddings)
    with open(OUTPUT_MAP, "w") as f:
        json.dump({str(k): v for k, v in remapped_to_text.items()}, f)
    print(f"✅ Done. Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
