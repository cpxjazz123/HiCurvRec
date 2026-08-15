"""Preprocess Musical_Instruments items → sentence-t5-xxl embeddings (R47).

Input: /home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments.item.json (9922 items)
Output: /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_emb.npy (9922, 768)

R36/R47: sentence-t5-xxl embedding via llm_client env var (HF cache on hj82_scratch2).
R44/R44b: dataset read from /home/wlia0047/ar57/wenyu/GeneRec/dataset/.
"""
import json
import os
import sys
import time
import numpy as np
import torch

ITEM_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments.item.json"
OUT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments"
OUT_NPY = os.path.join(OUT_DIR, "item_emb.npy")
OUT_IDS = os.path.join(OUT_DIR, "item_ids.json")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[preprocess] load {ITEM_JSON}")
    t0 = time.time()
    with open(ITEM_JSON, "r") as f:
        items = json.load(f)
    n_items = len(items)
    print(f"[preprocess] {n_items} items loaded in {time.time()-t0:.1f}s")

    # Build text: title + description (truncated to 512 chars)
    item_ids = sorted(items.keys(), key=lambda x: int(x))
    texts = []
    for iid in item_ids:
        it = items[iid]
        title = (it.get("title") or "").strip()
        desc = (it.get("description") or "").strip()[:512]
        text = f"{title}. {desc}".strip(". ")
        if not text:
            text = "(no description)"
        texts.append(text)
    print(f"[preprocess] built {len(texts)} texts, avg len = {sum(len(t) for t in texts)/len(texts):.0f} chars")

    # Load sentence-t5-xxl (fp32, 不换 dtype — user 要求保持原精度)
    from sentence_transformers import SentenceTransformer
    print(f"[preprocess] loading sentence-t5-xxl (fp32) ...")
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/sentence-t5-xxl", device="cuda")
    print(f"[preprocess] model loaded in {time.time()-t0:.1f}s, device={model.device}")

    # Multi-GPU encoding via start_multi_process_pool (sentence-transformers native)
    # 4 GPU 各跑独立 batch, texts 自动切片
    n_gpus = torch.cuda.device_count() if hasattr(torch, "cuda") else 1
    print(f"[preprocess] using {n_gpus} GPUs for encoding")
    t0 = time.time()
    if n_gpus > 1:
        pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"][:n_gpus])
        embeddings = model.encode_multi_process(
            texts,
            pool=pool,
            batch_size=16,  # per-GPU batch_size (4 GPU × 16 = 64 effective)
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        model.stop_multi_process_pool(pool)
    else:
        embeddings = model.encode(
            batch_size=64,
            sentences=texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    elapsed = time.time() - t0
    print(f"[preprocess] encoded in {elapsed:.1f}s ({n_items/elapsed:.1f} items/s)")
    print(f"[preprocess] embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Save
    np.save(OUT_NPY, embeddings.astype(np.float32))
    with open(OUT_IDS, "w") as f:
        json.dump(item_ids, f)
    print(f"[preprocess] saved:")
    print(f"  {OUT_NPY} ({os.path.getsize(OUT_NPY)/1024/1024:.1f} MB)")
    print(f"  {OUT_IDS} ({len(item_ids)} ids)")


if __name__ == "__main__":
    main()