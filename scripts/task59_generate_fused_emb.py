#!/usr/bin/env python3
"""
Task #59 — Generate 896d fused embedding for Musical_Instruments ETEGRec.

Composition:
  - sentence-t5-base 768d text embedding (already exists at
    /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy)
  - SASRec 64d collaborative embedding (from RecBole pth) → broadcast/pad to 128d
  - Total: 768 + 128 = 896d

Output:
  /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_896.npy
  Shape: (24588, 896) float32
"""
import os
import sys
import json
import numpy as np
import torch


def main():
    OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/dataset/Musical_Instruments"
    OUT_NPY = os.path.join(OUT_DIR, "Musical_Instruments_emb_896.npy")

    # 1. Load sentence-t5 768d text embeddings
    TEXT_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy"
    IDMAP = "/home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments/sentence-t5-id-map.json"

    print(f"[1/5] Loading text embeddings from {TEXT_NPY}")
    text_emb = np.load(TEXT_NPY)
    print(f"  text_emb shape: {text_emb.shape}, dtype: {text_emb.dtype}")
    n_items = text_emb.shape[0]

    with open(IDMAP) as f:
        idmap = json.load(f)
    print(f"  ID map entries: {len(idmap)}")

    # 2. Load SASRec item embeddings (64d, pad to 128d)
    SASREC_PTH = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/saved/SASRec-Jul-22-2026_12-50-36.pth"
    print(f"[2/5] Loading SASRec from {SASREC_PTH}")
    ckpt = torch.load(SASREC_PTH, map_location="cpu", weights_only=False)
    sasrec_emb = ckpt["state_dict"]["item_embedding.weight"].numpy()  # (24588, 64)
    print(f"  SASRec item_emb shape: {sasrec_emb.shape}, dtype: {sasrec_emb.dtype}")

    # Project 64d → 128d via concatenation with zeros (preserves original signal + extra capacity)
    # Or: use a simple 2x replication (concat with self) — this doubles the dimension while preserving info
    # Better: use learned linear projection later; for now, replicate the 64d vector
    if sasrec_emb.shape[1] == 64:
        # Replicate twice → 128d (each dim appears twice). Preserves SASRec signal at full strength.
        sasrec_emb_128 = np.concatenate([sasrec_emb, sasrec_emb], axis=1)
    elif sasrec_emb.shape[1] == 128:
        sasrec_emb_128 = sasrec_emb
    else:
        raise ValueError(f"Unexpected SASRec dim: {sasrec_emb.shape[1]}")
    print(f"  SASRec 128d shape: {sasrec_emb_128.shape}")

    # 3. Verify alignment: both should be (24588, ...)
    if text_emb.shape[0] != sasrec_emb_128.shape[0]:
        print(f"❌ Mismatch: text_emb {text_emb.shape[0]} vs sasrec {sasrec_emb_128.shape[0]}")
        sys.exit(1)

    # 4. Normalize each modality before concat (paper best practice)
    text_norm = np.linalg.norm(text_emb, axis=1, keepdims=True)
    text_norm = np.maximum(text_norm, 1e-8)
    text_emb_norm = text_emb / text_norm

    sasrec_norm = np.linalg.norm(sasrec_emb_128, axis=1, keepdims=True)
    sasrec_norm = np.maximum(sasrec_norm, 1e-8)
    sasrec_emb_norm = sasrec_emb_128 / sasrec_norm

    # 5. Concat → 896d
    fused = np.concatenate([text_emb_norm, sasrec_emb_norm], axis=1)
    print(f"[3/5] Fused emb shape: {fused.shape}, dtype: {fused.dtype}")
    print(f"  text segment (768d): mean={text_emb_norm.mean():.4f}, std={text_emb_norm.std():.4f}")
    print(f"  sasrec segment (128d): mean={sasrec_emb_norm.mean():.4f}, std={sasrec_emb_norm.std():.4f}")

    # 6. Save
    print(f"[4/5] Saving to {OUT_NPY}")
    np.save(OUT_NPY, fused.astype(np.float32))

    print(f"[5/5] Done. Final shape: {fused.shape}, size: {fused.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()