"""SID inference for Musical_Instruments using trained RqVae.

加载 rqvae_final.pt → 跑 9922 item embedding → 输出 (9922, 3) SID 矩阵
到 /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids.npy

R36/R47: 使用 sentence-t5-xxl 预计算的 item_emb.npy (已存在)
R5: seed=42, 单卡运行即可 (推理不是瓶颈)
"""
import json
import os
import sys
import time

import numpy as np
import torch

# R47 imports
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_LearnableKappa")
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode, QuantizeDistance
from modules.tokenizer.semids import SemanticIdTokenizer
from data.schemas import SeqBatch


SEED = 42
EMB_NPY = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_emb.npy"
CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_m3_learnable_kappa/rqvae_final.pt"
OUT_NPY = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids_m3.npy"
OUT_IDS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_ids.json"

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 0.25

# 推理超参 (硬编码 R30/R43)
INFER_BATCH_SIZE = 512


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== SID inference ===")
    print(f"ckpt: {CKPT}")
    print(f"emb: {EMB_NPY}")
    print(f"out: {OUT_NPY}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Load ckpt
    print(f"[ckpt] loading {CKPT}")
    state = torch.load(CKPT, map_location=device, weights_only=True)
    print(f"[ckpt] loaded at global_step={state['global_step']}")

    # Build model (codebook_kmeans_init=False, ckpt 覆盖权重)
    model = RqVae(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        codebook_kmeans_init=False,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=N_LAYERS,
        n_cat_features=0,
        commitment_weight=COMMITMENT_WEIGHT,
        hyperbolic_mechanism="learnable_kappa",
        vae_init_curvature=1.0,
        quantize_distance_mode=QuantizeDistance.L2,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,} loaded")

    # Build tokenizer (用于 precompute_corpus_ids 去重, 不直接 inference)
    tokenizer = SemanticIdTokenizer(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=EMBED_DIM,
        codebook_size=CODEBOOK_SIZE,
        n_layers=N_LAYERS,
        n_cat_feats=0,
        rqvae_codebook_normalize=False,
        rqvae_sim_vq=False,
    )
    tokenizer.rq_vae = model
    tokenizer.eval()
    print(f"[tokenizer] SemanticIdTokenizer ready (for dedup only)")

    # Load embeddings
    arr = np.load(EMB_NPY).astype(np.float32)
    embeddings = torch.from_numpy(arr).to(device)
    n_items = embeddings.shape[0]
    print(f"[data] {n_items} items, dim={embeddings.shape[1]}")

    # Run inference batched (用 model.get_semantic_ids(x).sem_ids 拿到 (bsz, 3) SID 矩阵)
    sids = np.zeros((n_items, N_LAYERS), dtype=np.int64)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_items, INFER_BATCH_SIZE):
            end = min(start + INFER_BATCH_SIZE, n_items)
            x = embeddings[start:end]
            # get_semantic_ids 是原项目标准推理入口 (modules/rqvae.py L118)
            out = model.get_semantic_ids(x)
            sids[start:end] = out.sem_ids.detach().cpu().numpy()
            if start % (INFER_BATCH_SIZE * 10) == 0:
                elapsed = time.time() - t0
                print(f"  infer {end}/{n_items} ({100*end/n_items:.1f}%) elapsed={elapsed:.1f}s", flush=True)
    print(f"[infer] done in {time.time()-t0:.1f}s")

    # Save
    np.save(OUT_NPY, sids)
    print(f"[save] {OUT_NPY}: shape={sids.shape}, dtype={sids.dtype}")

    # Collapse check: per-layer code usage
    print(f"=== codebook usage ===")
    for layer in range(N_LAYERS):
        used = len(np.unique(sids[:, layer]))
        # histogram top-10 codes
        codes, counts = np.unique(sids[:, layer], return_counts=True)
        top10 = sorted(zip(counts, codes), reverse=True)[:10]
        print(f"  layer {layer}: {used}/256 codes used ({100*used/256:.1f}%)")
        print(f"    top-10: {[(int(c), int(n)) for n, c in top10]}")

    # 验证 item_ids.json 长度对齐
    if os.path.exists(OUT_IDS_JSON):
        with open(OUT_IDS_JSON) as f:
            ids = json.load(f)
        print(f"[check] item_ids.json has {len(ids)} ids, sids has {len(sids)} rows → {'OK' if len(ids)==len(sids) else 'MISMATCH'}")


if __name__ == "__main__":
    main()
