"""v45 Stage 2 wrapper — Poincaré K-means 重训 v19 RQ-VAE codebook → 输出新 SID.

设计:
1. 加载 v19 RQ-VAE ckpt (rqvae_final.pt)
2. 把每层 codebook 用 Poincaré K-means (c=0.5) 重新初始化 (基于 v19 ckpt encoder 输出的 z)
3. 用新 codebook 推理 SID → (9922, 3) 矩阵
4. 转 HG-Rec 格式 (9922, 4) [sid0/1/2 + 0 PAD]

Stage 1 不重训 (R40 通过 RQ-VAE ckpt 复用满足自包含).
Stage 2 端曲率机制变更 — R36 硬编码 c=0.5, 无 LR/dropout/wd sweep.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")

from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode

SEED = 42
EMB_NPY = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_emb.npy"
CKPT_V19 = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt"
OUT_NPY_HGREC = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/dataset/Instruments/Instruments_v45_sids_for_hgrec.npy"
OUT_NPY_RAW = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids_v45_poincare_kmeans.npy"

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3

INFER_BATCH_SIZE = 512

# v45 创新: Poincaré K-means 曲率
POINCARE_C = 0.5


def reinit_codebooks_poincare(model, embeddings, c=POINCARE_C):
    """对 model 每层 codebook, 用 Poincaré K-means 重新初始化.

    1. 用 model.encode() 把 9922 个 item embedding 编码到 latent z
    2. 每层: 收集 (residual, c_l) per-item, 在 Poincaré 球面上跑 K-means
    3. 把新 codebook 写回 model.layers[li].embedding.weight
    """
    from _lib.poincare_kmeans_v45 import poincare_kmeans_init_

    print(f"[v45 poincare_kmeans] reinitializing {N_LAYERS} codebooks with c={c}")
    device = embeddings.device

    with torch.no_grad():
        z = model.encode(embeddings)  # (9922, 32)

        for li in range(N_LAYERS):
            q = model.layers[li]
            codebook = q.embedding.weight.data  # (K=256, D=32)

            # 收集该层 residual (考虑 c_l per-item, M2/M3 路径)
            residual = z.clone()  # 初始 residual = z
            prefix_codes = []
            prev_c = None
            import math as _math
            from modules.hyperbolic import _expmap0_t, _poincare_distance_t, _mobius_add_t, _logmap0_t, _transport_between_t, C_MAX as _C_MAX

            for prev_li in range(li):
                pq = model.layers[prev_li]
                c_l = pq.get_c_per_item(None) if hasattr(pq, 'get_c_per_item') else pq.get_c()
                if c_l.dim() == 0:
                    c_l = c_l.expand(z.shape[0])
                # 贪心选最近 codeword (沿用 get_semantic_ids 同口径)
                cb_exp = pq.embedding.weight
                if pq.hyperbolic_distance:
                    c_exp = c_l.view(-1, 1, 1)
                    latent_h = _expmap0_t(residual.unsqueeze(1), c_exp)
                    codebook_h = _expmap0_t(cb_exp.unsqueeze(0).expand(z.shape[0], 256, -1), c_exp)
                    dist = _poincare_distance_t(latent_h.expand(z.shape[0], 256, -1), codebook_h, c_exp).squeeze(-1)
                else:
                    dist = ((residual**2).sum(1, keepdim=True) + (cb_exp.T**2).sum(0, keepdim=True) - 2 * residual @ cb_exp.T)
                code_idx = dist.argmin(dim=-1)  # (9922,)
                emb = pq.get_item_embeddings(code_idx)
                if model.gate_M2_intrinsic:
                    c_per = c_l.view(-1, 1)
                    h_r = _expmap0_t(residual, c_per)
                    h_e = _expmap0_t(emb, c_per)
                    h_next = _mobius_add_t(-h_e, h_r, c_per)
                    residual = _logmap0_t(h_next, c_per)
                else:
                    residual = residual - emb
                if model.gate_M3_transport and prev_li < N_LAYERS - 1:
                    c_next = model.layers[prev_li + 1].get_c().view(-1, 1)
                    residual = _transport_between_t(residual, c_l.view(-1, 1), c_next)
                    prev_c = c_next.expand(z.shape[0], 1)
                prefix_codes.append(emb)

            # 用 Poincaré K-means 重训 codebook[li]
            old_norm = codebook.norm(dim=-1).mean().item()
            poincare_kmeans_init_(codebook, x=residual.detach(), c=c)
            new_norm = codebook.norm(dim=-1).mean().item()
            print(f"  layer {li}: Poincaré K-means done; old norm={old_norm:.4f}, new norm={new_norm:.4f}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== v45 Stage 2 SID inference (Poincaré K-means reinit) ===")
    print(f"ckpt: {CKPT_V19}")
    print(f"emb:  {EMB_NPY}")
    print(f"out:  {OUT_NPY_HGREC}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    state = torch.load(CKPT_V19, map_location=device, weights_only=True)
    print(f"[ckpt] loaded at global_step={state['global_step']}")

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
        commitment_weight=0.25,
        gate_M2_intrinsic=False,
        gate_M3_transport=True,
        hyperbolic_distance=True,
        sk_eps=0.0,
        prefix_router_layers=None,
        margin_reg_weight=0.0,
        use_tcu=False,
        use_mcdq=False,
        use_scs=False,
        scs_eps_scale=1.0,
        use_fixed_curvature=True,
        c_fixed=1.0,
        use_curriculum_curvature=True,
        c_start=0.05,
        c_end=0.7,
        curriculum_steps=1,
    ).to(device)
    model.set_curriculum_step(10**9)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,} loaded")

    arr = np.load(EMB_NPY).astype(np.float32)
    embeddings = torch.from_numpy(arr).to(device)
    n_items = embeddings.shape[0]
    print(f"[data] {n_items} items, dim={embeddings.shape[1]}")

    # v45 核心: Poincaré K-means 重训 codebook
    reinit_codebooks_poincare(model, embeddings, c=POINCARE_C)

    # 推理新 SID (用 Poincaré K-means 后的 codebook)
    sids = np.zeros((n_items, N_LAYERS), dtype=np.int64)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_items, INFER_BATCH_SIZE):
            end = min(start + INFER_BATCH_SIZE, n_items)
            x = embeddings[start:end]
            out = model.get_semantic_ids(x)
            sids[start:end] = out.sem_ids.detach().cpu().numpy()
            if start % (INFER_BATCH_SIZE * 10) == 0:
                elapsed = time.time() - t0
                print(f"  infer {end}/{n_items} ({100*end/n_items:.1f}%) elapsed={elapsed:.1f}s", flush=True)
    print(f"[infer] done in {time.time()-t0:.1f}s")

    # Save raw (9922, 3)
    np.save(OUT_NPY_RAW, sids)
    print(f"[save] {OUT_NPY_RAW}: shape={sids.shape}, dtype={sids.dtype}")

    # 转 HG-Rec 格式 (9922, 4) [sid0/1/2 + 0 PAD]
    sids_hgrec = np.zeros((n_items, 4), dtype=np.int64)
    sids_hgrec[:, :3] = sids
    np.save(OUT_NPY_HGREC, sids_hgrec)
    print(f"[save] {OUT_NPY_HGREC}: shape={sids_hgrec.shape}, dtype={sids_hgrec.dtype}")

    # codebook usage 统计
    print(f"=== codebook usage (after Poincaré K-means) ===")
    for layer in range(N_LAYERS):
        used = len(np.unique(sids[:, layer]))
        codes, counts = np.unique(sids[:, layer], return_counts=True)
        top10 = sorted(zip(counts, codes), reverse=True)[:10]
        print(f"  layer {layer}: {used}/256 codes used ({100*used/256:.1f}%)")
        print(f"    top-10: {[(int(c), int(n)) for n, c in top10]}")


if __name__ == "__main__":
    main()