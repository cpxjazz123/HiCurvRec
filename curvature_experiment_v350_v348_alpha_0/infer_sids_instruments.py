"""SID inference for Musical_Instruments using trained RqVae.

加载 ./out/rqvae/instruments/rqvae_final.pt → 跑 9922 item embedding →
输出 (9922, 3) SID 矩阵到 ./out/rqvae/instruments/sids_raw.npy

R36/R47: 使用 sentence-t5-xxl 预计算的 item_emb.npy
R52/R53: 路径相对 cwd, 从 curvature_config.py 硬编码导入, 无 env var / 无 CLI 参数.
R5: seed=42, 单卡运行即可 (推理不是瓶颈)
"""
import json
import os
import sys
import time

import numpy as np
import torch

# R47 imports
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3")
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from data.schemas import SeqBatch

# R52/R53: 从 curvature_config.py 硬编码导入路径
from curvature_config import (
    ITEM_EMB_NPY as EMB_NPY,
    RQVAE_CKPT_PATH as CKPT,
    RAW_SIDS_NPY as OUT_NPY,
    ITEM_IDS_JSON as OUT_IDS_JSON,
)

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 0.25

# R5/R51: seed=42 (推理端固定)
SEED = 42

# 推理超参 (硬编码 R30/R43)
INFER_BATCH_SIZE = 512
# A2 (SID beam search): 每层展开 top-BEAM 候选 (RQ-VAE 论文式 beam 解码, 降 SID 碰撞)
SID_BEAM = 8


def beam_search_sem_ids(model, x, beam=SID_BEAM, gumbel_t=0.001):
    """A2: RQ-VAE 论文式 beam search SID 解码 (贪心 argmin → top-beam 展开).

    每层对 beam 候选展开全部 codebook (K), 按累计 Poincaré 距离 (双曲 argmin) 保留 top-beam,
    并在 M2/M3 曲率路径下推进 residual. 返回 (B, n_layers) 最佳 beam 的 SID 矩阵.

    beam=1 退化为贪心 argmin (与 get_semantic_ids 一致), 用于 A/B 一致性验证.
    """
    import math as _math
    from modules.hyperbolic import (
        _expmap0_t, _poincare_distance_t, _mobius_add_t, _logmap0_t,
        _transport_between_t, C_MAX as _C_MAX,
    )
    B = x.shape[0]
    D = model.embed_dim
    K = model.codebook_size
    n_layers = model.n_layers
    device = x.device

    z = model.encode(x)  # (B, D)
    # beam 状态: residual (B, beam, D), codes (B, beam, L), scores (B, beam), prefix_embs (B, beam, L, D)
    beam = max(1, min(beam, K))
    residual = z.unsqueeze(1)  # 初始 beam=1 (所有成员同一 residual 会产出重复候选, 无法探索)
    codes = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
    scores = torch.zeros(B, 1, device=device)
    prefix_embs = torch.zeros(B, 1, 0, dtype=torch.float32, device=device)
    prev_c = None

    for li in range(n_layers):
        q = model.layers[li]
        codebook = q.embedding.weight  # (K, D)
        cur_beam = residual.shape[1]  # 实际 beam 宽度 (初始 1, 每层扩到 target)
        Bb = B * cur_beam
        res_flat = residual.reshape(Bb, -1)  # (B*cur_beam, D)
        # per-item 曲率 (L0 全局 / L1+ prefix-conditioned)
        if q.prefix_routing and prefix_embs.shape[2] > 0:
            parts = list(prefix_embs.reshape(Bb, -1, D).unbind(2))
            if model.gate_M3_transport and prev_c is not None:
                c_sig = torch.log(prev_c.clamp(min=1e-6)) / _math.log(_C_MAX)
                parts.append(c_sig.reshape(Bb, 1))
            prefix_emb = torch.cat(parts, dim=-1)
            c_l = q.get_c_per_item(prefix_emb)  # (Bb,)
        else:
            c_l = q.get_c().expand(Bb)
        c_exp = c_l.view(-1, 1, 1)
        # 距离矩阵 (Poincaré 双曲, 与 Quantize.forward 同口径)
        if q.hyperbolic_distance:
            latent_h = _expmap0_t(res_flat.unsqueeze(1), c_exp)
            cb_exp0 = codebook.unsqueeze(0).expand(Bb, K, -1)
            codebook_h = _expmap0_t(cb_exp0, c_exp)
            dist = _poincare_distance_t(latent_h.expand(Bb, K, -1), codebook_h, c_exp).squeeze(-1)  # (Bb, K)
        else:
            dist = (
                (res_flat**2).sum(1, keepdim=True)
                + (codebook.T**2).sum(0, keepdim=True)
                - 2 * res_flat @ codebook.T
            )
        # 累计得分 = 各层 Poincaré 距离和 (越小越好); 每层全局展开 cur_beam*N_EXPAND 候选取 top-beam,
        # 并按 code 去重填槽 (避免同一 code 在不同父路径重复占槽)
        N_EXPAND = min(beam, K)
        topd, topk_idx = dist.topk(N_EXPAND, dim=-1, largest=False)  # (Bb, N_EXPAND)
        cand_scores = scores.reshape(Bb, 1) + topd  # (Bb, N_EXPAND)
        cand_flat = cand_scores.reshape(B, cur_beam * N_EXPAND)
        # 全局排序; 但候选可能重复 (同 code 出现在多个父路径) — 用 codes 去重填槽
        cand_parent = torch.arange(cur_beam, device=device).view(1, cur_beam).expand(B, cur_beam).reshape(B, cur_beam, 1).expand(B, cur_beam, N_EXPAND).reshape(B, -1)  # (B, cur_beam*N) per-item 局部父索引
        cand_code = topk_idx.reshape(B, -1)  # (B, cur_beam*N)
        order = cand_flat.sort(dim=-1, descending=False).indices  # (B, cur_beam*N)
        chosen = []
        for bi in range(B):
            picked_codes = set()
            picked = []
            for oi in order[bi].tolist():
                c = int(cand_code[bi, oi].item())
                if c in picked_codes:
                    continue
                picked_codes.add(c)
                picked.append(oi)
                if len(picked) >= beam:
                    break
            chosen.append(picked)
        chosen = torch.tensor(chosen, device=device)  # (B, beam) 全局展开索引
        parent = cand_parent.gather(1, chosen)  # (B, beam)
        code_idx = cand_code.gather(1, chosen)  # (B, beam)
        top_scores = cand_flat.gather(1, chosen)

        # 推进 residual (M2 Möbius 减法 + M3 transport), 每个存活候选用自己的曲率
        sel_beam = chosen.shape[1]  # 存活候选数 (= target beam)
        Bsel = B * sel_beam
        chosen_emb = codebook[code_idx]  # (B, sel_beam, D)
        res_par = residual[torch.arange(B, device=device).unsqueeze(1), parent]  # (B, sel_beam, D)
        c_par = c_l.reshape(B, cur_beam)[torch.arange(B, device=device).unsqueeze(1), parent].view(-1, 1)  # (B*sel_beam,1)
        if model.gate_M2_intrinsic:
            h_r = _expmap0_t(res_par.reshape(Bsel, -1), c_par)
            h_e = _expmap0_t(chosen_emb.reshape(Bsel, -1), c_par)
            h_next = _mobius_add_t(-h_e, h_r, c_par)
            res_new = _logmap0_t(h_next, c_par)
        else:
            res_new = res_par.reshape(Bsel, -1) - chosen_emb.reshape(Bsel, -1)
        if model.gate_M3_transport and li < n_layers - 1:
            c_next = model.layers[li + 1].get_c().view(-1, 1)
            res_new = _transport_between_t(res_new, c_par, c_next)
            prev_c = c_next.reshape(1, 1).expand(Bsel, 1)
        residual = res_new.reshape(B, sel_beam, -1)
        codes = torch.cat([codes[torch.arange(B, device=device).unsqueeze(1), parent], code_idx.unsqueeze(-1)], dim=-1)
        scores = top_scores
        # prefix_embs 更新: 每候选已选 codeword 拼接
        pe = chosen_emb  # (B, beam, D)
        if prefix_embs.shape[2] > 0:
            parent_pe = prefix_embs[torch.arange(B, device=device).unsqueeze(1), parent]
            pe = torch.cat([parent_pe, pe.unsqueeze(2)], dim=2)  # (B, beam, L+1, D)
        else:
            pe = pe.unsqueeze(2)
        prefix_embs = pe

    return codes  # (B, beam, n_layers)


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
    # C26: 与训练一致 (HG-Rec 极简: 关 M2/M3/Sinkhorn, 固定 c=1)
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
        gate_M2_intrinsic=False,  # C28 (C29 rollback): 关 M2
        gate_M3_transport=True,   # C28: 开 M3 transport
        hyperbolic_distance=True,  # HG-Rec 双曲 argmin
        sk_eps=0.0,                # C26 HG-Rec 极简: 关 Sinkhorn
        prefix_router_layers=None,
        margin_reg_weight=0.0,     # C26 HG-Rec: 关 C5
        use_tcu=False,             # C26 HG-Rec: 关 TCU
        use_mcdq=False,            # C26 HG-Rec: 关 MCDQ
        use_scs=False,             # C26 HG-Rec: 关 SCS
        scs_eps_scale=1.0,
        use_fixed_curvature=True,  # v337: c=1.0 固定 (与 Stage 1 训练一致, 关 cyclic + 关 curriculum)
        c_fixed=1.0,
        use_curriculum_curvature=False,  # v337: 关 curriculum schedule (与 Stage 1 训练一致)
        c_start=0.05,
        c_end=1.0,  # v337: c_end=1.0 与 Stage 1 训练 c_fixed=1.0 一致 (vs baseline 0.7)
        curriculum_steps=1,  # inference 时 set_step(>=1) 强制 t=1 → c=c_end
    ).to(device)
    # v337: 强制 inference 走 fixed c=1.0 (与 Stage 1 训练 USE_CYCLIC_CURVATURE=False + c_fixed=1.0 一致)
    model.set_curriculum_step(10**9)  # step >> curriculum_steps → t=1 → c=1.0
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
    # Issue #159/#161/#162 迁移: 导出 per-item 曲率状态 + 响应特征
    # curvature_state: (n_items, 3) 每 item 每层有效曲率 c_l,i
    # response: (n_items, 3, 4) [c, boundary_distance, margin, residual_norm]
    curv_state = np.zeros((n_items, N_LAYERS), dtype=np.float32)
    response = np.zeros((n_items, N_LAYERS, 4), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_items, INFER_BATCH_SIZE):
            end = min(start + INFER_BATCH_SIZE, n_items)
            x = embeddings[start:end]
            # 用 native get_semantic_ids (Sinkhorn 均衡 + 贪心) — 与 tokenizer.precompute_corpus_ids 严格一致,
            # 保证 curvature_state/response 与 decoder codebook SID 路径对齐 (A2 beam 实测唯一 SID 更少, 判定 NO-GO)
            out = model.get_semantic_ids(x)
            sem_ids = out.sem_ids  # (B, n_layers)
            sids[start:end] = sem_ids.detach().cpu().numpy()
            # 逐层曲率状态 + 响应特征 (复刻 Quantize.forward 距离计算)
            z = model.encode(x)
            residual = z
            prefix_codes = []
            prev_c = None
            import math as _math
            from modules.hyperbolic import _expmap0_t, _poincare_distance_t, _mobius_add_t, _logmap0_t, _transport_between_t, C_MAX as _C_MAX
            for li in range(N_LAYERS):
                q = model.layers[li]
                # Issue #154: prefix_emb (与 get_semantic_ids 同路径)
                prefix_emb = None
                if q.prefix_routing:
                    parts = list(prefix_codes)
                    if model.gate_M3_transport and prev_c is not None:
                        c_sig = torch.log(prev_c.clamp(min=1e-6)) / _math.log(_C_MAX)
                        parts.append(c_sig)
                    prefix_emb = torch.cat(parts, dim=-1) if parts else None
                c_l = q.get_c_per_item(prefix_emb)  # (1,) 或 (B,) per-item
                curv_state[start:end, li] = c_l.detach().cpu().numpy()
                # 距离矩阵 (双曲分支, per-item c 广播)
                codebook = q.embedding.weight
                B = residual.shape[0]
                K = codebook.shape[0]
                if q.hyperbolic_distance:
                    c_exp = c_l.view(-1, 1, 1)
                    latent_h = _expmap0_t(residual.unsqueeze(1), c_exp)
                    cb_exp0 = codebook.unsqueeze(0).expand(B, K, -1)
                    codebook_h = _expmap0_t(cb_exp0, c_exp)
                    dist = _poincare_distance_t(latent_h.expand(B, K, -1), codebook_h, c_exp).squeeze(-1)
                else:
                    dist = (
                        (residual**2).sum(axis=1, keepdim=True)
                        + (codebook.T**2).sum(axis=0, keepdim=True)
                        - 2 * residual @ codebook.T
                    )
                dist_sorted, _ = dist.detach().sort(dim=-1)
                boundary = dist_sorted[:, 0]                 # 最近 codebook 距离
                margin = dist_sorted[:, 1] - dist_sorted[:, 0]  # top1-top2
                response[start:end, li, 0] = c_l.detach().cpu().numpy()
                response[start:end, li, 1] = boundary.cpu().numpy()
                response[start:end, li, 2] = margin.cpu().numpy()
                response[start:end, li, 3] = residual.norm(dim=-1).cpu().numpy()
                # residual 推进 (与 beam search / get_semantic_ids 同路径: M2 Möbius / M3 transport)
                emb = q.get_item_embeddings(sem_ids[:, li])
                if model.gate_M2_intrinsic:
                    c_per = c_l.view(-1, 1)
                    h_r = _expmap0_t(residual, c_per)
                    h_e = _expmap0_t(emb, c_per)
                    h_next = _mobius_add_t(-h_e, h_r, c_per)
                    residual = _logmap0_t(h_next, c_per)
                else:
                    residual = residual - emb
                if model.gate_M3_transport and li < N_LAYERS - 1:
                    c_next = model.layers[li + 1].get_c().view(-1, 1)
                    residual = _transport_between_t(residual, c_l.view(-1, 1), c_next)
                    prev_c = c_next.expand(B, 1)
                prefix_codes.append(emb)
            if start % (INFER_BATCH_SIZE * 10) == 0:
                elapsed = time.time() - t0
                print(f"  infer {end}/{n_items} ({100*end/n_items:.1f}%) elapsed={elapsed:.1f}s", flush=True)
    print(f"[infer] done in {time.time()-t0:.1f}s")

    # Save
    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, sids)
    print(f"[save] {OUT_NPY}: shape={sids.shape}, dtype={sids.dtype}")
    np.save(os.path.join(os.path.dirname(OUT_NPY), "curvature_state.npy"), curv_state)
    np.save(os.path.join(os.path.dirname(OUT_NPY), "response.npy"), response)
    print(f"[save] curvature_state.npy: {curv_state.shape} | response.npy: {response.shape}")
    print(f"[curv] per-layer c: {[float(curv_state[0, li]) for li in range(N_LAYERS)]}")
    print(f"[resp] boundary mean: {[float(response[:, li, 1].mean()) for li in range(N_LAYERS)]}")

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
