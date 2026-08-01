#!/usr/bin/env python3
"""
Task #457 / Issue #164 [方向D Gate 0-3] 深层码字靠近双曲边界判别间隔 + SID 翻转率零训练实证.

零训练 (frozen HRQ-VAE ckpt), CPU 即可 (纯矩阵运算, 无反向传播).
- Gate 0: 三种径向对象 (单层 codeword 半径 / 累计路径节点半径 / residual 半径) 逐层 profile
- Gate 1: 径向缩放 s in {0.6, 0.8, 1.0, 1.2, 1.4} 同时改 residual + codeword 半径, 测 Δ_l = d_2 - d_1 margin
- Gate 2: 扰动 σ in {0.001, 0.005, 0.01} 测 SID FlipRate
- Gate 3: 欧氏距离对照, 排除普通向量缩放效应

锚定 Task #84 健康 HRQ-VAE ckpt (R@10=0.1020, Musical_Instruments, c=1 baseline).
"""
import os, sys, json, hashlib, argparse
import numpy as np
import torch
import torch.nn.functional as F

# 强制 CPU 跑 (零训练, 避免 GPU 占用冲突)
DEVICE = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 切到 HG-Rec 目录 (上游代码在那里)
HG_REC_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec"
sys.path.insert(0, HG_REC_DIR)

from model.utils import (  # type: ignore  # from HG-Rec/model/utils.py
    artanh, proj_to_ball, mobius_add, lambda_x,
    expmap0, logmap0, poincare_distance, MLP,
    HVectorQuantization, HResidualVectorQuantization, EmbDataset,
)
from model.hrqvae import HRQVAE  # type: ignore
from torch.utils.data import DataLoader

# ====== 锚定路径 ======
# Task #84 Stage 1 HRQ-VAE baseline (Poincaré c=1, num_emb=[64,128,256], beta=1.0)
# best_collision_model: epoch 64 collision=0.0910 (健康 baseline, 跟 Stage 4 R@10=0.1020 一致)
CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"
ITEM_EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

# Task #84 baseline config (跟 Stage 1 train_hrqvae.py 一致)
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
LAYERS = [512, 256, 128, 64]
BETA = 0.25
SK_EPSILONS = [0.0, 0.0, 0.000]
C_POINCARE = 1.0  # HG-Rec Task #84 baseline c=1

# Issue #164 spec: s grid + sigma grid + tau
S_GRID = [0.6, 0.8, 1.0, 1.2, 1.4]
SIGMA_GRID = [0.001, 0.005, 0.01]
TAU_SMALL_MARGIN = 0.01  # 固定, 不允许事后选择


# ============================================================
# Poincaré 工具
# ============================================================
def radialize_ball(x: torch.Tensor, c: float):
    """project to Poincaré ball (内含)"""
    return proj_to_ball(x, c)


def radial_rescale_ball(x: torch.Tensor, s: float, c: float) -> torch.Tensor:
    """T_s(x) = exp_0^c(s · log_0^c(x)) 径向缩放 (s<1 向原点拉, s>1 向边界推).
    处理原点在 c=1 Poincaré ball = {x: ||x|| < 1/c^0.5 = 1}.
    """
    eps = 1e-10
    sqrt_c = c ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    # 对原点 (norm→0) 特殊处理, T_s(0) = 0 (径向缩放不动原点)
    is_zero = (norm < eps).float()
    safe_norm = norm * (1.0 - is_zero) + eps * is_zero
    # log_0^c(x) = (artanh(sqrt_c * ||x||) / (sqrt_c * ||x||)) * x
    log_x = (artanh(sqrt_c * safe_norm) / (sqrt_c * safe_norm)) * x
    log_x = log_x * (1.0 - is_zero)  # 原点 → 0
    # s · log_x
    s_log_x = s * log_x
    # exp_0^c(s_log_x) = (tanh(sqrt_c * ||s_log_x||) / (sqrt_c * ||s_log_x||)) * s_log_x
    s_norm = s_log_x.norm(dim=-1, keepdim=True).clamp_min(eps)
    s_is_zero = (s_norm < eps).float()
    safe_s_norm = s_norm * (1.0 - s_is_zero) + eps * s_is_zero
    exp_x = (torch.tanh(sqrt_c * safe_s_norm) / (sqrt_c * safe_s_norm)) * s_log_x
    exp_x = exp_x * (1.0 - s_is_zero)
    return proj_to_ball(exp_x, c)


# ============================================================
# Gate 0: 三种径向对象 profile
# ============================================================
def gate0_radial_profile(model: HRQVAE, embeddings: torch.Tensor):
    """三对象 × 三层 × 七统计量 (mean / std / p5 / p50 / p95 / min / max) profile.

    A. 单层 codeword 半径 ρ_l^code = d_c(0, e_{l,j})
       (对每层 VQ module 取 codebook entries, 每个做 proj_to_ball + 算到原点距离)
    B. 累计路径节点半径 ρ_l^path = d_c(0, e_{i,0} ⊕ ... ⊕ e_{i,l})
       (前向跑 Stage 1, 对每层索引取 codeword, Möbius add 累加, 算到原点距离)
    C. residual 半径 ρ_l^res = d_c(0, r_{i,l})
       (r_l = x - sum_{k<l} x_q^k)
    """
    model.eval()
    N = embeddings.shape[0]
    L = len(NUM_EMB_LIST)

    # ---- A. 单层码字半径 ρ_l^code ----
    code_radials = []  # list of L tensors
    with torch.no_grad():
        for vq in model.hrq.vq_layers:
            cb_raw = vq.embeddings.weight  # (K_l, e_dim)
            cb_h = proj_to_ball(expmap0(cb_raw, vq.c), vq.c)  # (K_l, e_dim) in ball
            zero = torch.zeros_like(cb_h[:1])
            r = poincare_distance(zero, cb_h, vq.c)  # (K_l,)
            code_radials.append(r.detach().cpu().numpy())

    # ---- B & C: 前向, 收集 residual 与累计路径 ----
    bs = 256
    n_batches = (N + bs - 1) // bs
    path_radials = [[] for _ in range(L)]
    res_radials = [[] for _ in range(L)]

    with torch.no_grad():
        for bi in range(n_batches):
            x = embeddings[bi * bs : (bi + 1) * bs].to(DEVICE)  # (B, e_dim) (already encoded)
            residual = x.clone()
            cum_x_q = torch.zeros_like(x)
            for li, vq in enumerate(model.hrq.vq_layers):
                # 当前 residual 进 VQ, 取 indices + quantized
                latent = residual
                codebook_raw = vq.embeddings.weight  # (K_l, e_dim)
                # 跑 VQ forward (但简化: 直接算距离 + argmin)
                codebook_h = proj_to_ball(expmap0(codebook_raw, vq.c), vq.c)
                # 算距离 (B, K_l)
                B = latent.shape[0]
                K = codebook_h.shape[0]
                lat_h = proj_to_ball(expmap0(latent, vq.c), vq.c)
                lat_exp = lat_h.unsqueeze(1).expand(B, K, -1)
                cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
                d = poincare_distance(lat_exp, cb_exp, vq.c).squeeze(-1)  # (B, K)
                indices = torch.argmin(d, dim=-1)  # (B,)
                x_q_l = codebook_raw[indices]  # (B, e_dim)
                # 累计路径 (在 tangent space add, logmap0 / expmap0 跟 util.py forward 同)
                # 简化: 用 tangent space (Euclidean) add, 然后 proj_to_ball + 算到原点距离
                # 跟 forward 的 x_q = x_q + x_res 一致 (在 tangent 空间)
                cum_x_q = cum_x_q + x_q_l
                # 算 ρ_l^path: cum_x_q 转回 ball 空间
                cum_h = proj_to_ball(expmap0(cum_x_q, vq.c), vq.c)
                zero = torch.zeros_like(cum_h[:1])
                r_path = poincare_distance(zero, cum_h, vq.c)  # (B,)
                path_radials[li].append(r_path.detach().cpu().numpy())
                # 算 ρ_l^res: residual = x - cum_x_q
                residual = residual - x_q_l
                res_h = proj_to_ball(expmap0(residual, vq.c), vq.c)
                r_res = poincare_distance(zero, res_h, vq.c)
                res_radials[li].append(r_res.detach().cpu().numpy())

    # ---- 统计量 ----
    def stats(arr):
        a = np.concatenate(arr) if isinstance(arr, list) else arr
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    out = {}
    for li in range(L):
        out[f"layer_{li}"] = {
            "K": NUM_EMB_LIST[li],
            "code_radial": stats(code_radials[li]),
            "path_radial": stats(path_radials[li]),
            "residual_radial": stats(res_radials[li]),
        }
    return out


# ============================================================
# Gate 1: 零训练径向干预, 测 margin
# ============================================================
def gate1_radial_margin(model: HRQVAE, embeddings: torch.Tensor):
    """s in S_GRID 同时缩放 residual + codeword, 测 Δ_l = d_2 - d_1 margin."""
    model.eval()
    N = embeddings.shape[0]
    L = len(NUM_EMB_LIST)

    bs = 256
    n_batches = (N + bs - 1) // bs

    out = {f"s={s}": {} for s in S_GRID}

    with torch.no_grad():
        for s in S_GRID:
            for li in range(L):
                deltas = []  # 所有样本的 Δ_l
                vq = model.hrq.vq_layers[li]
                cb_raw = vq.embeddings.weight  # (K_l, e_dim)
                # 缩放 codeword: T_s(e) (切到 ball 后再径向缩放, 跟实际量化在 ball 空间一致)
                cb_h = proj_to_ball(expmap0(cb_raw, vq.c), vq.c)
                cb_scaled = radial_rescale_ball(cb_h, s, vq.c)  # (K_l, e_dim) in ball
                # 缩放后转回 tangent (跟 model forward 保持一致, 因为 model 在 tangent 加 x_q)
                # 注: gate1 只测 Δ, 不需要跟 forward 完全一致, 直接在 ball 算距离
                # 但为了跟真实量化行为一致, 还是按 ball 距离

                for bi in range(n_batches):
                    x = embeddings[bi * bs : (bi + 1) * bs].to(DEVICE)
                    # 取该层 residual: x - sum_{k<li} x_q^k
                    residual = x.clone()
                    for k in range(li):
                        # 前向算 k 层 x_q
                        vq_k = model.hrq.vq_layers[k]
                        lat = residual
                        cb_k_raw = vq_k.embeddings.weight
                        cb_k_h = proj_to_ball(expmap0(cb_k_raw, vq_k.c), vq_k.c)
                        cb_k_scaled = radial_rescale_ball(cb_k_h, s, vq_k.c)
                        # 缩放后的 residual + 缩放后的 codebook 测 margin (Issue #164 spec)
                        # 但 Gate 1 spec 明确: 同时变换 residual + codeword (不是仅 codeword)
                        lat_h = proj_to_ball(expmap0(lat, vq_k.c), vq_k.c)
                        lat_scaled = radial_rescale_ball(lat_h, s, vq_k.c)
                        B = lat_scaled.shape[0]
                        K = cb_k_scaled.shape[0]
                        lat_exp = lat_scaled.unsqueeze(1).expand(B, K, -1)
                        cb_exp = cb_k_scaled.unsqueeze(0).expand(B, K, -1)
                        d = poincare_distance(lat_exp, cb_exp, vq_k.c).squeeze(-1)
                        idx = torch.argmin(d, dim=-1)
                        x_q_k = cb_k_raw[idx]
                        residual = residual - x_q_k
                    # 现在 residual 是 li 层 residual, 缩放
                    lat = residual
                    lat_h = proj_to_ball(expmap0(lat, vq.c), vq.c)
                    lat_scaled = radial_rescale_ball(lat_h, s, vq.c)
                    # 算距离 (B, K_l)
                    B = lat_scaled.shape[0]
                    K = cb_scaled.shape[0]
                    lat_exp = lat_scaled.unsqueeze(1).expand(B, K, -1)
                    cb_exp = cb_scaled.unsqueeze(0).expand(B, K, -1)
                    d = poincare_distance(lat_exp, cb_exp, vq.c).squeeze(-1)  # (B, K)
                    # Δ = d_2 - d_1 (最近 - 第二近)
                    d_sorted, _ = torch.sort(d, dim=-1)
                    delta = (d_sorted[:, 1] - d_sorted[:, 0]).cpu().numpy()
                    deltas.append(delta)

                all_deltas = np.concatenate(deltas)
                out[f"s={s}"][f"layer_{li}"] = {
                    "K": NUM_EMB_LIST[li],
                    "n_samples": int(all_deltas.shape[0]),
                    "margin_mean": float(np.mean(all_deltas)),
                    "margin_p5": float(np.percentile(all_deltas, 5)),
                    "margin_p50": float(np.percentile(all_deltas, 50)),
                    "margin_p95": float(np.percentile(all_deltas, 95)),
                    "small_margin_ratio": float(np.mean(all_deltas < TAU_SMALL_MARGIN)),
                }
    return out


# ============================================================
# Gate 2: 扰动 SID 翻转率
# ============================================================
def gate2_fliprate(model: HRQVAE, embeddings: torch.Tensor):
    """扰动 σ ∈ SIGMA_GRID, 测每个 (s, σ, layer) 组合的 SID FlipRate."""
    model.eval()
    N = embeddings.shape[0]
    L = len(NUM_EMB_LIST)

    bs = 256
    n_batches = (N + bs - 1) // bs

    out = {f"s={s}": {} for s in S_GRID}

    for s in S_GRID:
        for sigma in SIGMA_GRID:
            for li in range(L):
                total = 0
                flips = 0
                vq = model.hrq.vq_layers[li]
                cb_raw = vq.embeddings.weight
                cb_h = proj_to_ball(expmap0(cb_raw, vq.c), vq.c)
                cb_scaled = radial_rescale_ball(cb_h, s, vq.c)
                # 注: 扰动在 tangent 空间 (跟 Issue #164 spec 语义一致: r̃_l = r_l + ε)
                # r_l 在 ball 空间, 加 ε 直接 ε (tangent-space additive perturbation)
                for bi in range(n_batches):
                    x = embeddings[bi * bs : (bi + 1) * bs].to(DEVICE)
                    residual = x.clone()
                    for k in range(li):
                        vq_k = model.hrq.vq_layers[k]
                        lat = residual
                        cb_k_raw = vq_k.embeddings.weight
                        cb_k_h = proj_to_ball(expmap0(cb_k_raw, vq_k.c), vq_k.c)
                        cb_k_scaled = radial_rescale_ball(cb_k_h, s, vq_k.c)
                        lat_h = proj_to_ball(expmap0(lat, vq_k.c), vq_k.c)
                        lat_scaled = radial_rescale_ball(lat_h, s, vq_k.c)
                        B = lat_scaled.shape[0]
                        K = cb_k_scaled.shape[0]
                        lat_exp = lat_scaled.unsqueeze(1).expand(B, K, -1)
                        cb_exp = cb_k_scaled.unsqueeze(0).expand(B, K, -1)
                        d = poincare_distance(lat_exp, cb_exp, vq_k.c).squeeze(-1)
                        idx = torch.argmin(d, dim=-1)
                        x_q_k = cb_k_raw[idx]
                        residual = residual - x_q_k
                    # 该层 residual: 缩放后 + 扰动
                    lat = residual
                    lat_h = proj_to_ball(expmap0(lat, vq.c), vq.c)
                    lat_scaled = radial_rescale_ball(lat_h, s, vq.c)
                    # 原始码字分配
                    B = lat_scaled.shape[0]
                    K = cb_scaled.shape[0]
                    lat_exp = lat_scaled.unsqueeze(1).expand(B, K, -1)
                    cb_exp = cb_scaled.unsqueeze(0).expand(B, K, -1)
                    d_orig = poincare_distance(lat_exp, cb_exp, vq.c).squeeze(-1)
                    c_orig = torch.argmin(d_orig, dim=-1)
                    # 加扰动 (tangent 空间 additive)
                    eps = torch.randn_like(lat_scaled) * sigma
                    lat_perturbed = proj_to_ball(expmap0(lat_scaled + eps, vq.c), vq.c)
                    lat_exp_p = lat_perturbed.unsqueeze(1).expand(B, K, -1)
                    d_perturbed = poincare_distance(lat_exp_p, cb_exp, vq.c).squeeze(-1)
                    c_perturbed = torch.argmin(d_perturbed, dim=-1)
                    flips += (c_orig != c_perturbed).sum().item()
                    total += B
                flip_rate = flips / max(total, 1)
                out[f"s={s}"].setdefault(f"sigma={sigma}", {})[f"layer_{li}"] = {
                    "K": NUM_EMB_LIST[li],
                    "total": total,
                    "flips": flips,
                    "flip_rate": flip_rate,
                }
    return out


# ============================================================
# Gate 3: 欧氏距离对照
# ============================================================
def gate3_euclidean_comparison(model: HRQVAE, embeddings: torch.Tensor):
    """对相同坐标计算 d_E margin + FlipRate, 跟双曲比较."""
    model.eval()
    # 强制 no_grad 上下文 (避免 requires_grad tensor 转 numpy 报错)
    with torch.no_grad():
        N = embeddings.shape[0]
        L = len(NEMB := NUM_EMB_LIST)
        bs = 256
        n_batches = (N + bs - 1) // bs
    
        out_margin = {f"s={s}": {} for s in S_GRID}
        out_flip = {f"s={s}": {} for s in S_GRID}
    
        for s in S_GRID:
            for li in range(L):
                deltas = []
                vq = model.hrq.vq_layers[li]
                cb_raw = vq.embeddings.weight
                cb_h = proj_to_ball(expmap0(cb_raw, vq.c), vq.c)
                cb_scaled = radial_rescale_ball(cb_h, s, vq.c)
                for bi in range(n_batches):
                    x = embeddings[bi * bs : (bi + 1) * bs].to(DEVICE)
                    residual = x.clone()
                    for k in range(li):
                        vq_k = model.hrq.vq_layers[k]
                        lat = residual
                        cb_k_raw = vq_k.embeddings.weight
                        cb_k_h = proj_to_ball(expmap0(cb_k_raw, vq_k.c), vq_k.c)
                        cb_k_scaled = radial_rescale_ball(cb_k_h, s, vq_k.c)
                        lat_h = proj_to_ball(expmap0(lat, vq_k.c), vq_k.c)
                        lat_scaled = radial_rescale_ball(lat_h, s, vq_k.c)
                        B = lat_scaled.shape[0]
                        K = cb_k_scaled.shape[0]
                        lat_exp = lat_scaled.unsqueeze(1).expand(B, K, -1)
                        cb_exp = cb_k_scaled.unsqueeze(0).expand(B, K, -1)
                        d = poincare_distance(lat_exp, cb_exp, vq_k.c).squeeze(-1)
                        idx = torch.argmin(d, dim=-1)
                        x_q_k = cb_k_raw[idx]
                        residual = residual - x_q_k
                    lat = residual
                    lat_h = proj_to_ball(expmap0(lat, vq.c), vq.c)
                    lat_scaled = radial_rescale_ball(lat_h, s, vq.c)
                    # 欧氏距离: 切回 tangent (跟 codebook 比较)
                    lat_t = logmap0(lat_scaled, vq.c)  # tangent
                    cb_t = logmap0(cb_scaled, vq.c)
                    B = lat_t.shape[0]
                    K = cb_t.shape[0]
                    lat_exp = lat_t.unsqueeze(1).expand(B, K, -1)
                    cb_exp = cb_t.unsqueeze(0).expand(B, K, -1)
                    d_E = (lat_exp - cb_exp).norm(dim=-1)  # (B, K) Euclidean
                    d_sorted, _ = torch.sort(d_E, dim=-1)
                    delta = (d_sorted[:, 1] - d_sorted[:, 0]).cpu().numpy()
                    deltas.append(delta)
                all_d = np.concatenate(deltas)
                out_margin[f"s={s}"][f"layer_{li}"] = {
                    "K": NEMB[li],
                    "margin_mean": float(np.mean(all_d)),
                    "margin_p50": float(np.percentile(all_d, 50)),
                    "small_margin_ratio": float(np.mean(all_d < TAU_SMALL_MARGIN)),
                }
    
            # FlipRate (欧氏)
            for sigma in SIGMA_GRID:
                for li in range(L):
                    total = 0
                    flips = 0
                    vq = model.hrq.vq_layers[li]
                    cb_raw = vq.embeddings.weight
                    cb_h = proj_to_ball(expmap0(cb_raw, vq.c), vq.c)
                    cb_scaled = radial_rescale_ball(cb_h, s, vq.c)
                    for bi in range(n_batches):
                        x = embeddings[bi * bs : (bi + 1) * bs].to(DEVICE)
                        residual = x.clone()
                        for k in range(li):
                            vq_k = model.hrq.vq_layers[k]
                            lat = residual
                            cb_k_raw = vq_k.embeddings.weight
                            cb_k_h = proj_to_ball(expmap0(cb_k_raw, vq_k.c), vq_k.c)
                            cb_k_scaled = radial_rescale_ball(cb_k_h, s, vq_k.c)
                            lat_h = proj_to_ball(expmap0(lat, vq_k.c), vq_k.c)
                            lat_scaled = radial_rescale_ball(lat_h, s, vq_k.c)
                            B = lat_scaled.shape[0]
                            K = cb_k_scaled.shape[0]
                            lat_exp = lat_scaled.unsqueeze(1).expand(B, K, -1)
                            cb_exp = cb_k_scaled.unsqueeze(0).expand(B, K, -1)
                            d = poincare_distance(lat_exp, cb_exp, vq_k.c).squeeze(-1)
                            idx = torch.argmin(d, dim=-1)
                            x_q_k = cb_k_raw[idx]
                            residual = residual - x_q_k
                        lat = residual
                        lat_h = proj_to_ball(expmap0(lat, vq.c), vq.c)
                        lat_scaled = radial_rescale_ball(lat_h, s, vq.c)
                        lat_t = logmap0(lat_scaled, vq.c)
                        cb_t = logmap0(cb_scaled, vq.c)
                        B = lat_t.shape[0]
                        K = cb_t.shape[0]
                        lat_exp = lat_t.unsqueeze(1).expand(B, K, -1)
                        cb_exp = cb_t.unsqueeze(0).expand(B, K, -1)
                        d_E = (lat_exp - cb_exp).norm(dim=-1)
                        c_orig = torch.argmin(d_E, dim=-1)
                        eps = torch.randn_like(lat_t) * sigma
                        lat_perturbed_t = lat_t + eps
                        lat_exp_p = lat_perturbed_t.unsqueeze(1).expand(B, K, -1)
                        d_perturbed = (lat_exp_p - cb_exp).norm(dim=-1)
                        c_perturbed = torch.argmin(d_perturbed, dim=-1)
                        flips += (c_orig != c_perturbed).sum().item()
                        total += B
                    out_flip[f"s={s}"].setdefault(f"sigma={sigma}", {})[f"layer_{li}"] = {
                        "K": NEMB[li],
                        "flip_rate": flips / max(total, 1),
                    }
    
        return {"margin_euclidean": out_margin, "fliprate_euclidean": out_flip}


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Task #457 / Issue #164 [方向D Gate 0-3] 零训练径向判别间隔 + SID FlipRate")
    print("=" * 60)

    # ---- 加载 ckpt + 数据 ----
    print(f"\n[1] Loading ckpt: {CKPT_PATH}")
    with open(CKPT_PATH, "rb") as f:
        ckpt_bytes = f.read()
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
    print(f"    SHA256: {ckpt_sha256}")

    print(f"\n[2] Loading item embeddings: {ITEM_EMB_PATH}")
    data = EmbDataset(ITEM_EMB_PATH)
    print(f"    N = {len(data)}, dim = {data.dim}")

    # 构造 model + 加载 ckpt
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=NUM_EMB_LIST,
        e_dim=E_DIM,
        layers=LAYERS,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=BETA,
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=SK_EPSILONS,
        sk_iters=100,
    )
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    # ckpt 嵌套 {args, epoch, best_loss, best_collision_rate, state_dict, optimizer}
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    model.to(DEVICE)
    print(f"    Model loaded: c={model.hrq.vq_layers[0].c}")

    # ---- 加载所有 embed → encoder → latent (用于 Gate 0/1/2/3) ----
    print(f"\n[3] Encoding all items → tangent latent (e_dim={E_DIM})")
    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = model.encoder(batch.to(DEVICE))  # (B, e_dim)
            latents.append(x.cpu())
    embeddings = torch.cat(latents, dim=0)  # (N, e_dim)
    print(f"    Embeddings: {embeddings.shape}")

    # ---- Gate 0 ----
    print(f"\n[4] Gate 0: 径向对象 profile (三层 × 三对象 × 七统计量)")
    g0 = gate0_radial_profile(model, embeddings)
    print(json.dumps(g0, indent=2, ensure_ascii=False))

    # ---- Gate 1 ----
    print(f"\n[5] Gate 1: 径向缩放 s in {S_GRID}, margin profile")
    g1 = gate1_radial_margin(model, embeddings)
    print(json.dumps(g1, indent=2, ensure_ascii=False))

    # ---- Gate 2 ----
    print(f"\n[6] Gate 2: 扰动 σ in {SIGMA_GRID}, FlipRate profile")
    g2 = gate2_fliprate(model, embeddings)
    print(json.dumps(g2, indent=2, ensure_ascii=False))

    # ---- Gate 3 ----
    print(f"\n[7] Gate 3: 欧氏距离对照 (margin + FlipRate)")
    g3 = gate3_euclidean_comparison(model, embeddings)
    print(json.dumps(g3, indent=2, ensure_ascii=False))

    # ---- 落盘 ----
    out_dir = "/home/wlia0047/ar57/wenyu/GeneRec/products/task457_issue164_gate0_3_radial"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gate0_radial_profile.json"), "w") as f:
        json.dump(g0, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "gate1_radial_margin.json"), "w") as f:
        json.dump(g1, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "gate2_fliprate.json"), "w") as f:
        json.dump(g2, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "gate3_euclidean_comparison.json"), "w") as f:
        json.dump(g3, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "ckpt_sha256.txt"), "w") as f:
        f.write(f"{CKPT_PATH}\n{ckpt_sha256}\n")
    print(f"\n[8] Outputs saved to {out_dir}")

    # ---- 决策建议 (Gate 0-3 GO/NO-GO) ----
    print("\n" + "=" * 60)
    print("Gate 0-3 决策建议 (R11.5 自主判断)")
    print("=" * 60)

    # H1: margin 单调性 (s=1.4 > s=1.0 across layers)
    h1_pass = True
    for li in range(len(NUM_EMB_LIST)):
        m_low = g1[f"s=1.0"][f"layer_{li}"]["margin_mean"]
        m_high = g1[f"s=1.4"][f"layer_{li}"]["margin_mean"]
        if not (m_high > m_low):
            h1_pass = False
            print(f"  H1 Layer {li} FAIL: s=1.4 margin ({m_high:.4f}) <= s=1.0 ({m_low:.4f})")
        else:
            print(f"  H1 Layer {li} PASS: s=1.4 margin {m_high:.4f} > s=1.0 {m_low:.4f}")

    # H2: FlipRate 单调性 (s=1.4 < s=1.0)
    h2_pass = True
    for sigma in SIGMA_GRID:
        for li in range(len(NUM_EMB_LIST)):
            fr_low = g2[f"s=1.0"][f"sigma={sigma}"][f"layer_{li}"]["flip_rate"]
            fr_high = g2[f"s=1.4"][f"sigma={sigma}"][f"layer_{li}"]["flip_rate"]
            if not (fr_high < fr_low):
                h2_pass = False
                print(f"  H2 σ={sigma} Layer {li} FAIL: s=1.4 FlipRate ({fr_high:.4f}) >= s=1.0 ({fr_low:.4f})")
            else:
                print(f"  H2 σ={sigma} Layer {li} PASS: s=1.4 FlipRate {fr_high:.4f} < s=1.0 {fr_low:.4f}")

    # H3: 欧氏对照未现同等改善 (欧氏 margin / FlipRate 不随 s 大幅改善)
    h3_pass = True
    for li in range(len(NUM_EMB_LIST)):
        em_low = g3["margin_euclidean"][f"s=1.0"][f"layer_{li}"]["margin_mean"]
        em_high = g3["margin_euclidean"][f"s=1.4"][f"layer_{li}"]["margin_mean"]
        # 如果欧氏也显著改善 (>5%), 则不是双曲特有效应
        if em_high > em_low * 1.05:
            h3_pass = False
            print(f"  H3 Layer {li} FAIL: Euclidean margin also improves {em_low:.4f} → {em_high:.4f}")
        else:
            print(f"  H3 Layer {li} PASS: Euclidean margin stable {em_low:.4f} → {em_high:.4f}")

    decision = "GO" if (h1_pass and h2_pass and h3_pass) else "NO-GO"
    print(f"\n>>> Gate 0-3 整体决策: {decision}")
    print(f"    H1 (margin 单调) = {'PASS' if h1_pass else 'FAIL'}")
    print(f"    H2 (FlipRate 下降) = {'PASS' if h2_pass else 'FAIL'}")
    print(f"    H3 (欧氏对照无改善) = {'PASS' if h3_pass else 'FAIL'}")
    if decision == "GO":
        print("    → Gate 4 (Arm A/B/C/D 训练级验证) 推荐启动")
    else:
        print("    → Gate 4 不启动, 假设不支持, NO-GO 收口")

    summary = {
        "ckpt_sha256": ckpt_sha256,
        "ckpt_path": CKPT_PATH,
        "N_items": len(data),
        "embeddings_shape": list(embeddings.shape),
        "S_grid": S_GRID,
        "sigma_grid": SIGMA_GRID,
        "tau_small_margin": TAU_SMALL_MARGIN,
        "H1_margin_monotonic": h1_pass,
        "H2_fliprate_decrease": h2_pass,
        "H3_euclidean_no_improvement": h3_pass,
        "decision": decision,
        "gate4_recommendation": "Arm A/B/C/D 200 epoch Stage 1 + Sinkhorn + Stage 3 + Stage 4" if decision == "GO" else "NO-GO 收口",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {json.dumps(summary, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()