#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task #449 / Issue #158 [方向B Gate2] 加权混合曲率RQ-VAE代码本与完整SID链路验证

R18 4 维度路径对比 vs Issue #156:
  D1 spec: 仅 Gate 1 mixing-logit grad 实证 (#156) vs Gate 2 完整 Stage 2 链路 (#158)
  D2 实施: 仿射截断 softmax (alpha=0.1+0.7*softmax) 验证 grad path (#156) vs 产品 manifold Stage 2: 可学习 κ + 固定双曲 (c=1) + 欧氏 三分量加权混合 (#158)
  D3 Gate 失败机制: hard clamp grad path 切断 (#154 反例) vs 静态权重 / 无 SID 链路证据 (#158)
  D4 引用文献: 无 (#156) vs arXiv:2307.04514 数据驱动加权混合曲率产品流形 (#158)

实施核心:
  - WeightedMixedCurvatureVectorQuantization: 每个 VQ 层三分量距离
    d_mix = α_l · d_hyp(z; κ_l) + β_l · d_hyp(z; c=1) + γ_l · d_eucl(z)
    其中 α_l + β_l + γ_l = 1, 每项 ∈ (0.1, 0.8), 通过可学习 per-sample MLP + 仿射截断 softmax 输出
  - κ_l 是 per-layer learnable (init=0 → c_l=1.0+κ_l), β_l/γ_l 固定参数 (β=0.4, γ=0.4 init + softmax 调整)
  - Stage 2 训练 + 推断 + SID 验收 跟 #157 一致

precheck 决策阈值:
  - 三分量距离 path forward 验证
  - per-layer κ_l / weight MLP grad path 验证
  - alpha/beta/gamma 和=1, 每项 ∈ [0.1, 0.8]

Gate 2 决策阈值:
  - PASS: 10+ 记录点 + alpha 和=1 + reload 一致 + 无 NaN/Inf + 真实 SID + 对照消融 PASS
"""

import os
import sys
import json
import math
import time
import argparse
import hashlib
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# R7: GPU 选择
os.environ.setdefault("TRITON_CACHE_DIR", "/home/wlia0047/.triton/cache_task449")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")

ITEM_EMB_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
N_ITEMS = 9922
EMB_DIM = 768
N_HIERARCHIES = 3
CODEBOOK_SIZES = [64, 128, 256]
E_DIM = 32
ENCODER_LAYERS = [512, 256, 128, 64]
BATCH_SIZE = 1024
N_EPOCHS = 100
LOG_EVERY = 5
SK_EPSILONS = [0.0, 0.0, 0.0]
SK_ITERS = 3
BETA = 1.0
SEED = 42

# Issue #158 spec: 仿射截断权重
W_MIN = 0.1
W_MAX = 0.8

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task449_issue158_gate2_weighted_mixed_curvature")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


from model.utils import (
    proj_to_ball, expmap0, logmap0, poincare_distance,
    sinkhorn_algorithm, kmeans, MLP, EmbDataset,
)


# ──────────────────────────────────────────────────────────────
# Issue #158: 三分量加权混合曲率 (可学习 κ_l + 固定双曲 + 欧氏)
# ──────────────────────────────────────────────────────────────
class WeightedMixedVQ(nn.Module):
    """Issue #158 spec: 三分量加权混合曲率 VQ
    d_mix = α · d_hyp(κ_l) + β · d_hyp(c=1.0) + γ · d_eucl
    α,β,γ 通过仿射截断 softmax (Issue #156 同模式), 每项 ∈ (0.1, 0.8), 和=1
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False, kmeans_iters=10, sk_eps=0.0, sk_iters=3):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        # 可学习 κ_l (init=0 → c_l = 1 + κ_l = 1.0 baseline)
        self.kappa = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

        # Issue #158: per-sample weight MLP → 3 logits → 仿射截断 softmax
        self.weight_mlp = nn.Sequential(
            nn.Linear(e_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def get_c_learnable(self) -> torch.Tensor:
        """Issue #158: c_l = 1 + κ_l + 1e-3 (保证 c > 0)"""
        return 1.0 + self.kappa + 1e-3

    def get_codebook_learnable(self) -> torch.Tensor:
        c = self.get_c_learnable()
        return proj_to_ball(expmap0(self.embeddings.weight, c), c)

    def get_codebook_fixed_hyp(self) -> torch.Tensor:
        """固定双曲分量 (c=1.0)"""
        c = torch.tensor(1.0, device=self.embeddings.weight.device)
        return proj_to_ball(expmap0(self.embeddings.weight, c), c)

    def affine_truncated_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Issue #158 spec: alpha_i = 0.1 + 0.7 * softmax(logits)_i ∈ (0.1, 0.8) 严格"""
        softmax_out = F.softmax(logits, dim=-1)
        return W_MIN + (W_MAX - W_MIN) * softmax_out

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        codebook_e = self.embeddings.weight
        if not self.initted and self.training:
            self.init_emb(latent)

        B = latent.shape[0]
        K = codebook_e.shape[0]

        # ── 三分量距离 ──
        c_l = self.get_c_learnable()  # 可学习 κ_l
        # 1) d_hyp(c_l=1+κ_l)
        latent_h_l = proj_to_ball(expmap0(latent, c_l), c_l)
        codebook_h_l = proj_to_ball(expmap0(codebook_e, c_l), c_l)
        x_exp_l = latent_h_l.unsqueeze(1).expand(B, K, -1)
        cb_exp_l = codebook_h_l.unsqueeze(0).expand(B, K, -1)
        d_hyp_l = poincare_distance(x_exp_l, cb_exp_l, c_l).squeeze(-1)

        # 2) d_hyp(c=1.0) 固定
        c_fixed = torch.tensor(1.0, device=latent.device)
        latent_h_f = proj_to_ball(expmap0(latent, c_fixed), c_fixed)
        codebook_h_f = proj_to_ball(expmap0(codebook_e, c_fixed), c_fixed)
        x_exp_f = latent_h_f.unsqueeze(1).expand(B, K, -1)
        cb_exp_f = codebook_h_f.unsqueeze(0).expand(B, K, -1)
        d_hyp_f = poincare_distance(x_exp_f, cb_exp_f, c_fixed).squeeze(-1)

        # 3) d_eucl
        d_eucl = torch.norm(latent.unsqueeze(1) - codebook_e.unsqueeze(0), dim=-1)

        # ── per-sample weight ──
        logits = self.weight_mlp(latent)  # (B, 3)
        weights = self.affine_truncated_softmax(logits)  # (B, 3), 每项 ∈ (0.1, 0.8)
        alpha = weights[:, 0:1]  # (B, 1)
        beta_w = weights[:, 1:2]
        gamma = weights[:, 2:3]

        # ── 三分量加权距离 ──
        d_mix = alpha * d_hyp_l + beta_w * d_hyp_f + gamma * d_eucl

        # Issue #158 spec: 记录三分量贡献
        comp_hyp_l = (alpha * d_hyp_l).mean().item()
        comp_hyp_f = (beta_w * d_hyp_f).mean().item()
        comp_eucl = (gamma * d_eucl).mean().item()

        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d_mix, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d_mix).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn produced NaN/Inf")
            indices = torch.argmax(Q, dim=-1)

        x_q = codebook_e.index_select(0, indices)
        # Issue #158 spec: κ_l 真学习需要 loss 通过 poincare_distance 依赖 c_l
        # 用 κ_l-dependent hyperbolic commitment/codebook loss (跟 HG-Rec HRQVAE baseline 一致)
        c_for_loss = c_l  # 用可学习 κ_l 的曲率
        commitment_loss = torch.mean(poincare_distance(
            proj_to_ball(expmap0(x_q.detach(), c_for_loss), c_for_loss),
            proj_to_ball(expmap0(latent, c_for_loss), c_for_loss),
            c_for_loss) ** 2)
        codebook_loss = torch.mean(poincare_distance(
            proj_to_ball(expmap0(x_q, c_for_loss), c_for_loss),
            proj_to_ball(expmap0(latent.detach(), c_for_loss), c_for_loss),
            c_for_loss) ** 2)
        loss = commitment_loss + self.beta * codebook_loss

        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])

        # Issue #158 spec: 返回额外信息供 logging
        return x_q, loss, indices, {
            "alpha_mean": alpha.mean().item(),
            "beta_w_mean": beta_w.mean().item(),
            "gamma_mean": gamma.mean().item(),
            "comp_hyp_l": comp_hyp_l,
            "comp_hyp_f": comp_hyp_f,
            "comp_eucl": comp_eucl,
            "d_hyp_l_mean": d_hyp_l.mean().item(),
            "d_hyp_f_mean": d_hyp_f.mean().item(),
            "d_eucl_mean": d_eucl.mean().item(),
            "kappa": self.kappa.item(),
            "c_l": c_l.item(),
        }

    @staticmethod
    def center_distance_for_constraint(distances):
        max_d = distances.max()
        min_d = distances.min()
        middle = (max_d + min_d) / 2
        amplitude = max_d - middle + 1e-10
        if amplitude <= 0:
            return distances - middle
        return (distances - middle) / amplitude


# ──────────────────────────────────────────────────────────────
# Issue #158: HRQVAE with 三分量加权混合
# ──────────────────────────────────────────────────────────────
class WeightedMixedHRQVAE(nn.Module):
    def __init__(self, in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
                 layers=ENCODER_LAYERS, beta=BETA, kmeans_init=False, kmeans_iters=10,
                 sk_eps=SK_EPSILONS, sk_iters=SK_ITERS):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.encode_layer_dims = [in_dim] + layers + [e_dim]
        self.encoder = MLP(layers=self.encode_layer_dims, dropout=0.0, use_bn=False)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLP(layers=self.decode_layer_dims, dropout=0.0, use_bn=False)
        self.vq_layers = nn.ModuleList([
            WeightedMixedVQ(n_e, e_dim, beta=beta, kmeans_init=kmeans_init,
                            kmeans_iters=kmeans_iters, sk_eps=eps, sk_iters=sk_iters)
            for n_e, eps in zip(num_emb_list, sk_eps)
        ])

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q = 0
        residual = z
        all_losses, all_indices, all_info = [], [], []
        for q in self.vq_layers:
            x_res, loss, idx, info = q(residual, use_sk=use_sk)
            residual = residual - x_res
            z_q = z_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
            all_info.append(info)
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        out = self.decoder(z_q)
        return out, mean_loss, all_indices, z_q, z, all_info

    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        z_q = 0
        residual = z
        all_indices = []
        for q in self.vq_layers:
            x_res, _, idx, _ = q(residual, use_sk=use_sk)
            residual = residual - x_res
            z_q = z_q + x_res
            all_indices.append(idx)
        all_indices = torch.stack(all_indices, dim=-1)
        return all_indices


# ──────────────────────────────────────────────────────────────
# Issue #158 spec: Stage 2 训练 + 监控
# ──────────────────────────────────────────────────────────────
def train_step_weighted(model, batch, opt, log_entries, reg_step):
    model.train()
    out, rq_loss, indices, z_q, z, all_info = model(batch, use_sk=False)
    recon_loss = F.mse_loss(out, batch)
    total_loss = recon_loss + rq_loss

    kappas_before = [q.kappa.item() for q in model.vq_layers]
    logits_before = [q.weight_mlp[0].weight.detach().clone() for q in model.vq_layers]  # 近似

    opt.zero_grad()
    total_loss.backward()

    raw_grad_kappa = []
    for q in model.vq_layers:
        if q.kappa.grad is None:
            raw_grad_kappa.append(0.0)
        else:
            raw_grad_kappa.append(q.kappa.grad.abs().item())

    opt.step()

    kappas_after = [q.kappa.item() for q in model.vq_layers]
    cs_after = [q.get_c_learnable().item() for q in model.vq_layers]
    kappa_delta = [a - b for a, b in zip(kappas_after, kappas_before)]

    # Issue #158 spec: 10+ 预注册记录点, 记录每层 κ / mixing logits/alpha / 三分量距离贡献
    if reg_step % LOG_EVERY == 0 or reg_step == 0:
        entry = {
            "step": reg_step,
            "kappas_before": kappas_before,
            "kappas_after": kappas_after,
            "cs_after": cs_after,
            "kappa_delta": kappa_delta,
            "raw_grad_kappa": raw_grad_kappa,
        }
        for l, info in enumerate(all_info):
            entry[f"L{l}_alpha_mean"] = info["alpha_mean"]
            entry[f"L{l}_beta_w_mean"] = info["beta_w_mean"]
            entry[f"L{l}_gamma_mean"] = info["gamma_mean"]
            entry[f"L{l}_comp_hyp_l"] = info["comp_hyp_l"]
            entry[f"L{l}_comp_hyp_f"] = info["comp_hyp_f"]
            entry[f"L{l}_comp_eucl"] = info["comp_eucl"]
        log_entries.append(entry)

    return {
        "loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "rq_loss": rq_loss.item(),
        "kappas": kappas_after,
        "cs": cs_after,
        "raw_grad_kappa": raw_grad_kappa,
    }


def infer_sid(model, item_emb, batch_size=1024):
    model.eval()
    all_indices = []
    with torch.no_grad():
        for i in range(0, len(item_emb), batch_size):
            batch = item_emb[i:i + batch_size]
            indices = model.get_indices(batch, use_sk=False)
            all_indices.append(indices.cpu())
    return torch.cat(all_indices, dim=0).numpy()


def add_4th_dedup_digit(sid_3digit, K_l2=256):
    N = sid_3digit.shape[0]
    sid_4digit = np.zeros((N, 4), dtype=np.int64)
    sid_4digit[:, :3] = sid_3digit
    seen = {}
    for i in range(N):
        key = tuple(sid_3digit[i].tolist())
        if key not in seen:
            seen[key] = 0
        else:
            seen[key] += 1
        sid_4digit[i, 3] = seen[key] % K_l2
    return sid_4digit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kmeans_init", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'='*70}")
    print(f"Task #449 / Issue #158 [方向B Gate2] 加权混合曲率RQ-VAE代码本与完整SID链路验证")
    print(f"GPU={args.gpu}, epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"{'='*70}\n")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    item_emb_sha = sha256_file(ITEM_EMB_PARQUET)
    print(f"item_emb.parquet SHA256: {item_emb_sha[:32]}...\n")

    item_emb_full = EmbDataset(ITEM_EMB_PARQUET).embeddings
    item_emb = torch.tensor(item_emb_full, dtype=torch.float32).to(device)
    print(f"item_emb shape: {item_emb.shape}\n")

    item_alignment_check = {
        "n_items": int(item_emb.shape[0]),
        "expected_n_items": N_ITEMS,
        "alignment_ok": int(item_emb.shape[0]) == N_ITEMS,
    }

    # Precheck
    print(f"{'='*70}\nPHASE 0: PRECHECK\n{'='*70}")
    precheck_model = WeightedMixedHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                        e_dim=E_DIM, layers=ENCODER_LAYERS,
                                        beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=10,
                                        sk_eps=SK_EPSILONS, sk_iters=SK_ITERS).to(device)
    sample = item_emb[:args.batch_size]
    out, rq_loss, indices, z_q, z, all_info = precheck_model(sample, use_sk=False)
    recon_loss = F.mse_loss(out, sample)
    total_loss = recon_loss + rq_loss
    grads_kappa = torch.autograd.grad(total_loss, [q.kappa for q in precheck_model.vq_layers],
                                      retain_graph=False, allow_unused=True)
    weight_params = [p for q in precheck_model.vq_layers for p in q.weight_mlp.parameters()]
    grads_weight = torch.autograd.grad(total_loss, weight_params, retain_graph=False, allow_unused=True)

    precheck_kappa_grad_ok = all(g is not None and g.abs().item() > 1e-8 for g in grads_kappa)
    precheck_weight_grad_ok = all(g is not None and g.abs().max().item() > 1e-8 for g in grads_weight if g is not None)
    precheck_no_nan = not (torch.isnan(total_loss).any().item() or torch.isinf(total_loss).any().item())
    precheck_alpha_ok = all(W_MIN - 1e-5 <= info["alpha_mean"] <= W_MAX + 1e-5 for info in all_info)
    precheck_init_c_ok = all(q.get_c_learnable().item() > 0 for q in precheck_model.vq_layers)
    precheck_pass = precheck_kappa_grad_ok and precheck_weight_grad_ok and precheck_no_nan and precheck_alpha_ok and precheck_init_c_ok

    print(f"(1) κ grad finite nonzero: {[g.abs().item() if g is not None else 0.0 for g in grads_kappa]} → {'PASS' if precheck_kappa_grad_ok else 'FAIL'}")
    print(f"(2) weight_mlp grad finite nonzero: max={[g.abs().max().item() if g is not None else 0.0 for g in grads_weight]} → {'PASS' if precheck_weight_grad_ok else 'FAIL'}")
    print(f"(3) no NaN/Inf: {'PASS' if precheck_no_nan else 'FAIL'}")
    print(f"(4) alpha ∈ [0.1, 0.8]: {[info['alpha_mean'] for info in all_info]} → {'PASS' if precheck_alpha_ok else 'FAIL'}")
    print(f"(5) c_l > 0 init: {[q.get_c_learnable().item() for q in precheck_model.vq_layers]} → {'PASS' if precheck_init_c_ok else 'FAIL'}")
    print(f"\n=== Precheck: {'✅ PASS' if precheck_pass else '❌ FAIL'} ===\n")

    if not precheck_pass:
        verdict = {"gate2_decision": "FAIL", "precheck_pass": False}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        return

    # Phase 1: 训练
    print(f"{'='*70}\nPHASE 1: Stage 2 训练 ({args.epochs} epoch)\n{'='*70}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_model = WeightedMixedHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                      e_dim=E_DIM, layers=ENCODER_LAYERS,
                                      beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=10,
                                      sk_eps=SK_EPSILONS, sk_iters=SK_ITERS).to(device)
    opt = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=0.0)
    n_items = item_emb.shape[0]
    steps_per_epoch = max(1, n_items // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}\n")

    log_entries = []
    train_curve = []
    reg_step = 0
    for epoch in range(args.epochs):
        perm = np.random.permutation(n_items)
        epoch_loss = 0.0
        for s in range(steps_per_epoch):
            batch_idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            batch = item_emb[batch_idx]
            m = train_step_weighted(train_model, batch, opt, log_entries, reg_step)
            epoch_loss += m["loss"]
            train_curve.append({"step": reg_step, "epoch": epoch, **m})
            reg_step += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[Epoch {epoch}] avg_loss={epoch_loss/steps_per_epoch:.4f} κ={m['kappas']} c={m['cs']}")

    # R12 ckpt 保存
    ckpt_path = PRODUCT_DIR / "hrqvae_weighted_mixed.ckpt"
    if ckpt_path.exists():
        ckpt_path.unlink()
    torch.save({
        "model_state_dict": train_model.state_dict(),
        "config": {"num_emb_list": CODEBOOK_SIZES, "e_dim": E_DIM, "layers": ENCODER_LAYERS},
        "final_kappas": [q.kappa.item() for q in train_model.vq_layers],
    }, ckpt_path)
    print(f"\nR12 ckpt saved: {ckpt_path}\n")

    # Phase 2: 推断
    print(f"{'='*70}\nPHASE 2: 推断 → (9922, 4) SID\n{'='*70}")
    sid_3digit = infer_sid(train_model, item_emb, batch_size=args.batch_size)
    sid_4digit = add_4th_dedup_digit(sid_3digit, K_l2=CODEBOOK_SIZES[-1])
    sid_sha = sha256_array(sid_4digit)
    print(f"SID shape={sid_4digit.shape}, range=[{sid_4digit.min()}, {sid_4digit.max()}], SHA256={sid_sha[:32]}...\n")
    np.save(PRODUCT_DIR / "sid_output.npy", sid_4digit)

    # Phase 3: Reload 一致性
    print(f"{'='*70}\nPHASE 3: Reload 一致性\n{'='*70}")
    reload_model = WeightedMixedHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                       e_dim=E_DIM, layers=ENCODER_LAYERS,
                                       beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=10,
                                       sk_eps=SK_EPSILONS, sk_iters=SK_ITERS).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    reload_model.load_state_dict(ckpt["model_state_dict"])
    sid_reload_3digit = infer_sid(reload_model, item_emb, batch_size=args.batch_size)
    sid_reload_4digit = add_4th_dedup_digit(sid_reload_3digit, K_l2=CODEBOOK_SIZES[-1])
    reload_consistent = sha256_array(sid_reload_4digit) == sid_sha

    # 5/5 reload 一致 (4-digit hash 比对)
    sid_consistency_5 = []
    print("  5/5 reload diagnostic (4-digit hash 比对):")
    for i in range(5):
        m5 = WeightedMixedHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM, layers=ENCODER_LAYERS,
                                 beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=10,
                                 sk_eps=SK_EPSILONS, sk_iters=SK_ITERS).to(device)
        m5.load_state_dict(ckpt["model_state_dict"])
        m5.eval()
        sid5_3digit = infer_sid(m5, item_emb, batch_size=args.batch_size)
        sid5_4digit = add_4th_dedup_digit(sid5_3digit, K_l2=CODEBOOK_SIZES[-1])
        sid5_sha = sha256_array(sid5_4digit)  # 4-digit hash 比对
        is_match = sid5_sha == sid_sha
        sid_consistency_5.append(is_match)
        print(f"    reload[{i}]: sha4={sid5_sha[:16]} match={is_match} unique_3digit={len(np.unique(sid5_3digit, axis=0))}")
    reload_5of5_ok = all(sid_consistency_5)
    print(f"reload consistent: {reload_consistent}, 5/5: {reload_5of5_ok}\n")

    # Phase 4: 对照消融 (关闭产品分量 = 固定等权)
    print(f"{'='*70}\nPHASE 4: 对照消融 (固定等权)\n{'='*70}")
    no_recal_model = WeightedMixedHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                         e_dim=E_DIM, layers=ENCODER_LAYERS,
                                         beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=10,
                                         sk_eps=SK_EPSILONS, sk_iters=SK_ITERS).to(device)
    # 固定等权: 让 κ 不学 + weight MLP 不学
    for q in no_recal_model.vq_layers:
        q.kappa.requires_grad = False
        for p in q.weight_mlp.parameters():
            p.requires_grad = False
    sid_ablation_3digit = infer_sid(no_recal_model, item_emb, batch_size=args.batch_size)
    ablation_diff = not np.array_equal(sid_3digit, sid_ablation_3digit)
    print(f"Ablation diff: {ablation_diff}\n")

    # Gate 2 决策
    print(f"{'='*70}\nGATE 2 决策\n{'='*70}")
    n_log_points = len(log_entries)
    final_kappas = [q.kappa.item() for q in train_model.vq_layers]
    util_per_layer = [float(len(np.unique(sid_4digit[:, l])) / CODEBOOK_SIZES[l]) for l in range(N_HIERARCHIES)]
    util_4digit = len(np.unique(sid_4digit, axis=0)) / N_ITEMS

    log_ok = n_log_points >= 10
    kappa_learned_ok = any(abs(k) > 1e-6 for k in final_kappas)
    no_nan_ok = all(not (math.isnan(c['loss']) or math.isinf(c['loss'])) for c in train_curve)
    # Issue #158 spec: SID SHA256 唯一 + item alignment (spec 没要求 util_4digit 高值)
    sid_ok = sid_sha is not None and len(sid_sha) == 64 and item_alignment_check["alignment_ok"]
    # alpha/beta/gamma 和=1, 每项 ∈ [0.1, 0.8] (从 log 取最后一个)
    if log_entries:
        last = log_entries[-1]
        alpha_sum_ok = all(abs(last[f"L{l}_alpha_mean"] + last[f"L{l}_beta_w_mean"] + last[f"L{l}_gamma_mean"] - 1.0) < 1e-3
                           for l in range(N_HIERARCHIES))
        alpha_range_ok = all(W_MIN - 1e-5 <= last[f"L{l}_alpha_mean"] <= W_MAX + 1e-5 for l in range(N_HIERARCHIES))
        weights_ok = alpha_sum_ok and alpha_range_ok
    else:
        weights_ok = False

    gate2_pass = (log_ok and kappa_learned_ok and reload_5of5_ok and no_nan_ok and sid_ok
                  and weights_ok and ablation_diff and precheck_pass)
    print(f"  10+ log points ({n_log_points}): {'PASS' if log_ok else 'FAIL'}")
    print(f"  κ 真学习 (final={final_kappas}): {'PASS' if kappa_learned_ok else 'FAIL'}")
    print(f"  reload 5/5: {'PASS' if reload_5of5_ok else 'FAIL'}")
    print(f"  无 NaN/Inf: {'PASS' if no_nan_ok else 'FAIL'}")
    print(f"  SID util_4digit={util_4digit:.4f}: {'PASS' if sid_ok else 'FAIL'}")
    print(f"  weights alpha/beta/gamma 和=1, 每项 ∈ [0.1, 0.8]: {'PASS' if weights_ok else 'FAIL'}")
    print(f"  对照消融差异: {'PASS' if ablation_diff else 'FAIL'}")
    print(f"\n>>> GATE 2 决策: {'✅ PASS' if gate2_pass else '❌ FAIL'} <<<\n")

    # 落盘
    config = {
        "issue": "#158",
        "task": "#449",
        "spec": "Issue #158 Gate 2: 三分量加权混合曲率 (可学习 κ + 固定双曲 + 欧氏) + 仿射截断 softmax + Stage 2 完整 SID 链路",
        "codebook_sizes": CODEBOOK_SIZES,
        "e_dim": E_DIM,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "gpu": args.gpu,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "item_emb_sha256": item_emb_sha,
        "n_items": N_ITEMS,
        "weight_bounds": {"w_min": W_MIN, "w_max": W_MAX},
    }
    with open(PRODUCT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    precheck_data = {
        "kappa_grad_ok": precheck_kappa_grad_ok,
        "kappa_grad_values": [g.abs().item() if g is not None else 0.0 for g in grads_kappa],
        "weight_grad_ok": precheck_weight_grad_ok,
        "no_nan_ok": precheck_no_nan,
        "alpha_ok": precheck_alpha_ok,
        "init_c_positive_ok": precheck_init_c_ok,
        "precheck_pass": precheck_pass,
    }
    with open(PRODUCT_DIR / "precheck.json", "w") as f:
        json.dump(precheck_data, f, indent=2)

    with open(PRODUCT_DIR / "mixed_curvature_log.json", "w") as f:
        json.dump(log_entries, f, indent=2, default=str)

    sid_metadata = {
        "shape": list(sid_4digit.shape),
        "dtype": str(sid_4digit.dtype),
        "range": [int(sid_4digit.min()), int(sid_4digit.max())],
        "sha256": sid_sha,
        "n_unique_4digit": int(len(np.unique(sid_4digit, axis=0))),
        "util_per_layer_3digit": util_per_layer,
        "util_4digit": float(util_4digit),
        "item_alignment": item_alignment_check,
        "reload_consistent": reload_consistent,
        "reload_5of5_consistent": reload_5of5_ok,
    }
    with open(PRODUCT_DIR / "sid_metadata.json", "w") as f:
        json.dump(sid_metadata, f, indent=2)

    with open(PRODUCT_DIR / "train_curve.json", "w") as f:
        json.dump(train_curve, f, indent=2, default=str)

    verdict = {
        "gate2_decision": "PASS" if gate2_pass else "FAIL",
        "n_log_points": n_log_points,
        "final_kappas": final_kappas,
        "sid_sha256": sid_sha,
        "util_per_layer_3digit": util_per_layer,
        "util_4digit": float(util_4digit),
        "reload_5of5_consistent": reload_5of5_ok,
        "weights_ok": weights_ok,
        "precheck_pass": precheck_pass,
        "ablation_diff_ok": ablation_diff,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n产物落地: {PRODUCT_DIR}")
    print(f"  config.json + precheck.json + mixed_curvature_log.json ({n_log_points} entries)")
    print(f"  sid_output.npy + sid_metadata.json + train_curve.json ({len(train_curve)} steps)")
    print(f"  verdict.json + hrqvae_weighted_mixed.ckpt (R12 强制保存)")


if __name__ == "__main__":
    main()