#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #298 / Issue #28 — train_hrqvae_gumbel.py (Stage 1 训练 wrapper)

背景 (Issue #28 §Gate 0):
  实现 `train_hrqvae_gumbel.py` 继承 baseline, 新增:
    - per-layer τ_l = [1.0, 0.5, 0.1] (per-layer 异构温度)
    - per-layer c_k range = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (沿用 task242 Arm A)
    - per-layer Gumbel-Softmax soft-assign 替换 baseline argmin hard-assign
    - Stage 1 训练时 soft-assign (训练目标: soft probability × hyperbolic distance)
    - Stage 2 推断时 hard argmin (保持 baseline SID 解码路径)

实施:
  - 不修改 HG-Rec/model/utils.py / hrqvae.py (R11.4 critical decision, 不动上游)
  - 复用 baseline HRQVAE 训练器 (HG-Rec/model/hrqvae_trainer.py:Trainer)
  - 用 monkey-patch 把 baseline HResidualVectorQuantization.forward 替换成
    GumbelSoftmaxQuantization.forward (per-layer τ_l, per-layer c_k range)
  - Stage 2 / Sinkhorn 推断 仍用 baseline 硬分配 (Task #298 §Gate 0 明确)
"""
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')
sys.path.insert(0, REPO)

from model.hrqvae import HRQVAE
from model.hrqvae_trainer import Trainer
from model.utils import EmbDataset, poincare_distance


# ============================================================================
# Gumbel-Softmax Soft-Assign + Per-Layer c_k Range
# ============================================================================

def _ensure_per_layer_c_k(codebook: torch.Tensor, layer_idx: int,
                          c_k_range: Tuple[float, float], seed: int) -> torch.Tensor:
    """Task #242 + Issue #28: per-layer c_k 从 (c_k_min, c_k_max) 懒初始化一次.

    Args:
        codebook: (K, e_dim) — 码字 (用于 device/dtype 对齐).
        layer_idx: 0/1/2.
        c_k_range: (low, high) — per-layer c_k range.
        seed: RNG seed.

    Returns:
        c_k: (K,) tensor on codebook.device/dtype, U[c_k_range[0], c_k_range[1]].
    """
    K = codebook.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed + layer_idx * 1000)
    c_k = torch.zeros(K).uniform_(c_k_range[0], c_k_range[1], generator=g)
    return c_k.to(device=codebook.device, dtype=codebook.dtype)


def gumbel_softmax_assign_per_layer(
    latent: torch.Tensor,
    codebook: torch.Tensor,
    c_k: torch.Tensor,
    tau: float,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-layer Gumbel-Softmax soft-assign with per-codeword κ-Stereographic distance.

    数学:
      - per-codeword 距离 d_k = poincare_dist(z, e_k, c_k).squeeze(-1)  (B, K)
      - Gumbel-Softmax 概率:
          logit_k = -d_k² / τ + g_k    (g_k ~ Gumbel(0,1) 仅训练)
          prob = softmax(logit_k, dim=-1)
      - straight-through estimator:
          x_q_soft = prob @ codebook
          x_q_hard = codebook[argmax(prob)]
          x_q = x_q_hard + (x_q_soft - x_q_hard).detach()

    Args:
        latent: (B, e_dim)
        codebook: (K, e_dim) — 切空间码字
        c_k: (K,) per-codeword curvature (per-layer 异构)
        tau: Gumbel-Softmax 温度
        training: True → 加 gumbel noise; False → 不加

    Returns:
        (x_q_st, indices, prob):
          x_q_st: (B, e_dim) straight-through quantized output
          indices: (B,) hard 选中的码字 index
          prob: (B, K) soft-assign probability (供 Stage 2 硬分配 fallback)
    """
    B, K = latent.shape[0], codebook.shape[0]
    # per-codeword κ-Stereographic distance (Task #218+#219 + Issue #28)
    eps = 1e-5
    lat_n = latent.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    cb_n = codebook.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    lat_in = latent / latent.norm(dim=-1, keepdim=True).clamp(min=1e-12) * lat_n
    cb_in = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cb_n
    inv_sqrt_c = 1.0 / c_k.sqrt().clamp(min=1e-6)  # (K,)
    # sphere formula: arccosh(1 + 2·c·‖z-e‖²/[(1-c·‖z‖²)(1-c·‖e‖²)])
    lat_sq = (lat_in ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
    cb_sq = (cb_in ** 2).sum(dim=-1, keepdim=True)  # (K, 1)
    cross = lat_in @ cb_in.t()  # (B, K)
    eucl_d_sq = lat_sq + cb_sq.t() - 2 * cross  # (B, K)
    cz = c_k.view(1, K) * lat_sq  # (B, K)
    ce = c_k.view(1, K) * cb_sq.t()  # (B, K)
    one_m_cz = (1 - cz).clamp(min=1e-6)
    one_m_ce = (1 - ce).clamp(min=1e-6)
    inner = 1 + 2 * c_k.view(1, K) * eucl_d_sq / (one_m_cz * one_m_ce)
    inner = inner.clamp(min=1.0 + 1e-12)
    d_per_codeword = inv_sqrt_c.view(1, K) * torch.acosh(inner)  # (B, K) — per-codeword 距离
    d_sq_per_codeword = d_per_codeword ** 2
    # Gumbel-Softmax prob = softmax(-d²/τ)
    log_prob = -d_sq_per_codeword / max(tau, 1e-10)
    if training:
        gumbel = -torch.log(-torch.log(torch.rand_like(log_prob).clamp(min=1e-12)) + 1e-12)
        log_prob = log_prob + gumbel
    prob = F.softmax(log_prob, dim=-1)  # (B, K)
    indices = prob.argmax(dim=-1)  # (B,)
    # Straight-through estimator (Jang et al. 2017 Categorical Reparameterization):
    #   forward:  x_q_st = x_q_hard (用 hard argmax — 跟 baseline 一致)
    #   backward: d(x_q_st)/d(params) = d(x_q_soft)/d(params) (用 soft 加权)
    # 写法: x_q_st = (x_q_hard - x_q_soft).detach() + x_q_soft
    #       forward  : (x_q_hard - x_q_soft) + x_q_soft = x_q_hard
    #       backward : d(x_q_soft)/d(params) only (x_q_hard detached)
    x_q_soft = prob @ codebook  # (B, e_dim)
    x_q_hard = codebook[indices]
    x_q_st = (x_q_hard - x_q_soft).detach() + x_q_soft
    return x_q_st, indices, prob


class GumbelSoftmaxResidualQuantization(torch.nn.Module):
    """Task #298 / Issue #28: 包装 baseline HResidualVectorQuantization,
    把每层 argmin hard-assign 替换为 Gumbel-Softmax soft-assign + per-layer τ_l +
    per-layer c_k range (Issue #28 §Gate 0). Stage 2 推断 (eval mode) 退化为 baseline argmin.

    Args:
        base_hrq: 原始 HResidualVectorQuantization (from baseline HRQVAE.hrq)
        tau_list: per-layer τ_l (len = len(n_e_list))
        c_k_range_list: per-layer c_k range (len = len(n_e_list))
        c_k_seed: per-codeword c_k 抽样 seed
    """

    def __init__(self, base_hrq, tau_list: List[float], c_k_range_list: List[Tuple[float, float]],
                 c_k_seed: int = 42):
        super().__init__()
        self.base_hrq = base_hrq
        self.tau_list = list(tau_list)
        self.c_k_range_list = list(c_k_range_list)
        self.c_k_seed = c_k_seed
        # 懒初始化 per-layer c_k
        self._c_k_per_layer: List[torch.Tensor] = [None] * len(base_hrq.vq_layers)
        # 暴露跟 base_hrq 兼容的属性 (HRQVAE.forward 用 self.hrq.vq_layers / .n_e_list 等)
        self.n_e_list = base_hrq.n_e_list
        self.e_dim = base_hrq.e_dim
        self.beta = base_hrq.beta
        self.vq_layers = base_hrq.vq_layers  # 给 hrqvae_trainer.ReviveDeadCodes 等兼容

    def _get_layer_c_k(self, layer_idx: int, codebook: torch.Tensor) -> torch.Tensor:
        if self._c_k_per_layer[layer_idx] is None:
            self._c_k_per_layer[layer_idx] = _ensure_per_layer_c_k(
                codebook, layer_idx, self.c_k_range_list[layer_idx], self.c_k_seed)
        return self._c_k_per_layer[layer_idx]

    def forward(self, x, use_sk=True, return_x_res=False):
        # Stage 2 推断 / eval mode → 退回 baseline hard argmin (Issue #28 §Gate 0 明确)
        if not self.training or not use_sk:
            return self.base_hrq(x, use_sk=use_sk, return_x_res=return_x_res)
        # Stage 1 训练: Gumbel-Softmax soft-assign (Issue #28 §Gate 0)
        all_losses = []
        all_indices = []
        all_x_res = [] if return_x_res else None
        x_q = 0
        residual = x
        B = x.shape[0]
        # 跟 baseline 一样: 在循环里算 commitment_loss + codebook_loss
        for li, quantizer in enumerate(self.vq_layers):
            # Issue #28 fix: 用 get_codebook() (Poincaré ball, post proj_to_ball + expmap0),
            # 而不是 quantizer.embeddings.weight (切空间 raw, norm≈0.01).
            # 后者会让 poincare_distance 几乎全 0 → softmax 均匀 → straight-through 无信号 → 坍缩.
            codebook = quantizer.get_codebook()  # (K, e_dim) Poincaré ball, norm ≤ 1-eps
            c_k = self._get_layer_c_k(li, codebook)
            tau = self.tau_list[li]
            x_q_st, indices, prob = gumbel_softmax_assign_per_layer(
                residual, codebook, c_k, tau, training=True,
            )
            # commitment / codebook loss (跟 baseline 一致 — Poincaré distance²)
            # baseline 用的距离是 self.c (固定 1.0 for κ-decouple), 这里也用 c=1.0
            # 因为 per-codeword 距离用于 softmax (Gumbel), 但 commit/code loss 用 baseline 公式
            d_cl = poincare_distance(x_q_st.detach(), residual, c=1.0).squeeze(-1) ** 2
            d_ql = poincare_distance(x_q_st, residual.detach(), c=1.0).squeeze(-1) ** 2
            loss = d_cl.mean() + self.beta * d_ql.mean()
            all_losses.append(loss)
            all_indices.append(indices)
            if return_x_res:
                all_x_res.append(x_q_st)
            # residual 更新 (跟 baseline 一样减)
            residual = residual - x_q_st
            x_q = x_q + x_q_st
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        if return_x_res:
            return x_q, mean_loss, all_indices, all_x_res
        return x_q, mean_loss, all_indices

    def get_codebook(self):
        return self.base_hrq.get_codebook()


# ============================================================================
# Train Launcher (继承 baseline args + 加 Gumbel-Softmax 专属 args)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Task #298 / Issue #28 train_hrqvae_gumbel')
    # 复用 baseline args (HRQVAE + EmbDataset)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100, help='Issue #28 Gate 1 = 100 epoch')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--learner', type=str, default='AdamW')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--data_path', type=str, default=f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--kmeans_init', type=bool, default=True)
    parser.add_argument('--kmeans_iters', type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument('--sk_iters', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--loss_mult_codebook', type=float, default=1.0)
    parser.add_argument('--radii', type=float, nargs='+', default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--save_limit', type=int, default=1, help='R12: 只保留最新')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/Instruments_gumbel')
    parser.add_argument('--seed', type=int, default=42)
    # Issue #28 专属参数
    parser.add_argument('--tau_list', type=float, nargs='+', default=[1.0, 0.5, 0.1],
                        help='per-layer Gumbel-Softmax 温度 (Issue #28 §Gate 0). '
                             'L0=1.0 软探索, L1=0.5 medium, L2=0.1 near-hard.')
    parser.add_argument('--c_k_range_list', type=str, default='1.0:5.0,0.5:20.0,0.5:20.0',
                        help='per-layer c_k range (Issue #28 §Gate 0). '
                             '格式 "low1:high1,low2:high2,..." (CSV).')
    parser.add_argument('--c_k_seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 解析 c_k_range_list (CSV → List[Tuple[float, float]])
    c_k_range_list = [tuple(map(float, r.split(':'))) for r in args.c_k_range_list.split(',')]
    assert len(c_k_range_list) == len(args.num_emb_list), \
        f"c_k_range_list len {len(c_k_range_list)} != num_emb_list len {len(args.num_emb_list)}"
    assert len(args.tau_list) == len(args.num_emb_list), \
        f"tau_list len {len(args.tau_list)} != num_emb_list len {len(args.num_emb_list)}"

    print(f'[Issue #28] τ_l = {args.tau_list} (per-layer 异构温度)')
    print(f'[Issue #28] c_k_range_list = {c_k_range_list} (per-layer 异构 metric)')

    # baseline HRQVAE (vanilla κ-decouple setup, 跟 task144/task287 一样)
    data = EmbDataset(args.data_path)
    print(f'[Issue #28] Dataset: {len(data)} items, dim={data.dim}')
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
        curvature_list=None,  # 单一 curvature=1.0 (跟 task144 一致)
        euclidean_qloss=False,
        loss_mult_codebook=args.loss_mult_codebook,
        radii=args.radii,
    )

    # Monkey-patch: 用 GumbelSoftmaxResidualQuantization 替换 baseline HResidualVectorQuantization
    print(f'[Issue #28] Monkey-patch model.hrq → GumbelSoftmaxResidualQuantization')
    model.hrq = GumbelSoftmaxResidualQuantization(
        base_hrq=model.hrq,
        tau_list=args.tau_list,
        c_k_range_list=c_k_range_list,
        c_k_seed=args.c_k_seed,
    )

    # Trainer (跟 baseline train_hrqvae.py 一致)
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Issue #28] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()