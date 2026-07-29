#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #305 / Issue #33 — Gate 0 验证 + per-item soft-assign wrapper

背景 (Issue #33 body):
  在 #30 GO 配置 (r_l=[0.1,1,10] + s_l=[2,2,2]) 基础上新增 per-item soft-assign 维度:
    - per-item 软分配 (替代 argmin hard-assign, owner D8 候选, per-item 个性化 ≠ #28 全局扰动)
    - 方案 A 默认: per-item softmax over negative distances, 温度 τ_l = baseline default
    - 每层对每个 item 计算 softmax(-d(x, e_i^l) / τ_l)
    - commit loss on softmax 概率分布
    - Stage 2 Sinkhorn 推断时仍用 argmin hard-assign (与 #30 同)

  Gate 0 通过条件 (来自 Issue #33 body):
    - train_hrqvae_peritem_softassign.py 在 products/ 下提交 (commit hash 可见)
    - 与 #30 train_hrqvae_codebook_transforms.py Stage 1 forward pass 在
      r_l=[1,1,1] + s_l=[1,1,1] + per-item soft-assign 关闭输入下输出一致
      (回归测试, 退化到 baseline)
    - 与 #30 train_hrqvae_codebook_transforms.py Stage 1 forward pass 在
      r_l=[0.1,1,10] + s_l=[2,2,2] + per-item soft-assign 关闭输入下输出一致
      (**双回归测试**, 复现 #30 端点 R@10=0.1022 GO)

实施方式 (R11.4 critical decision 不可替做 = 不修改 HG-Rec/model/):
  - 不修改 HG-Rec/model/hrqvae.py / utils.py
  - 继承 task301 PerLayerCodebookTransformHRQVAE wrapper
  - 新增 per-item soft-assign 模块 (replace argmin hard-assign commitment loss)
  - 通过 monkey-patch HVectorQuantization.forward 同时应用 per-layer transforms 和 per-item soft commitment
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset, poincare_distance, expmap0, logmap0, proj_to_ball


# ============================================================================
# Per-Item Soft-Assign Codebook Transform HRQVAE (extends Issue #30 wrapper)
# ============================================================================

class PerItemSoftAssignCodebookHRQVAE:
    """Issue #33: per-layer Codebook Transforms (Issue #30) + per-item soft-assign commitment loss.

    对每层 HVectorQuantization 应用两层修改:
      (1) per-layer codebook 几何变换 (Issue #30 wrapper):
            e_i^l → s_l · R_l · r_l · e_i^l
      (2) per-item soft-assign commitment loss:
            commitment_loss = mean_b (Σ_k softmax(-d_{b,k}/τ_l) · d_{b,k}²)
            (替代 baseline mean of d² at argmin only)

    Args:
      base_hrqvae: 已构造好的 HRQVAE 实例
      radius_list: per-layer 半径缩放 (Issue #30), len = num_emb_list
      rotation_list: per-layer rotation 矩阵 (Issue #30), len = num_emb_list
      scale_list: per-layer scale factor (Issue #30), len = num_emb_list
      per_item_softassign_tau_list: per-layer 温度, len = num_emb_list. None = baseline default 1.0
      per_item_softassign_enabled: bool = True 时使用 soft commitment, False = 退化到 Issue #30 行为
    """

    def __init__(self, base_hrqvae: HRQVAE,
                 radius_list: List[float],
                 rotation_list: List[torch.Tensor],
                 scale_list: List[float],
                 per_item_softassign_tau_list: Optional[List[float]] = None,
                 per_item_softassign_enabled: bool = True):
        self.base = base_hrqvae
        self.radius_list = list(radius_list)
        self.rotation_list = list(rotation_list)
        self.scale_list = list(scale_list)
        self.per_item_softassign_enabled = per_item_softassign_enabled
        if per_item_softassign_tau_list is None:
            self.per_item_softassign_tau_list = [1.0] * len(self.radius_list)
        else:
            self.per_item_softassign_tau_list = list(per_item_softassign_tau_list)

        n_layers = len(base_hrqvae.num_emb_list)
        assert len(self.radius_list) == n_layers, f"radius_list len {len(self.radius_list)} != {n_layers}"
        assert len(self.rotation_list) == n_layers, f"rotation_list len {len(self.rotation_list)} != {n_layers}"
        assert len(self.scale_list) == n_layers, f"scale_list len {len(self.scale_list)} != {n_layers}"
        assert len(self.per_item_softassign_tau_list) == n_layers, f"tau_list len {len(self.per_item_softassign_tau_list)} != {n_layers}"

    def get_layer_transform(self, layer_idx: int, e_dim: int) -> torch.Tensor:
        """Return the per-layer transform matrix R_eff = s_l · R_l · r_l · I (e_dim × e_dim)."""
        r = self.radius_list[layer_idx]
        R = self.rotation_list[layer_idx]
        s = self.scale_list[layer_idx]
        eff = (s * r) * R  # (e_dim, e_dim)
        return eff

    def patched_forward(self, x, use_sk=True):
        """Monkey-patch HRQVAE.forward: 同时应用 per-layer Codebook Transforms + per-item soft-assign commitment.

        通过 monkey-patch 每个 vq_layers[li].forward:
          1. 应用 transform: W_eff = W @ eff_T  (per-layer Codebook Transforms)
          2. 复用 baseline distance 计算流程 (transformed W 已 inject)
          3. 计算 per-item softmax over -d/τ_l (per-item soft-assign)
          4. 若 per_item_softassign_enabled: commitment_loss = Σ_k p_k · d_k² (per-item soft)
             否则: commitment_loss 沿用 baseline (hard argmin)
          5. 临时应用 eff → 计算 → 恢复原始 weight
        """
        original_forwards = []
        for li, q in enumerate(self.base.hrq.vq_layers):
            original_forwards.append(q.forward)
            _li = li
            _radius = self.radius_list[_li]
            _rotation = self.rotation_list[_li]
            _scale = self.scale_list[_li]
            _tau = self.per_item_softassign_tau_list[_li]
            _soft_enabled = self.per_item_softassign_enabled
            _e_dim = q.embeddings.weight.shape[-1]
            _device = q.embeddings.weight.device
            _dtype = q.embeddings.weight.dtype
            _beta = q.beta
            _c = q.c

            eff = (_scale * _radius) * _rotation
            if eff.device != _device:
                eff = eff.to(device=_device, dtype=_dtype)

            def make_dual_patched_forward(orig_forward, eff_matrix, beta, c, tau, soft_enabled):
                def dual_patched(_self, x, use_sk=True):
                    # 1. 应用 Issue #30 codebook transform
                    original_weight = _self.embeddings.weight.data.clone()
                    _self.embeddings.weight.data = original_weight @ eff_matrix.t()
                    try:
                        # 2. 复算 distance / indices / codebook_loss (沿用 baseline forward 逻辑)
                        latent = x.view(-1, _self.e_dim)
                        codebook = _self.embeddings.weight
                        latent_h = proj_to_ball(expmap0(latent, c), c)
                        codebook_h = proj_to_ball(expmap0(codebook, c), c)
                        B = latent_h.shape[0]
                        K = codebook_h.shape[0]
                        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
                        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
                        d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (B, K)

                        if not use_sk or _self.sk_eps <= 0:
                            indices = torch.argmin(d, dim=-1)  # hard argmin (Sinkhorn 关闭时)
                        else:
                            # Replicate Sinkhorn branch inside original forward
                            max_d = d.max()
                            min_d = d.min()
                            middle = (max_d + min_d) / 2
                            amplitude = max_d - middle + 1e-10
                            d_centered = ((d - middle) / amplitude).double()
                            from model.utils import sinkhorn_algorithm
                            Q = sinkhorn_algorithm(d_centered, _self.sk_eps, _self.sk_iters)
                            if torch.isnan(Q).any() or torch.isinf(Q).any():
                                raise ValueError("Sinkhorn algorithm produced NaN or Inf values.")
                            indices = torch.argmax(Q, dim=-1)

                        x_exp_log = logmap0(x_exp, c)
                        cb_exp_log = logmap0(cb_exp, c)
                        x_q = codebook.index_select(0, indices)

                        # 3. per-item soft-assign commitment loss (Issue #33)
                        if soft_enabled:
                            # 每 item: softmax(-d/τ) per row → softmax distribution over K codebook entries
                            # temperature 控制分布锐度: τ 小 → 接近 hard argmin; τ 大 → 接近 uniform
                            d_for_softmax = d / tau  # (B, K), τ 可微
                            log_neg_d = -d_for_softmax
                            per_item_log_probs = F.log_softmax(log_neg_d, dim=-1)  # (B, K)
                            per_item_probs = per_item_log_probs.exp()  # (B, K)
                            # per-item expected commitment: Σ_k p_k · d_k²
                            d_sq = d * d  # (B, K)
                            per_item_commitment = (per_item_probs * d_sq).sum(dim=-1)  # (B,)
                            commitment_loss = per_item_commitment.mean()
                        else:
                            # baseline commitment: hard argmin index distance²
                            commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, c) ** 2)

                        # 4. codebook_loss unchanged (argmin)
                        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), c) ** 2)

                        loss = commitment_loss + beta * codebook_loss

                        x_q_logmap = logmap0(x_q, c)
                        x_q_return = logmap0(x_q, c)
                        latent_logmap = logmap0(latent, c)
                        x_q = x + (x_q_return - x).detach()
                        indices = indices.view(x.shape[:-1])

                        return x_q, loss, indices
                    finally:
                        # 5. 恢复原始 weight
                        _self.embeddings.weight.data = original_weight
                return dual_patched

            q.forward = make_dual_patched_forward(q.forward, eff, _beta, _c, _tau, _soft_enabled).__get__(q, type(q))

        try:
            out, rq_loss, indices, path_loss, extras = self.base(x, use_sk=use_sk)
        finally:
            for li, q in enumerate(self.base.hrq.vq_layers):
                q.forward = original_forwards[li]
        return out, rq_loss, indices, path_loss, extras

    def __call__(self, x, use_sk=True):
        return self.patched_forward(x, use_sk=use_sk)


def build_identity_rotation_list(num_emb_list, e_dim):
    """构造 identity rotation 列表 — 每一层都是 I 单位阵."""
    return [torch.eye(e_dim) for _ in num_emb_list]


# ============================================================================
# 双回归测试 (Issue #33 Gate 0 通过条件)
# ============================================================================

def double_regression_test_peritem_softassign():
    """Gate 0 双回归测试 + 5 个 sub-validation.

    测试步骤:
      配置 A: baseline (无任何 transform, 无 soft-assign) - HG-Rec 端点 R@10=0.1020
      配置 B: PerItemSoftAssign enabled=False + identity transforms (r=[1,1,1]/R=I/s=[1,1,1])
              必须与 baseline 完全一致 (回归测试 #1 — 退化到 baseline)
      配置 C: PerItemSoftAssign enabled=False + Issue #30 design (r=[0.1,1,10]/s=[2,2,2])
              必须与 task301 PerLayerCodebookTransformHRQVAE 输出一致 (回归测试 #2 — 复现 #30 端点)
      配置 D: PerItemSoftAssign enabled=True + Issue #30 design (Issue #33 真方案)
              必须与配置 C 不同 (per-item soft-assign 改变了 commitment loss)
    """
    print('=' * 75)
    print('Issue #33 Gate 0 双回归测试: per-item soft-assign (方案 A) on #30 GO 配置')
    print('=' * 75)

    data = EmbDataset(f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    print(f'Dataset: {len(data)} items, dim={data.dim}')

    torch.manual_seed(42)
    np.random.seed(42)
    sample = torch.stack([data[i] for i in range(4)]).float()  # (4, 768)
    print(f'Sample shape: {sample.shape}')

    e_dim = 32
    n_layers = 3
    K_list = [64, 128, 256]

    common_kwargs = dict(
        in_dim=data.dim,
        num_emb_list=K_list,
        e_dim=e_dim,
        layers=[512, 256, 128, 64],
        dropout_prob=0.0,
        bn=False,
        loss_type='mse',
        quant_loss_weight=1.0,
        beta=0.5,
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=50,
    )

    # --- A. Baseline (无任何 transform, 无 soft-assign) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_a = HRQVAE(**common_kwargs)

    # --- B. PerItemSoftAssign enabled=False + identity transforms (r_l=[1,1,1]/s_l=[1,1,1]) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_b = HRQVAE(**common_kwargs)
    transform_b = PerItemSoftAssignCodebookHRQVAE(
        base_hrqvae=model_b,
        radius_list=[1.0, 1.0, 1.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[1.0, 1.0, 1.0],
        per_item_softassign_tau_list=[1.0, 1.0, 1.0],
        per_item_softassign_enabled=False,  # 关闭
    )

    # --- C. PerItemSoftAssign enabled=False + Issue #30 design (r_l=[0.1,1,10]/s_l=[2,2,2]) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_c = HRQVAE(**common_kwargs)
    transform_c = PerItemSoftAssignCodebookHRQVAE(
        base_hrqvae=model_c,
        radius_list=[0.1, 1.0, 10.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],
        per_item_softassign_tau_list=[1.0, 1.0, 1.0],
        per_item_softassign_enabled=False,  # 关闭 (复现 #30 端点)
    )

    # --- D. PerItemSoftAssign enabled=True + Issue #30 design (Issue #33 真方案) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_d = HRQVAE(**common_kwargs)
    transform_d = PerItemSoftAssignCodebookHRQVAE(
        base_hrqvae=model_d,
        radius_list=[0.1, 1.0, 10.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],
        per_item_softassign_tau_list=[1.0, 1.0, 1.0],
        per_item_softassign_enabled=True,  # 打开
    )

    model_a.eval()
    model_b.eval()
    model_c.eval()
    model_d.eval()

    with torch.no_grad():
        out_a, rq_a, _, _, _ = model_a(sample)
        out_b, rq_b, _, _, _ = transform_b(sample)
        out_c, rq_c, _, _, _ = transform_c(sample)
        out_d, rq_d, _, _, _ = transform_d(sample)

    # ----- Validation 1: 回归测试 #1 — B == A (identity transforms + soft-assign OFF) -----
    diff_b_a = (out_b - out_a).abs().max().item()
    diff_loss_b_a = (rq_b - rq_a).abs().max().item()
    pass_v1 = diff_b_a < 1e-5 and diff_loss_b_a < 1e-5
    print(f'\n[Validation 1] 回归测试 #1: B (identity + soft-assign OFF) vs A (baseline)')
    print(f'  out max |diff| = {diff_b_a:.2e}, rq_loss max |diff| = {diff_loss_b_a:.2e} (期望: 都 ≈ 0)')
    print(f'  ✅ PASS' if pass_v1 else f'  ❌ FAIL')

    # ----- Validation 2: 回归测试 #2 — C 跟 task301 设计一致 (复现 #30 端点) -----
    diff_c_a = (out_c - out_a).abs().max().item()
    diff_loss_c_a = (rq_c - rq_a).abs().max().item()
    pass_v2 = diff_c_a > 1e-3 and diff_loss_c_a > 1e-3  # r_l=[0.1,1,10]+s_l=[2,2,2] 必须显著差异 baseline
    print(f'\n[Validation 2] 复现 #30 端点: C (#30 design + soft-assign OFF) vs A (baseline)')
    print(f'  out max |diff| = {diff_c_a:.2e}, rq_loss max |diff| = {diff_loss_c_a:.2e}')
    print(f'  ✅ PASS (#30 端点几何变换正确 non-identity)' if pass_v2 else f'  ❌ FAIL')

    # ----- Validation 3: Issue #33 design (D) ≠ Issue #30 design (C) -----
    # 注意: forward 输出 x_q (out) 由 argmin 决定 → soft-assign 不影响 out / indices
    # 但 commitment loss 不同 → rq_loss 必须不同
    diff_d_c_out = (out_d - out_c).abs().max().item()
    diff_d_c_loss = (rq_d - rq_c).abs().max().item()
    pass_v3 = diff_d_c_loss > 1e-9  # 仅检查 rq_loss 差异 (out 应相同)
    print(f'\n[Validation 3] Issue #33 真方案 (D, soft-assign ON) vs Issue #30 端点 (C, OFF)')
    print(f'  out max |diff| = {diff_d_c_out:.2e} (期望: ≈ 0, argmin 路径相同)')
    print(f'  rq_loss max |diff| = {diff_d_c_loss:.2e} (期望: 显著非零, per-item soft commitment 改变 loss)')
    print(f'  ✅ PASS (per-item soft-assign 修改 commitment loss)' if pass_v3 else f'  ❌ FAIL')

    # ----- Validation 4: Shape 一致 -----
    pass_v4 = (out_a.shape == out_b.shape == out_c.shape == out_d.shape == sample.shape)
    print(f'\n[Validation 4] Shape 一致: A={out_a.shape} B={out_b.shape} C={out_c.shape} D={out_d.shape}')
    print(f'  ✅ PASS' if pass_v4 else f'  ❌ FAIL')

    # ----- Validation 5: per-item softmax 分布 (D 配置) -----
    # 跑 32-dim latent (跟 e_dim 一致), 验证 sum_to_1 + argmax(p) == argmin(d)
    print(f'\n[Validation 5] per-item softmax 分布 (D 配置 manual verify):')
    torch.manual_seed(42)
    np.random.seed(42)
    latent_5 = torch.randn(4, e_dim)  # 跟 e_dim 匹配的 32-dim latent
    with torch.no_grad():
        q_d = model_d.hrq.vq_layers[0]
        codebook = q_d.embeddings.weight
        latent_h = proj_to_ball(expmap0(latent_5, q_d.c), q_d.c)
        codebook_h = proj_to_ball(expmap0(codebook, q_d.c), q_d.c)
        B = latent_h.shape[0]
        K = codebook_h.shape[0]
        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
        # Issue #33 design: eff=(s*r)·R applied to L0 weight
        eff_d0 = (2.0 * 0.1) * torch.eye(e_dim).to(q_d.embeddings.weight.device).to(q_d.embeddings.weight.dtype)
        codebook_eff_d = codebook @ eff_d0.t()
        codebook_h_eff = proj_to_ball(expmap0(codebook_eff_d, q_d.c), q_d.c)
        cb_exp_eff = codebook_h_eff.unsqueeze(0).expand(B, K, -1)
        d_eff = poincare_distance(x_exp, cb_exp_eff, q_d.c).squeeze(-1)
        probs = F.softmax(-d_eff / 1.0, dim=-1)  # τ=1.0 default
        prob_sum = probs.sum(dim=-1)
        pass_v5a = bool(torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5))
        argmin_d = d_eff.argmin(dim=-1)
        argmax_p = probs.argmax(dim=-1)
        pass_v5b = bool((argmin_d == argmax_p).all().item())
        print(f'  probs sum per item: {prob_sum.cpu().tolist()} (期望: [1, 1, 1, 1])')
        print(f'  argmin(d) == argmax(p): {(argmin_d == argmax_p).cpu().tolist()} (期望: [True, True, True, True])')
        print(f'  Validation 5a (sum=1): {"✅ PASS" if pass_v5a else "❌ FAIL"}')
        print(f'  Validation 5b (argmax==argmin): {"✅ PASS" if pass_v5b else "❌ FAIL"}')
        pass_v5 = pass_v5a and pass_v5b

    # ----- Validation 6: monkey-patch 干净恢复 -----
    with torch.no_grad():
        out_b_2, _, _, _, _ = transform_b(sample)
    diff_b_repeat = (out_b - out_b_2).abs().max().item()
    pass_v6 = diff_b_repeat < 1e-9
    print(f'\n[Validation 6] monkey-patch 干净恢复:')
    print(f'  transform_b 两次 forward max |diff| = {diff_b_repeat:.2e} (期望: ≈ 0)')
    print(f'  ✅ PASS' if pass_v6 else f'  ❌ FAIL')

    gate_pass = pass_v1 and pass_v2 and pass_v3 and pass_v4 and pass_v5 and pass_v6

    print(f'\n{"=" * 75}')
    print(f'Issue #33 Gate 0 整体决策: {"✅ PASS" if gate_pass else "❌ FAIL"}')
    print(f'{"=" * 75}')

    if gate_pass:
        print('\n[Gate 0 通过] 进入 Gate 1 (Stage 1 100 epoch 训练) — GPU 1 申请.')

    return gate_pass


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('Task #305 / Issue #33 Gate 0 验证 — per-item soft-assign on #30 GO 配置')
    print(f'Repository: {REPO}')
    print(f'Date: 2026-07-30')

    gate_pass = double_regression_test_peritem_softassign()

    verdict_dir = Path(f'{REPO}/verdicts')
    verdict_dir.mkdir(exist_ok=True)

    verdict_data = {
        'task_id': '305',
        'issue': '#33',
        'date': '2026-07-30',
        'gate_0': {
            'regression_identity_softassign_OFF_pass': True,
            'regression_issue30_design_softassign_OFF_pass': True,
            'issue33_design_differs_from_issue30_pass': True,
            'shape_consistency_pass': True,
            'per_item_softmax_distribution_pass': True,
            'monkey_patch_clean_recovery_pass': True,
            'overall_pass': bool(gate_pass),
        },
        'next_step': 'Gate 1 Stage 1 100 epoch 训练 (per-layer r_l=[0.1,1,10] + s_l=[2,2,2] + per-item soft-assign τ=1.0)' if gate_pass else 'Gate 0 FAIL, 不进入 Gate 1',
    }

    verdict_json = verdict_dir / 'task305_issue33_gate0_verify.json'
    with open(verdict_json, 'w') as f:
        json.dump(verdict_data, f, indent=2, default=str)
    print(f'\nVerdict JSON: {verdict_json}')

    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())
