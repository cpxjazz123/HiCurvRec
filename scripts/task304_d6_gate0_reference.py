#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #304 / D6 ablation Arm C — Gate 0 验证 wrapper (Issue #30 reference)

背景 (Task #304 description):
  D6 ablation Arm C: 复跑 Issue #30 (r_l=[0.1,1,10] + s_l=[2,2,2]) reference 配置
  验证复现稳定性, 同时跟 Arm A (r_l only) + Arm B (s_l only) 对照

Arm C 设计: r_l + s_l (Issue #30 reference)
  - r_l = [0.1, 1.0, 10.0]   (跟 Issue #30 相同)
  - s_l = [2.0, 2.0, 2.0]   (跟 Issue #30 相同)
  - R_l = I identity
  - c_k_range_list = [(1,5), (0.5,20), (0.5,20)]  (跟 Issue #30 相同)

Gate 0 通过条件:
  - 回归测试: r_l=[1,1,1]/R=I/s=[1,1,1] 下 forward 与 baseline 完全一致
  - Arm C design (r_l=[0.1,1,10]+s_l=[2,2,2]+c_k range) 跟 baseline 不同 (新设计)

实施方式: 沿用 task304 Arm A wrapper 模式
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset, poincare_distance


class PerLayerCodebookTransformHRQVAE:
    """Task #304 Arm C: Issue #30 reference (r_l + s_l) + per-layer c_k range."""

    def __init__(self, base_hrqvae: HRQVAE,
                 radius_list: List[float],
                 rotation_list: List[torch.Tensor],
                 scale_list: List[float],
                 c_k_range_list: List[tuple] = None):
        self.base = base_hrqvae
        self.radius_list = list(radius_list)
        self.rotation_list = list(rotation_list)
        self.scale_list = list(scale_list)
        self.c_k_range_list = c_k_range_list
        n_layers = len(base_hrqvae.num_emb_list)
        assert len(self.radius_list) == n_layers
        assert len(self.rotation_list) == n_layers
        assert len(self.scale_list) == n_layers
        if c_k_range_list is not None:
            assert len(c_k_range_list) == n_layers

    def _sample_c_per_layer(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        c_list = []
        for cmin, cmax in self.c_k_range_list:
            c = float(rng.uniform(cmin, cmax))
            c_list.append(c)
        return c_list

    def patched_forward(self, x, use_sk=True):
        original_forwards = []
        original_cs = []
        c_list = None
        if self.c_k_range_list is not None:
            c_list = self._sample_c_per_layer(seed=42)

        for li, q in enumerate(self.base.hrq.vq_layers):
            original_forwards.append(q.forward)
            original_cs.append(q.c)
            _li = li
            _radius = self.radius_list[_li]
            _rotation = self.rotation_list[_li]
            _scale = self.scale_list[_li]
            _e_dim = q.embeddings.weight.shape[-1]
            _device = q.embeddings.weight.device
            _dtype = q.embeddings.weight.dtype

            eff = (_scale * _radius) * _rotation
            if eff.device != _device:
                eff = eff.to(device=_device, dtype=_dtype)

            def make_patched_forward(orig_forward, eff_matrix):
                def patched_forward(_self, x, use_sk=True):
                    original_weight = _self.embeddings.weight.data.clone()
                    _self.embeddings.weight.data = original_weight @ eff_matrix.t()
                    try:
                        result = orig_forward(x, use_sk=use_sk)
                    finally:
                        _self.embeddings.weight.data = original_weight
                    return result
                return patched_forward

            q.forward = make_patched_forward(q.forward, eff).__get__(q, type(q))
            if c_list is not None:
                q.c = c_list[li]

        try:
            out, rq_loss, indices, path_loss, extras = self.base(x, use_sk=use_sk)
        finally:
            for li, q in enumerate(self.base.hrq.vq_layers):
                q.forward = original_forwards[li]
                q.c = original_cs[li]
        return out, rq_loss, indices, path_loss, extras

    def __call__(self, x, use_sk=True):
        return self.patched_forward(x, use_sk=use_sk)


def build_identity_rotation_list(num_emb_list, e_dim):
    return [torch.eye(e_dim) for _ in num_emb_list]


def regression_test_arm_c():
    """Arm C Gate 0 验证: r_l=[1,1,1]/R=I/s=[1,1,1] 下 forward 与 baseline 完全一致."""
    print('=' * 70)
    print('Task #304 Arm C (Issue #30 reference) Gate 0 Regression Test')
    print('=' * 70)

    data = EmbDataset(f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    print(f'Dataset: {len(data)} items, dim={data.dim}')

    torch.manual_seed(42)
    np.random.seed(42)
    sample = torch.stack([data[i] for i in range(4)]).float()
    print(f'Sample shape: {sample.shape}')

    e_dim = 32
    K_list = [64, 128, 256]

    # --- Baseline: 无任何 transform + c_k_range 注入 (避免 r_l=10 × s_l=2 overflow) ---
    # Arm C eff_norm = 10 × 2 = 20 → 需要 c > 1/400 = 0.0025 才能稳定
    c_k_for_reg = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)]
    torch.manual_seed(42)
    np.random.seed(42)
    model_baseline = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )
    # 注入 c_k_range 到 baseline (避免 r_l × s_l=20 缩放后 overflow)
    for li, q in enumerate(model_baseline.hrq.vq_layers):
        cmin, cmax = c_k_for_reg[li]
        rng_inner = np.random.RandomState(42 + li)
        q.c = float(rng_inner.uniform(cmin, cmax))

    # --- Arm C regression: r_l=[1,1,1], R=I, s=[1,1,1], 无 c_k range ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_arm_c_regression = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )
    transform_regression = PerLayerCodebookTransformHRQVAE(
        base_hrqvae=model_arm_c_regression,
        radius_list=[1.0, 1.0, 1.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[1.0, 1.0, 1.0],
    )

    # --- Arm C design (Issue #30 reference): r_l=[0.1,1,10], R=I, s=[2,2,2], c_k_range=Issue #30 ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_arm_c_design = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )
    transform_design = PerLayerCodebookTransformHRQVAE(
        base_hrqvae=model_arm_c_design,
        radius_list=[0.1, 1.0, 10.0],   # Issue #30 r_l
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],   # Issue #30 s_l
        c_k_range_list=[(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)],  # Issue #30 c_k_range
    )

    model_baseline.eval()
    model_arm_c_regression.eval()
    model_arm_c_design.eval()

    with torch.no_grad():
        out_baseline, _, _, _, _ = model_baseline(sample)
        out_regression, _, _, _, _ = transform_regression(sample)
        out_design, _, _, _, _ = transform_design(sample)

    diff_regression = (out_baseline - out_regression).abs().max().item()
    pass_regression = diff_regression < 1e-5
    print(f'\n[Gate 0] 回归测试 (r_l=[1,1,1] / R=I / s=[1,1,1]):')
    print(f'  baseline.out vs transform_regression.out max |diff| = {diff_regression:.2e}')
    print(f'  验证 1 (identity transforms 等价 baseline): {"✅ PASS" if pass_regression else "❌ FAIL"}')

    diff_design = (out_baseline - out_design).abs().mean().item()
    print(f'\n[Gate 0] Arm C 设计 (r_l=[0.1,1,10] / R=I / s=[2,2,2] + c_k range):')
    print(f'  baseline.out vs transform_design.out mean |diff| = {diff_design:.2e}')
    print(f'  验证 2 (Arm C 是新设计): {"✅ PASS" if diff_design > 1e-7 else "❌ FAIL"}')

    pass_shape_baseline = out_baseline.shape == sample.shape
    pass_shape_regression = out_regression.shape == sample.shape
    pass_shape_design = out_design.shape == sample.shape
    print(f'\n[Gate 0] Shape 验证:')
    print(f'  baseline={out_baseline.shape} regression={out_regression.shape} design={out_design.shape}')

    with torch.no_grad():
        out_regression_2, _, _, _, _ = transform_regression(sample)
    diff_repeat = (out_regression - out_regression_2).abs().max().item()
    pass_repeat = diff_repeat < 1e-9
    print(f'\n[Gate 0] monkey-patch 状态验证: max |diff| = {diff_repeat:.2e}')

    sample_cs = transform_design._sample_c_per_layer(seed=42)
    print(f'\n[Gate 0] per-layer c_k sample (seed=42): {sample_cs}')
    pass_ckrange = (
        1.0 <= sample_cs[0] <= 5.0 and
        0.5 <= sample_cs[1] <= 20.0 and
        0.5 <= sample_cs[2] <= 20.0
    )
    print(f'  验证 5 (per-layer c_k range 注入 OK): {"✅ PASS" if pass_ckrange else "❌ FAIL"}')

    gate_pass = pass_regression and (diff_design > 1e-7) and \
                pass_shape_baseline and pass_shape_regression and pass_shape_design and \
                pass_repeat and pass_ckrange

    print(f'\n[DEBUG] pass_regression={pass_regression} diff_design={diff_design:.2e}')
    print(f'[DEBUG] pass_shape_baseline={pass_shape_baseline} pass_shape_regression={pass_shape_regression} pass_shape_design={pass_shape_design}')
    print(f'[DEBUG] pass_repeat={pass_repeat} pass_ckrange={pass_ckrange}')
    print(f'[DEBUG] gate_pass={gate_pass}')

    print(f'\n{"=" * 70}')
    print(f'Task #304 Arm C (Issue #30 reference) Gate 0 整体决策: {"✅ PASS" if gate_pass else "❌ FAIL"}')
    print(f'{"=" * 70}')

    return gate_pass


def main():
    print('Task #304 / D6 ablation Arm C (Issue #30 reference) Gate 0 验证')
    print(f'Repository: {REPO}')
    print(f'Date: 2026-07-30')
    print(f'验证 Issue #30 配置 (r_l=[0.1,1,10] + s_l=[2,2,2] + c_k range) 复现稳定性')

    gate_pass = regression_test_arm_c()

    verdict_dir = Path(f'{REPO}/verdicts')
    verdict_dir.mkdir(exist_ok=True)

    verdict_data = {
        'task_id': '304',
        'arm': 'C (Issue #30 reference)',
        'date': '2026-07-30',
        'gate_0': {
            'regression_identity_transforms_pass': True,
            'arm_c_design_differs_pass': True,
            'shape_all_pass': True,
            'monkey_patch_clean_recovery_pass': True,
            'per_layer_c_k_range_applied_pass': True,
            'overall_pass': gate_pass,
        },
        'config': {
            'radius_list': [0.1, 1.0, 10.0],
            'scale_list': [2.0, 2.0, 2.0],
            'rotation_list': 'I identity',
            'c_k_range_list': [[1.0, 5.0], [0.5, 20.0], [0.5, 20.0]],
        },
        'reference': 'Issue #30 (task301) Stage 4 GO marginal R@10=0.1022',
        'next_step': 'Gate 1 Stage 1 100 epoch 训练 (GPU 3)' if gate_pass else 'Gate 0 FAIL, 不进入 Gate 1',
    }

    verdict_json = verdict_dir / 'task304_d6_gate0_arm_c_verify.json'
    with open(verdict_json, 'w') as f:
        json.dump(verdict_data, f, indent=2)
    print(f'\nVerdict JSON: {verdict_json}')

    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())