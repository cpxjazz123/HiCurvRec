#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #302 / Issue #31 — Gate 0 wrapper + reg test

背景 (Issue #31 body §"实验设计" + Gate 0 通过条件):
  Stage 1 per-layer 异构 encoder regularization:
    - β_l = [0.1, 0.3, 0.5]   (per-layer 异构 commit loss 权重, baseline 0.5 -> per-layer 异构)
    - α_l = [0.01, 0.005, 0.001]   (per-layer 异构 codebook center anchor)
    - γ_l = [0.001, 0.0005, 0.0001]   (per-layer 异构 encoder L2 正则)
    - c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)]   (task242 Arm A 协同)

Gate 0 通过条件 (Issue #31 §Gate 0):
  wrapper 在 β_l=[0.5,0.5,0.5] + α_l=[0,0,0] + γ_l=[0,0,0] 输入下, forward 输出 baseline 等价
  (回归测试, 三组正则全 0 -> baseline)

实施方式 (R11.4 + R11.5):
  - 不修改 HG-Rec/model/ 上游源码
  - 继承 HRQVAE, patch 每层 vq_layers[li].beta = beta_list[li] (per-layer β 异构)
  - 重写 forward 累加 anchor_loss (α_l * ||codebook||²) 和 enc_reg_loss (γ_l * ||residual_l||²)
  - α=γ=0 时, anchor_loss=enc_reg_loss=0, total=rq_loss → baseline 等价
"""
import sys
from typing import List, Optional

import numpy as np
import torch

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


class EncoderRegHRQVAE(HRQVAE):
    """Per-layer 异构 encoder regularization 包装 (Issue #31).

    在 baseline HRQVAE 上叠加:
      - per-layer commit loss β_l (patch vq_layers[li].beta)
      - per-layer codebook center anchor α_l (L_anchor = Σ α_l[i] * ||codebook_i||²)
      - per-layer encoder L2 γ_l (L_enc = Σ γ_l[i] * ||residual_l_input||², 防 encoder trivial solution)

    默认 β_list = α_list = γ_list = None 时, 行为与 baseline 完全一致 (reg test 条件).
    """

    def __init__(self,
                 *args,
                 beta_list: Optional[List[float]] = None,
                 alpha_list: Optional[List[float]] = None,
                 gamma_list: Optional[List[float]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        n_layers = len(self.hrq.vq_layers)

        # Patch per-layer β_l (Issue #31 §A)
        if beta_list is None:
            self.beta_list = [self.hrq.beta] * n_layers  # baseline β
        else:
            assert len(beta_list) == n_layers, \
                f"beta_list len {len(beta_list)} != n_layers {n_layers}"
            self.beta_list = list(beta_list)
            for li, q in enumerate(self.hrq.vq_layers):
                q.beta = float(beta_list[li])

        if alpha_list is None:
            self.alpha_list = [0.0] * n_layers  # baseline 关闭 anchor
        else:
            assert len(alpha_list) == n_layers, \
                f"alpha_list len {len(alpha_list)} != n_layers {n_layers}"
            self.alpha_list = [float(a) for a in alpha_list]

        if gamma_list is None:
            self.gamma_list = [0.0] * n_layers  # baseline 关闭 encoder L2
        else:
            assert len(gamma_list) == n_layers, \
                f"gamma_list len {len(gamma_list)} != n_layers {n_layers}"
            self.gamma_list = [float(g) for g in gamma_list]

    def forward(self, x, use_sk: bool = True, rho_target_batch=None):
        x_enc = self.encoder(x)

        # 手动跑 HRVQ + 累积 per-layer residual (用于 γ_l L2 正则)
        all_losses: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []
        x_q = 0.0
        residual = x_enc
        enc_reg_loss = x_enc.new_zeros(())
        for li, q in enumerate(self.hrq.vq_layers):
            # γ_l: 输入到当前 layer 的 residual 的 L2 norm
            if self.gamma_list[li] != 0.0:
                enc_reg_loss = enc_reg_loss + self.gamma_list[li] * (residual ** 2).sum()
            x_res, loss, indices = q(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)

        rq_loss = torch.stack(all_losses).mean()
        indices = torch.stack(all_indices, dim=-1)
        out = self.decoder(x_q)

        # α_l: codebook center anchor (codebook 范数 L2)
        anchor_loss = x_enc.new_zeros(())
        for li, q in enumerate(self.hrq.vq_layers):
            if self.alpha_list[li] != 0.0:
                anchor_loss = anchor_loss + self.alpha_list[li] * (q.embeddings.weight ** 2).sum()

        total_quant = rq_loss + enc_reg_loss + anchor_loss

        path_loss = None
        _div_ent = (None, None, anchor_loss, None)
        return out, total_quant, indices, path_loss, _div_ent


def make_baseline_hrqvae(data, layers_arch, num_emb_list, e_dim=32, beta=0.5, seed=42,
                          init_uniform: bool = True):
    """构造 baseline HRQVAE (reg test 对照组).

    Args:
      init_uniform: True 时用 uniform(-0.01, 0.01) 初始化 codebook (避免 kmeans_init 的 train-mode 依赖).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        layers=layers_arch,
        dropout_prob=0.0,
        bn=False,
        loss_type='poincare',
        quant_loss_weight=1.0,
        beta=beta,
        kmeans_init=False,  # reg test 关闭 kmeans, 改用 uniform init
        kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=50,
    )
    if init_uniform:
        with torch.no_grad():
            for q in model.hrq.vq_layers:
                q.embeddings.weight.data.uniform_(-0.01, 0.01)
                q.initted = True
    return model


def make_wrapper_hrqvae(data, layers_arch, num_emb_list, e_dim=32,
                        beta_list=None, alpha_list=None, gamma_list=None,
                        seed=42, beta_baseline=0.5,
                        init_uniform: bool = True):
    """构造 EncoderRegHRQVAE wrapper (reg test 实验组).

    Args:
      init_uniform: True 时用 uniform(-0.01, 0.01) 初始化 codebook (避免 kmeans_init 的 train-mode 依赖).
                     默认 True 用于 reg test; Gate 1 Stage 1 训练时设 False 用 kmeans_init.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EncoderRegHRQVAE(
        in_dim=data.dim,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        layers=layers_arch,
        dropout_prob=0.0,
        bn=False,
        loss_type='poincare',
        quant_loss_weight=1.0,
        beta=beta_baseline,  # 传给 super().__init__, 会被 beta_list patch 覆盖
        kmeans_init=False,  # reg test 关闭 kmeans, 改用 uniform init
        kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=50,
        beta_list=beta_list,
        alpha_list=alpha_list,
        gamma_list=gamma_list,
    )
    if init_uniform:
        with torch.no_grad():
            for q in model.hrq.vq_layers:
                q.embeddings.weight.data.uniform_(-0.01, 0.01)
                q.initted = True
    return model


def regtest_compare(baseline: HRQVAE, wrapper: HRQVAE, x: torch.Tensor,
                     label: str = "regtest", verbose: bool = True):
    """Forward 同一输入 x, 比较 (out, rq_loss, indices) 是否完全一致."""
    baseline.eval()
    wrapper.eval()

    with torch.no_grad():
        out_b, rq_b, idx_b, _, _ = baseline(x, use_sk=False)  # 关闭 sinkhorn, 保证确定性
        out_w, rq_w, idx_w, _, _ = wrapper(x, use_sk=False)

    out_diff = (out_b - out_w).abs().max().item()
    rq_diff = (rq_b - rq_w).abs().max().item()
    idx_diff = (idx_b - idx_w).abs().max().item()
    idx_eq = bool(torch.equal(idx_b, idx_w))

    passed = (out_diff == 0.0) and (rq_diff < 1e-6) and idx_eq

    if verbose:
        print(f"[{label}] out_max_diff = {out_diff:.6e}")
        print(f"[{label}] rq_loss_max_diff = {rq_diff:.6e}")
        print(f"[{label}] idx_max_diff = {idx_diff}, idx_equal = {idx_eq}")
        print(f"[{label}] baseline rq_loss = {rq_b.item():.6f}, wrapper rq_loss = {rq_w.item():.6f}")
        print(f"[{label}] {'✅ PASS' if passed else '❌ FAIL'}")

    return passed


def main():
    """Gate 0 reg test 主流程."""
    import argparse
    parser = argparse.ArgumentParser(description='Task #302 / Issue #31 Gate 0 reg test')
    parser.add_argument('--data_path', type=str,
                        default=f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size 必须 >= max(num_emb_list), 因 kmeans_init=True 需要')
    parser.add_argument('--beta_baseline', type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("Task #302 / Issue #31 — Gate 0 reg test")
    print("=" * 60)
    print(f"data_path = {args.data_path}")
    print(f"num_emb_list = {args.num_emb_list}")
    print(f"e_dim = {args.e_dim}")
    print(f"layers = {args.layers}")
    print(f"seed = {args.seed}")

    # 加载数据
    data = EmbDataset(args.data_path)
    print(f"Dataset: {len(data)} items, dim={data.dim}")

    # reg test 输入: 随机 batch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    idx = torch.randperm(len(data))[: args.batch_size]
    x = torch.stack([data[int(i)].float() for i in idx])
    print(f"Input batch shape: {tuple(x.shape)}")

    # Test 1: β_l=[0.5,0.5,0.5], α_l=[0,0,0], γ_l=[0,0,0] → baseline 等价
    print()
    print("=" * 60)
    print("Test 1: β_l=[0.5,0.5,0.5], α_l=[0,0,0], γ_l=[0,0,0]")
    print("=" * 60)
    baseline = make_baseline_hrqvae(
        data, args.layers, args.num_emb_list,
        e_dim=args.e_dim, beta=args.beta_baseline, seed=args.seed)
    wrapper = make_wrapper_hrqvae(
        data, args.layers, args.num_emb_list,
        e_dim=args.e_dim,
        beta_list=[args.beta_baseline]*len(args.num_emb_list),
        alpha_list=[0.0]*len(args.num_emb_list),
        gamma_list=[0.0]*len(args.num_emb_list),
        seed=args.seed, beta_baseline=args.beta_baseline)
    test1_pass = regtest_compare(baseline, wrapper, x, label="Test1 β=[0.5]*3 α=γ=0")

    # Test 2: β_l=[0.1,0.3,0.5], α_l=[0,0,0], γ_l=[0,0,0] → 仅 β 异构, 期望 forward 跟 baseline 不同 (因为 commit loss 改变)
    # 但 α=γ=0 时, encoder/codebook 行为应仅在 commit loss 项不同 (loss 数字不同, 但 argmin 路径相同)
    print()
    print("=" * 60)
    print("Test 2: β_l=[0.1,0.3,0.5], α_l=[0,0,0], γ_l=[0,0,0]")
    print("=" * 60)
    wrapper2 = make_wrapper_hrqvae(
        data, args.layers, args.num_emb_list,
        e_dim=args.e_dim,
        beta_list=[0.1, 0.3, 0.5],
        alpha_list=[0.0]*len(args.num_emb_list),
        gamma_list=[0.0]*len(args.num_emb_list),
        seed=args.seed, beta_baseline=args.beta_baseline)
    baseline.eval()
    wrapper2.eval()
    with torch.no_grad():
        out_b, rq_b, idx_b, _, _ = baseline(x, use_sk=False)
        out_w, rq_w, idx_w, _, _ = wrapper2(x, use_sk=False)
    out_diff2 = (out_b - out_w).abs().max().item()
    rq_diff2 = (rq_b - rq_w).abs().max().item()
    idx_eq2 = bool(torch.equal(idx_b, idx_w))
    print(f"[Test2] out_max_diff = {out_diff2:.6e}")
    print(f"[Test2] rq_loss_max_diff = {rq_diff2:.6e}")
    print(f"[Test2] idx_equal = {idx_eq2}")
    print(f"[Test2] baseline rq_loss = {rq_b.item():.6f}, wrapper rq_loss = {rq_w.item():.6f}")
    # 注: β 改变会改变 commit loss 数值, 但不影响 argmin 路径 (Sinkhorn 关闭)
    test2_pass = (out_diff2 == 0.0) and idx_eq2  # argmin 路径一致 → indices 一致 → out 一致
    print(f"[Test2] {'✅ PASS (β 异构但 indices 一致)' if test2_pass else '❌ FAIL'}")

    # Test 3: β_l=[0.5,0.5,0.5], α_l=[0.01,0.005,0.001], γ_l=[0,0,0] → 仅 anchor
    print()
    print("=" * 60)
    print("Test 3: β_l=[0.5,0.5,0.5], α_l=[0.01,0.005,0.001], γ_l=[0,0,0]")
    print("=" * 60)
    wrapper3 = make_wrapper_hrqvae(
        data, args.layers, args.num_emb_list,
        e_dim=args.e_dim,
        beta_list=[args.beta_baseline]*len(args.num_emb_list),
        alpha_list=[0.01, 0.005, 0.001],
        gamma_list=[0.0]*len(args.num_emb_list),
        seed=args.seed, beta_baseline=args.beta_baseline)
    wrapper3.eval()
    with torch.no_grad():
        out_w3, rq_w3, idx_w3, _, de_w3 = wrapper3(x, use_sk=False)
    anchor_loss_w3 = de_w3[2].item() if de_w3[2] is not None else 0.0
    print(f"[Test3] anchor_loss (α_l) = {anchor_loss_w3:.6f}")
    print(f"[Test3] {'✅ PASS (anchor 正则项非零)' if anchor_loss_w3 > 0 else '❌ FAIL'}")
    test3_pass = anchor_loss_w3 > 0

    # Test 4: β_l=[0.5,0.5,0.5], α_l=[0,0,0], γ_l=[0.001,0.0005,0.0001] → 仅 encoder L2
    print()
    print("=" * 60)
    print("Test 4: β_l=[0.5,0.5,0.5], α_l=[0,0,0], γ_l=[0.001,0.0005,0.0001]")
    print("=" * 60)
    wrapper4 = make_wrapper_hrqvae(
        data, args.layers, args.num_emb_list,
        e_dim=args.e_dim,
        beta_list=[args.beta_baseline]*len(args.num_emb_list),
        alpha_list=[0.0]*len(args.num_emb_list),
        gamma_list=[0.001, 0.0005, 0.0001],
        seed=args.seed, beta_baseline=args.beta_baseline)
    wrapper4.eval()
    with torch.no_grad():
        out_w4, rq_w4, idx_w4, _, _ = wrapper4(x, use_sk=False)
    rq_diff4 = (rq_b - rq_w4).abs().max().item()
    print(f"[Test4] rq_loss 增量 (γ_l) = {rq_w4.item() - rq_b.item():.6f}")
    print(f"[Test4] {'✅ PASS (encoder L2 正则生效)' if rq_w4.item() > rq_b.item() else '❌ FAIL'}")
    test4_pass = rq_w4.item() > rq_b.item()

    # 汇总
    print()
    print("=" * 60)
    print("Gate 0 reg test 汇总")
    print("=" * 60)
    print(f"Test 1 (β 全 0.5, α=γ=0):  {'✅ PASS' if test1_pass else '❌ FAIL'} (核心 reg test)")
    print(f"Test 2 (β 异构, α=γ=0):    {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (α 异构, β=γ=0):    {'✅ PASS' if test3_pass else '❌ FAIL'}")
    print(f"Test 4 (γ 异构, β=α=0):    {'✅ PASS' if test4_pass else '❌ FAIL'}")
    overall_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print()
    print(f"{'✅ Gate 0 PASS' if overall_pass else '❌ Gate 0 FAIL'}")
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())