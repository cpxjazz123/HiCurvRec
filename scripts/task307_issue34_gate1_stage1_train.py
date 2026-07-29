#!/usr/bin/env python3
"""Task #307 / Issue #34 / D9 — Gate 1 Stage 1 100 epoch training wrapper

(per-layer 异构 hash 函数族 on #30 GO 配置 r_l=[0.1,1,10] + s_l=[2,2,2])

继承 task301 Issue #30 的 in-place per-layer transforms 模式 (直接修改 embeddings.weight).
per-layer hash 函数族仅在 evaluation 阶段 (eval_step) 用于 Gate 1 (f) 验证,
不影响 forward training loss (保留 hard argmin commitment).

关键 R11.5 决策:
  - per-layer transforms: r_l=[0.1, 1, 10] + s_l=[2, 2, 2] (沿用 #30 极端值, in-place)
  - per-layer hash: 仅 eval 阶段 (step2 monitor) 用 wrapper class 验证 candidates
  - β=0.5 (与 #30 一致)
  - c_k: baseline U(0.5, 5) (不动, 不走 c_k range 双轴)
  - K=[64, 128, 256] (baseline)
  - seed=42 (单 seed, R11.5 禁 multi-seed)
  - hard argmin commitment (与 #30 一致, 不引入 expected-loss 形式)

注: 训练期间 hash 仅作 metadata 记录 (不修改 forward gradient path),
hash_top_k 硬编码 L0=3/L1=5/L2=7 (issue #34 body §3 explicit),
仅在 Gate 1 (f) 验证阶段用 wrapper class 加载.
"""
import sys, os

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, os.path.join(REPO, 'HG-Rec'))
sys.path.insert(0, os.path.join(REPO, 'HG-Rec', 'model'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import train_hrqvae as _baseline


def apply_per_layer_codebook_transforms(model, radius_list, scale_list):
    """In-place per-layer 几何变换 (沿用 #30 task301 模式, 直接修改 weight)."""
    n_layers = len(model.num_emb_list)
    assert len(radius_list) == n_layers
    assert len(scale_list) == n_layers

    import torch
    for li, q in enumerate(model.hrq.vq_layers):
        r = radius_list[li]
        s = scale_list[li]
        e_dim = q.embeddings.weight.shape[-1]
        device = q.embeddings.weight.device
        dtype = q.embeddings.weight.dtype

        # Issue #30 模式: weight = weight @ eff.T  (eff = (s · r) · I = (s · r) · eye)
        eff = (s * r) * torch.eye(e_dim, device=device, dtype=dtype)
        with torch.no_grad():
            q.embeddings.weight.data = q.embeddings.weight.data @ eff.t()

        eff_norm = eff.norm().item()
        weight_norm_after = q.embeddings.weight.norm().item()
        print(f'[TASK307] Layer {li}: r_l={r}, s_l={s}, R_l=I, eff_norm={eff_norm:.4f}, weight_norm_after={weight_norm_after:.4f}')


def main():
    args = _baseline.parse_args()

    if args.epochs == 1000:
        args.epochs = 100

    import torch
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from model.utils import EmbDataset
    from model.hrqvae import HRQVAE
    from model.hrqvae_trainer import Trainer

    seed = 42
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'[TASK307 GATE1] per-layer hash (eval-only), hash_top_k=[3,5,7] hardcoded per Issue #34 §3')

    data = EmbDataset(args.data_path)
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
    )

    # In-place per-layer transforms (沿用 #30 task301 模式)
    radius_list = [0.1, 1.0, 10.0]
    scale_list = [2.0, 2.0, 2.0]
    apply_per_layer_codebook_transforms(model, radius_list, scale_list)

    print(f'[TASK307 GATE1] data dim = {data.dim}, dataset size = {len(data)}')

    data_loader = DataLoader(
        data, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))

    print(f'[TASK307 GATE1] 启动 Stage 1 100 epoch 训练 (per-layer transform in-place + #30 r_l/s_l GO 配置)')
    print(f'[TASK307 GATE1] Hash candidates 验证 (Gate 1 (f)) 将在 eval 阶段通过 hash wrapper 验证')
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print(f'[TASK307 GATE1] Best Loss = {best_loss}')
    print(f'[TASK307 GATE1] Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()
