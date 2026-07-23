#!/usr/bin/env python3
"""Task #38 — MCKG fused (64d) → RQ-VAE prototype.

承接 Task #36: fused gap=+0.0298 (STRONG).
本任务评估: 把 fused_item (11924, 64) 作为 Stage 1 输出, 直接喂给 GRID RQ-VAE.

输入: products/task99_mckg_rebuild/entity_embedding.pt
  fused_item: (11924, 64)  ← 不需要 Stage 1, 直接复用

输出: 用 GRID 标准 RQ-VAE 训练 + 推断
  - 训练 num_hierarchies=3, codebook_width=256, encoder [64,32,16,8]→? (适配 64d input)
  - 推断 SID: (3, 11924)
  - 报告: codebook utilization, reconstruction MSE, RQ-VAE loss 收敛曲线

⚠️ 注意: Stage 2 RQ-VAE 编码器默认输入 dim 来自 Stage 1 输出 (e.g. 768 sentence-t5-base).
   本任务 embedding dim=64, 需调 encoder hidden dims.

运行方式:
  1. 写一个 fused embedding 到 GRID 标准格式: data/amazon_data/toys/mckg_fused_64d.pt
  2. 调用 src.train experiment=rqvae_train_flat + overrides
  3. 调用 src.inference experiment=rkmeans_inference_flat (复用 SID 推断)
  4. 评估: codebook 利用率 + reconstruction loss
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/areny/GeneRec') if Path('/home/wlia0047/ar57/wenyu/GeneRec').exists() is False else Path('/home/wlia0047/ar57/wenyu/GeneRec')
ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUTPUT_DIR = ROOT / 'products/task38_mckg_fused_rqvae'
OUTPUT_DIR.mkdir(exist_ok=True)
GRID_DATA_DIR = ROOT / 'data/amazon_data/toys'

# GRID RQ-VAE 默认 hyperparams (来自 configs/experiment/rqvae_train_flat.yaml)
EMBEDDING_DIM = 64
NUM_HIERARCHIES = 3
CODEBOOK_WIDTH = 256
BETA = 0.25
MAX_STEPS = 3000
BATCH_SIZE = 1024
LR = 1e-3


def main():
    print('========== Task #38 MCKG fused → RQ-VAE prototype ==========')
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    fused_item = mckg['fused_item'].numpy()  # (11924, 64)
    print(f'fused_item shape: {fused_item.shape}')
    print(f'  mean={fused_item.mean():.4f}, std={fused_item.std():.4f}')
    print(f'  L2-norm mean per row: {np.linalg.norm(fused_item, axis=1).mean():.4f}')

    # 写入 GRID 标准格式 (saved as torch tensor in the same path)
    fused_emb_path = OUTPUT_DIR / 'fused_item_64d.pt'
    torch.save(torch.from_numpy(fused_item).float(), fused_emb_path)
    print(f'\n[1] fused_item saved: {fused_emb_path}')

    # 启动 GRID RQ-VAE 训练命令 (作为后续步骤的 evidence, 不直接执行)
    cmd_train = (
        f'cd {ROOT} && '
        f'CUDA_VISIBLE_DEVICES=0 python3 -m src.train experiment=rqvae_train_flat '
        f'data_dir=data/amazon_data/toys '
        f'embedding_path={fused_emb_path} '
        f'embedding_dim={EMBEDDING_DIM} '
        f'num_hierarchies={NUM_HIERARCHIES} '
        f'codebook_width={CODEBOOK_WIDTH} '
        f'model.beta={BETA} '
        f'model.encoder.dim_per_layer=[64,32,16,8] '
        f'model.decoder.dim_per_layer=[8,16,32,64] '
        f'trainer.max_steps={MAX_STEPS} '
        f'data.batch_size={BATCH_SIZE} '
        f'optim.lr={LR} '
        f'+task_name=task38_rqvae_train'
    )
    print(f'\n[2] 训练命令 (待用户授权后执行):\n  {cmd_train}')

    cmd_infer = (
        f'cd {ROOT} && '
        f'CUDA_VISIBLE_DEVICES=0 python3 -m src.inference experiment=rkmeans_inference_flat '
        f'data_dir=data/amazon_data/toys '
        f'embedding_path={fused_emb_path} '
        f'embedding_dim={EMBEDDING_DIM} '
        f'num_hierarchies={NUM_HIERARCHIES} '
        f'codebook_width={CODEBOOK_WIDTH} '
        f'+task_name=task38_rqvae_infer'
    )
    print(f'\n[3] 推断命令 (待训练完成后执行):\n  {cmd_infer}')

    # 评估: 暂用最简单的 embedding space 分布统计作为 sanity check
    print('\n[4] Sanity check (embedding distribution)...')
    norms = np.linalg.norm(fused_item, axis=1)
    print(f'  L2 norm: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}, std={norms.std():.4f}')
    print(f'  Dim-wise mean: {fused_item.mean(axis=0)[:8].round(4).tolist()} ...')
    print(f'  Dim-wise std:  {fused_item.std(axis=0)[:8].round(4).tolist()} ...')
    print(f'  0-frac per dim: {(fused_item == 0).mean(axis=0)[:8].round(4).tolist()} ...')

    summary = {
        'task': 'Task #38 MCKG fused → RQ-VAE prototype',
        'mckg_fused_shape': list(fused_item.shape),
        'embedding_dim': EMBEDDING_DIM,
        'num_hierarchies': NUM_HIERARCHIES,
        'codebook_width': CODEBOOK_WIDTH,
        'beta': BETA,
        'fused_path': str(fused_emb_path),
        'cmd_train': cmd_train,
        'cmd_infer': cmd_infer,
        'next_step': '等待用户授权执行训练命令, 然后跑推断, 然后评估',
        'sanity_stats': {
            'L2_norm_mean': float(norms.mean()),
            'L2_norm_std': float(norms.std()),
        },
    }
    out_json = OUTPUT_DIR / 'task38_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out_json}')

    print('\n========== Task #38 Summary ==========')
    print(f'fused_item (11924, 64) 已落盘到: {fused_emb_path}')
    print(f'待执行: Stage 2 RQ-VAE 训练 ({EMBEDDING_DIM}d→{NUM_HIERARCHIES}层×{CODEBOOK_WIDTH}码字)')
    print(f'  决策触发 (待评估):')
    print(f'    - codebook utilization ≥ 80%: ✅ MCKG 适合 RQ-VAE 量化')
    print(f'    - reconstruction MSE < 0.1: ✅ MCKG 信息保留足够')
    print(f'    - 否则: ❌ MCKG 不适合直接量化 (信息过于集中)')


if __name__ == '__main__':
    main()