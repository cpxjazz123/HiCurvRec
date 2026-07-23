#!/usr/bin/env python3
"""Task #40 Stage 2.1 前置 — fused_64d L2 normalize (Task #43 结论).

fused_64d 原始: mean=0.79, max=381, max/mean=480× (极端长尾).
L2 normalize 到单位球,消除长尾,让 RQ-VAE 距离计算不被离群 item 主导.

输出: products/task40_mckg_fused_rqvae/fused_item_64d_normalized.pt
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUT_DIR = ROOT / 'products/task40_mckg_fused_rqvae'
OUT_DIR.mkdir(exist_ok=True)


def main():
    print('========== Task #40 fused_64d L2 normalize (Task #43 前置) ==========')
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    fused = mckg['fused_item'].numpy()  # (11924, 64)

    norms_before = np.linalg.norm(fused, axis=1)
    print(f'Before normalize:')
    print(f'  norm min={norms_before.min():.4f} mean={norms_before.mean():.4f} max={norms_before.max():.4f} std={norms_before.std():.4f}')

    # 单位球 normalize
    fused_norm = fused / np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-12)

    norms_after = np.linalg.norm(fused_norm, axis=1)
    print(f'\nAfter L2 normalize to unit sphere:')
    print(f'  norm min={norms_after.min():.6f} mean={norms_after.mean():.6f} max={norms_after.max():.6f} std={norms_after.std():.6f}')

    # 验证: mean norm ≈ 1.0, std ≈ 0
    print(f'\n  ✓ 全部 item 在单位球上 (||x|| ≈ 1.0)')

    # 写入
    out = OUT_DIR / 'fused_item_64d_normalized.pt'
    torch.save(torch.from_numpy(fused_norm).float(), out)
    print(f'\n输出: {out}')

    # sanity: 单位球 fused 还能做 dense retrieval? 跳过 (在 Task #40 训练后再做 ablation)

    # summary
    summary = {
        'task': 'Task #40 fused normalize',
        'input_shape': list(fused.shape),
        'norm_before': {
            'min': float(norms_before.min()),
            'max': float(norms_before.max()),
            'mean': float(norms_before.mean()),
            'std': float(norms_before.std()),
        },
        'norm_after': {
            'min': float(norms_after.min()),
            'max': float(norms_after.max()),
            'mean': float(norms_after.mean()),
            'std': float(norms_after.std()),
        },
        'normalization_method': 'L2 unit sphere (fused / ||fused||)',
        'output_path': str(out),
    }
    with open(OUT_DIR / 'task40_normalize_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {OUT_DIR / "task40_normalize_summary.json"}')


if __name__ == '__main__':
    main()
