#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task53_log1p_preprocess.py — L1 log1p 后处理 (Task #52 G1 verdict 推荐路线)

公式: e' = e / ||e|| · log(1 + ||e||)
  - 归一化方向 (e / ||e||): 保留语义角度
  - log(1+||e||) 压缩: 把 norm 长尾压缩, 避免 RQ 拆分放大 norm² loss

参照:
  - verdicts/task52_g1_verdict_result.md / task52_final_verdict.md
  - Task #51 Stage 2 proxy 验证: SCR 4.22x → 0.31x, RQ recon 12.72 → 0.05

执行:
  python3 scripts/task53_log1p_preprocess.py \
      --input products/task48_s4_ae/entity_embedding.pt \
      --output products/task53_log1p_s4_ae/entity_embedding.pt
"""
import argparse
import json
import os
from pathlib import Path

import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')


def l1_log1p_normalize(emb: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """L1 log1p 后处理: e' = e / ||e||_2 · log(1 + ||e||_2)

    等价于: 沿 dim=-1 先 L2-normalize 再乘 log(1+原始 norm)
    - 输入: (N, D)
    - 输出: (N, D), 形状不变, 方向相同, norm 被压缩
    """
    norm = torch.norm(emb, p=2, dim=-1, keepdim=True)  # (N, 1)
    unit = emb / (norm + eps)                            # 方向
    scale = torch.log1p(norm)                            # 压缩因子
    return unit * scale


def main():
    parser = argparse.ArgumentParser(description='L1 log1p post-processing for S4 AE embedding')
    parser.add_argument('--input', required=True, help='input .pt path (e.g. products/task48_s4_ae/entity_embedding.pt)')
    parser.add_argument('--output', required=True, help='output .pt path')
    parser.add_argument('--summary', default=None, help='optional JSON summary path')
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f'输入 embedding 不存在: {input_path}')

    print(f'[task53_log1p_preprocess] 输入: {input_path}')
    print(f'[task53_log1p_preprocess] 输出: {output_path}')

    emb = torch.load(input_path, weights_only=False, map_location='cpu')

    # 处理 dict 容器 (S4 AE 输出是 {subspace_item, fused_item, method, config, trajectory})
    if isinstance(emb, dict):
        if 'fused_item' not in emb:
            raise KeyError(f'输入 dict 缺 fused_item 键: keys={list(emb.keys())}')
        fused = emb['fused_item']
        if torch.is_tensor(fused):
            fused = fused.float()
        else:
            fused = torch.from_numpy(fused).float()
        emb = fused
        print(f'[task53_log1p_preprocess] 从 dict 提取 fused_item')
    elif not torch.is_tensor(emb):
        raise TypeError(f'期望 torch.Tensor 或 dict, 实际 {type(emb)}')
    else:
        emb = emb.float()

    print(f'[task53_log1p_preprocess] 原始 shape={tuple(emb.shape)}, dtype={emb.dtype}')
    pre_norm = torch.norm(emb, p=2, dim=-1)
    print(f'[task53_log1p_preprocess] 原始 norm: min={pre_norm.min().item():.4f}, '
          f'max={pre_norm.max().item():.4f}, mean={pre_norm.mean().item():.4f}, '
          f'p99={torch.quantile(pre_norm, 0.99).item():.4f}')

    emb_log1p = l1_log1p_normalize(emb)
    post_norm = torch.norm(emb_log1p, p=2, dim=-1)
    print(f'[task53_log1p_preprocess] log1p 后 norm: min={post_norm.min().item():.4f}, '
          f'max={post_norm.max().item():.4f}, mean={post_norm.mean().item():.4f}, '
          f'p99={torch.quantile(post_norm, 0.99).item():.4f}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_log1p, output_path)
    print(f'[task53_log1p_preprocess] ✅ 已保存 {output_path}')

    if args.summary:
        summary_path = ROOT / args.summary
        summary = {
            'input': str(input_path),
            'output': str(output_path),
            'shape': list(emb.shape),
            'dtype': str(emb.dtype),
            'pre_norm': {
                'min': float(pre_norm.min()), 'max': float(pre_norm.max()),
                'mean': float(pre_norm.mean()), 'p99': float(torch.quantile(pre_norm, 0.99)),
            },
            'post_norm': {
                'min': float(post_norm.min()), 'max': float(post_norm.max()),
                'mean': float(post_norm.mean()), 'p99': float(torch.quantile(post_norm, 0.99)),
            },
            'norm_max_reduction_ratio': float(pre_norm.max() / post_norm.max()),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'[task53_log1p_preprocess] ✅ summary → {summary_path}')
        print(f'[task53_log1p_preprocess] norm_max 压缩比 = {summary["norm_max_reduction_ratio"]:.2f}x')


if __name__ == '__main__':
    main()