#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #95 — HGN standalone test evaluation (paper Table 2 baseline #6).

HGN 训练 2026-07-22 00:44-00:48 被 SIGTERM 中断 (log 显示 epoch 4, 但 ckpt 是 epoch 3 best valid_score=0.0554).
test eval 当时没跑到. 现有 best ckpt: RecBole/saved/HGN-Jul-22-2026_00-44-41.pth.
paper HGN R@10=0.0960 (Instruments). 我们验证 vs paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer, ModelType, init_seed
from logging import getLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config_files', nargs='+',
                   default=['musical_instruments_sequential_paper.yaml'])
    p.add_argument('--ckpt',
                   default='/home/wlia0047/ar57/wenyu/GeneRec/RecBole/saved/HGN-Jul-22-2026_00-44-41.pth')
    p.add_argument('--gpu_id', default='0')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2025)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_files = [os.path.join('RecBole', f) for f in args.config_files]
    print(f"[Task #95 HGN standalone eval] loading from {args.ckpt}")
    print(f"  config_files: {config_files}")
    print(f"  gpu_id: {args.gpu_id}, seed={args.seed}")

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    init_seed(args.seed, True)

    # load_data_and_model returns 6 values: config, model, dataset, train_data, valid_data, test_data
    config, model, dataset, train_data, valid_data, test_data = \
        load_data_and_model(model_file=args.ckpt)

    print(f"[Task #95] model loaded: {model.__class__.__name__}")
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  test_data size: {len(test_data)}")

    # HGN 是 SEQUENTIAL model
    model_type = ModelType.SEQUENTIAL
    trainer = get_trainer(model_type, model_name=config['model'])(config, model)

    # 直接 evaluate test set
    print("\n[Test evaluation on best valid ckpt]")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    print(f"\n[Test Result]")
    for k, v in test_result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # paper HGN Instruments baseline (HG-Rec paper Table 1)
    paper_hgn = {
        'recall@5': 0.0781,
        'recall@10': 0.0960,
        'ndcg@5': 0.0654,
        'ndcg@10': 0.0712,
    }

    comparison = {}
    for k_paper, v_paper in paper_hgn.items():
        v_repro = float(test_result.get(k_paper, 0.0))
        delta_abs = v_repro - v_paper
        delta_pct = (delta_abs / v_paper) * 100 if v_paper > 0 else 0.0
        comparison[k_paper] = {
            'paper': v_paper,
            'repro': round(v_repro, 4),
            'delta_abs': round(delta_abs, 4),
            'delta_pct': round(delta_pct, 1),
        }
        print(f"  {k_paper}: paper={v_paper:.4f}, repro={v_repro:.4f}, "
              f"Δ={delta_abs:+.4f} ({delta_pct:+.1f}%)")

    # Output JSON
    output = {
        'task': 'Task #95 HGN standalone test eval',
        'ckpt': args.ckpt,
        'config_files': args.config_files,
        'gpu_id': args.gpu_id,
        'seed': args.seed,
        'test_result': {k: round(float(v), 4) if isinstance(v, float) else v for k, v in test_result.items()},
        'paper_hgn_instruments': paper_hgn,
        'comparison': comparison,
        'paper_source': 'HG-Rec ICML 2026 paper Table 1 Instruments HGN',
        'dataset': 'Musical_Instruments 5-core (data/recbole/Musical_Instruments/)',
        'eval_mode': 'full ranking over 24588 items (paper-aligned)',
        'valid_metric_at_best_epoch': 0.0554,  # epoch 3 from training log
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {args.output}")


if __name__ == '__main__':
    main()
