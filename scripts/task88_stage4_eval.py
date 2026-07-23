#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #88 — Stage 4 standalone evaluation: T5 ckpt -> test R@5/R@10/NDCG.

Loads T5 ckpt (state_dict only) + codebook, runs evaluate() on TEST set.
Mirror Stage 3 architecture params (T5-small 6 enc + 4 dec layers, d_model=128).

R88 (Task #88): standalone Stage 4 eval for each grid combo.
Uses test.parquet split + codebook .npy from Stage 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f'preds.shape[1]={preds.shape[1]}!={maxk}'
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def evaluate_t5_test(model, test_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            for k in topk_list:
                recalls['Recall@' + str(k)].append(recall_at_k(pos_index, k).mean().item())
                ndcgs['NDCG@' + str(k)].append(ndcg_at_k(pos_index, k).mean().item())
    return {k: sum(v) / len(v) for k, v in recalls.items()}, \
           {k: sum(v) / len(v) for k, v in ndcgs.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--curvature_tag', required=True,
                   help='e.g. curv_1.0_1.0_1.0 — match Stage 3 ckpt subdir naming')
    p.add_argument('--ckpt', default=None,
                   help='T5 ckpt path. Default: latest HG_Rec_best.pth in ckpt_hgrec/{dataset}/{tag}/*/')
    p.add_argument('--dataset_path', default='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/')
    p.add_argument('--dataset_name', default='Instruments')
    p.add_argument('--code_path_suffix', default=None,
                   help='code_path suffix for test eval. Default: auto-derived from curvature_tag')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--beam_size', type=int, default=20)
    p.add_argument('--max_len', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=96)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_decoder_layers', type=int, default=4)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--d_kv', type=int, default=64)
    p.add_argument('--vocab_size', type=int, default=1025)
    p.add_argument('--pad_token_id', type=int, default=0)
    p.add_argument('--eos_token_id', type=int, default=0)
    p.add_argument('--dropout_rate', type=float, default=0.1)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    import glob
    from data.dataset import GenRecDataset
    from data.dataloader import GenRecDataLoader
    from model.HG_Rec import HG_Rec

    args = parse_args()

    # Derive code_path_suffix if not given
    if args.code_path_suffix is None:
        # curv_1.0_1.0_1.0 -> _curv_1.0_1.0_1.0_t5_hrqvae_poincare.npy
        curv_str = args.curvature_tag.replace('curv_', '')
        args.code_path_suffix = f"_curv_{curv_str}_t5_hrqvae_poincare.npy"

    # Find ckpt (R88: Stage 3 ckpts are in timestamped subdirs, not curv_tag subdirs)
    if args.ckpt is None:
        ckpt_pattern = (
            f"/home/wlia0047/ar57/wenyu/GeneRec/products/task88/ckpt_hgrec/"
            f"{args.dataset_name}/*/HG_Rec_best.pth"
        )
        candidates = sorted(glob.glob(ckpt_pattern))
        if not candidates:
            raise FileNotFoundError(f"❌ No HG_Rec_best.pth in {ckpt_pattern}")
        # R88: when multiple ckpts exist, prefer the most recent one whose
        # Stage 3 log matches curvature_tag (heuristic: latest overall).
        # For correctness, caller should pass --ckpt explicitly when ambiguous.
        args.ckpt = candidates[-1]
        print(f"  [WARN] auto-selected latest ckpt {args.ckpt}; "
              f"if multiple Stage 3 jobs ran, pass --ckpt explicitly to disambiguate.")

    print(f"=== Task #88 Stage 4 eval [{args.curvature_tag}] ===")
    print(f"  ckpt: {args.ckpt}")
    print(f"  code_path_suffix: {args.code_path_suffix}")
    print(f"  device: {args.device}")

    device = torch.device(args.device)

    # Build T5 config from args (mirror Stage 3 invocation)
    config = {
        'num_layers': args.num_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_heads': args.num_heads,
        'd_kv': args.d_kv,
        'dropout_rate': args.dropout_rate,
        'vocab_size': args.vocab_size,
        'pad_token_id': args.pad_token_id,
        'eos_token_id': args.eos_token_id,
        'max_len': args.max_len,
        'beam_size': args.beam_size,
        'infer_size': args.batch_size,
        'codebook_size': [64, 128, 256, 1],
        'feed_forward_proj': 'relu',
    }
    model = HG_Rec(config).to(device)
    state_dict = torch.load(args.ckpt, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Test dataset
    test_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, args.dataset_name, 'test.parquet'),
        code_path=os.path.join(args.dataset_path, args.dataset_name, args.dataset_name + args.code_path_suffix),
        mode='evaluation',
        codebook_size=[64, 128, 256, 1],
        max_len=args.max_len,
    )
    test_loader = GenRecDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Eval
    recalls, ndcgs = evaluate_t5_test(model, test_loader, topk_list=[5, 10], beam_size=args.beam_size, device=device)
    print(f"\n[Test Result for {args.curvature_tag}]")
    for k, v in recalls.items():
        print(f"  {k}: {v:.4f}")
    for k, v in ndcgs.items():
        print(f"  {k}: {v:.4f}")

    # Save
    out = {
        'task': f'Task #88 Stage 4 eval [{args.curvature_tag}]',
        'curvature_tag': args.curvature_tag,
        'curvatures': [float(x) for x in args.curvature_tag.replace('curv_', '').split('_')],
        'ckpt': args.ckpt,
        'code_path_suffix': args.code_path_suffix,
        'test_result': {**recalls, **ndcgs},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {args.output}")


if __name__ == '__main__':
    main()