"""
Task #278 — 通用 Stage 4 R@10 eval driver
2026-07-29

Purpose: 批量 evaluate 任何 ckpt 在 Instruments test set 上的 R@5/10/20 + NDCG.
Usage: python3 task278_batch_stage4_eval.py --ckpt_path X --code_path _foo.npy --output_path Y

复用 task243 v2 修过的 exclude-start-token 逻辑 (跟 task84 evaluate() 一致).
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import random

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, os.path.join(REPO, "HG-Rec"))

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu(); labels = labels.detach().cpu()
    matches = (preds == labels.unsqueeze(1)).all(dim=2)
    return matches


def evaluate(model, dataset, beam_size=20, topk_list=(5, 10, 20), device='cuda:0'):
    loader = GenRecDataLoader(dataset, batch_size=128, shuffle=False)
    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        with torch.no_grad():
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds.view(input_ids.size(0), beam_size, -1)[:, :, 1:5]  # exclude start token (跟 task84 evaluate 一致)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    results = {}
    for k in topk_list:
        matches = (preds_all[:, :k] == labels_all.unsqueeze(1)).all(dim=2)
        results[f'Recall@{k}'] = matches.any(dim=1).float().mean().item()
    for k in topk_list:
        ndcg_list = []
        matches = (preds_all[:, :k] == labels_all.unsqueeze(1)).all(dim=2)
        for i in range(len(labels_all)):
            hits = matches[i]
            if not hits.any(): ndcg_list.append(0.0); continue
            first_hit = hits.nonzero()[0].item()
            ndcg_list.append(1.0 / np.log2(first_hit + 2))
        results[f'NDCG@{k}'] = float(np.mean(ndcg_list))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--code_path", required=True, help="e.g. _t5_rqvae_code_default.npy")
    p.add_argument("--codebook_size", default="64,128,256,1",
                   help="comma-separated, e.g. '32,64,256,1' (R11.5: 必须 match training config)")
    p.add_argument("--dataset_name", default="Instruments")
    p.add_argument("--dataset_path", default=f"{REPO}/HG-Rec/dataset/")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--beam_size", type=int, default=20)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--d_kv", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=1025)
    p.add_argument("--pad_token_id", type=int, default=0)
    p.add_argument("--eos_token_id", type=int, default=0)
    p.add_argument("--feed_forward_proj", default="relu")
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--dropout_rate", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", required=True)
    p.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()

    codebook_size = [int(x) for x in args.codebook_size.split(",")]
    print(f"[Task #278] codebook_size={codebook_size} (from CLI)")
    if codebook_size != [64, 128, 256, 1]:
        print(f"[Task #278] ⚠️  codebook_size override (R11.5: 必须 match training)")

    config = {
        'dataset_name': args.dataset_name, 'dataset_path': args.dataset_path,
        'code_path': args.code_path, 'codebook_size': codebook_size,
        'num_epochs': 200, 'batch_size': 256, 'lr': 1e-4,
        'num_layers': args.num_layers, 'num_decoder_layers': args.num_decoder_layers,
        'd_model': args.d_model, 'd_ff': args.d_ff, 'num_heads': args.num_heads, 'd_kv': args.d_kv,
        'vocab_size': args.vocab_size, 'pad_token_id': args.pad_token_id, 'eos_token_id': args.eos_token_id,
        'feed_forward_proj': args.feed_forward_proj, 'max_len': args.max_len, 'dropout_rate': args.dropout_rate,
        'device': args.device, 'mode': 'evaluation', 'log_path': '/tmp',
        'seed': args.seed, 'early_stop': 20, 'beam_size': args.beam_size, 'infer_size': 96,
        'save_path': '/tmp/task278',
    }

    print(f"[Task #278] ckpt={args.ckpt_path}")
    print(f"[Task #278] code_path={args.code_path}")

    set_seed(args.seed)
    model = HG_Rec(config).to(args.device)
    sd = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[Task #278] model loaded: {sum(p.numel() for p in model.parameters())} params")

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, args.dataset_name, 'test.parquet'),
        code_path=os.path.join(args.dataset_path, args.dataset_name, args.dataset_name + args.code_path),
        mode='evaluation', codebook_size=config['codebook_size'], max_len=args.max_len,
    )
    print(f"[Task #278] test_dataset: {len(test_dataset)} samples")

    results = evaluate(model, test_dataset, beam_size=args.beam_size, device=args.device)
    print(f"[Task #278] results: {results}")

    # Add metadata
    output = {
        'ckpt_path': args.ckpt_path,
        'code_path': args.code_path,
        'n_test_examples': len(test_dataset),
        'beam_size': args.beam_size,
        **results,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"[Task #278] saved to {args.output_path}")

    # GO/NO-GO
    HG_REC_BASELINE = 0.1020
    r10 = results.get('Recall@10', 0)
    if r10 > HG_REC_BASELINE:
        print(f"[Task #278] ✅ GO: R@10={r10:.4f} > HG-Rec baseline {HG_REC_BASELINE}")
    else:
        print(f"[Task #278] ❌ NO-GO: R@10={r10:.4f} <= HG-Rec baseline {HG_REC_BASELINE}")


if __name__ == '__main__':
    main()