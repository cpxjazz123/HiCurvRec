#!/usr/bin/env python3
"""Task #318 Stage 4 eval — run on 4 optimizer ckpts at K=100"""

import sys, os, json, time
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import torch
import argparse
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s3_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s3_mod = _ilu.module_from_spec(_s3_spec)
_s3_spec.loader.exec_module(_s3_mod)
evaluate = _s3_mod.evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    config = {
        'batch_size': 256, 'infer_size': 96, 'lr': 1e-4,
        'device': f'cuda:{args.gpu}',
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
        'feed_forward_proj': 'relu', 'max_len': 20,
        'dataset_name': 'Instruments',
        'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
        'codebook_size': [64, 128, 256, 1],
        'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
        'topk_list': [5, 10, 20],
        'beam_size': args.beam_size,
    }

    device = torch.device(f'cuda:{args.gpu}')
    model = HG_Rec(config)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f'[task318/{args.optimizer} beam={args.beam_size}] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
    model.to(device); model.eval()

    test_ds = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size=config['codebook_size'],
        max_len=config['max_len'],
    )

    from transformers import T5Config
    tokenizer = T5Config(
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        d_kv=config['d_kv'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dropout_rate=config['dropout_rate'],
    )
    test_ds.tokenizer = tokenizer

    test_dl = GenRecDataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    t0 = time.time()
    avg_recalls, avg_ndcgs = evaluate(model, test_dl, config['topk_list'], config['beam_size'], device)
    elapsed = time.time() - t0

    out = {
        'task': f'task318_issue38_arm1_optimizer_{args.optimizer}_beam{args.beam_size}',
        'optimizer': args.optimizer,
        'ckpt_path': args.ckpt_path,
        'beam_size': args.beam_size,
        'elapsed_sec': elapsed,
        **{f'test_{k}': v for k, v in avg_recalls.items()},
        **{f'test_{k}': v for k, v in avg_ndcgs.items()},
    }
    metrics_dir = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts'
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, f'task318_{args.optimizer}_beam{args.beam_size}_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'[task318/{args.optimizer} beam={args.beam_size}] elapsed={elapsed:.1f}s', flush=True)
    print(f'  Recall@5/10/20 = [{avg_recalls["Recall@5"]:.4f}, {avg_recalls["Recall@10"]:.4f}, {avg_recalls["Recall@20"]:.4f}]')
    print(f'  NDCG@5/10/20 = [{avg_ndcgs["NDCG@5"]:.4f}, {avg_ndcgs["NDCG@10"]:.4f}, {avg_ndcgs["NDCG@20"]:.4f}]')
    print(f'  Metrics written to {out_path}')


if __name__ == '__main__':
    main()