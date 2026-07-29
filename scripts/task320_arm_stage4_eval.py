#!/usr/bin/env python3
"""
Task #320 — Stage 4 K=100 eval per arm (Issue #38 5-arm retraining)
Generic Stage 4 eval that loads a task320 arm ckpt and runs on test set with beam=100
(Issue #30 ε ceiling decision threshold)
"""
import sys, os, json, time, argparse

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')
sys.path.insert(0, f'{REPO}/scripts')

import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('task84_s3_train', f'{REPO}/scripts/task84_hgrec_stage3_train.py')
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evaluate = _mod.evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['A', 'B', 'C', 'D', 'E'])
    p.add_argument('--ckpt_path', required=True)
    p.add_argument('--beam_size', type=int, default=100)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--task_id', type=int, default=320)
    p.add_argument('--code_path', default='_t5_hrqvae_issue30_per_layer_transforms.npy')
    args = p.parse_args()

    config = {
        'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': f'cuda:{args.gpu}',
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
        'feed_forward_proj': 'relu', 'max_len': 20,
        'dataset_name': 'Instruments',
        'dataset_path': f'{REPO}/HG-Rec/dataset/',
        'codebook_size': [64, 128, 256, 1],
        'code_path': args.code_path,
        'topk_list': [5, 10, 20],
        'beam_size': args.beam_size,
    }

    device = torch.device(f'cuda:{args.gpu}')
    model = HG_Rec(config)

    if not os.path.exists(args.ckpt_path):
        print(f"ERROR: no ckpt at {args.ckpt_path}", flush=True)
        sys.exit(1)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f'[TASK{args.task_id}/Arm{args.arm}] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
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
        'task': f'task{args.task_id}_arm{args.arm}_stage4_eval_beam{args.beam_size}',
        'arm': args.arm,
        'ckpt_path': args.ckpt_path,
        'code_path': args.code_path,
        'beam_size': args.beam_size,
        'elapsed_sec': elapsed,
        **{f'test_{k}': v for k, v in avg_recalls.items()},
        **{f'test_{k}': v for k, v in avg_ndcgs.items()},
    }
    out_dir = f'{REPO}/verdicts'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'task{args.task_id}_arm{args.arm}_beam{args.beam_size}_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'[TASK{args.task_id}/Arm{args.arm}] metrics -> {out_path}')
    print(f'[TASK{args.task_id}/Arm{args.arm}] elapsed={elapsed:.1f}s')
    print(f'[TASK{args.task_id}/Arm{args.arm}] Recall@5/10/20 = {[avg_recalls["Recall@5"], avg_recalls["Recall@10"], avg_recalls["Recall@20"]]}')
    print(f'[TASK{args.task_id}/Arm{args.arm}] NDCG@5/10/20 = {[avg_ndcgs["NDCG@5"], avg_ndcgs["NDCG@10"], avg_ndcgs["NDCG@20"]]}')


if __name__ == '__main__':
    main()
