"""Task #144 Stage 4 — Test evaluation on best T5-small ckpt for both arms.
Loads HG_Rec_best.pth from products/task144/ckpt_arm_X/, evaluates on test set,
saves R@5/10/20 + NDCG@5/10/20 to verdict-ready JSON.

Usage:
    python3 scripts/task144_stage4_eval.py --arm A
    python3 scripts/task144_stage4_eval.py --arm B
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s3_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s3_mod = _ilu.module_from_spec(_s3_spec)
_s3_spec.loader.exec_module(_s3_mod)
evaluate = _s3_mod.evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    ckpt_glob = f"/home/wlia0047/ar57/wenyu/GeneRec/products/task144/ckpt_arm_{args.arm}/Instruments/*/HG_Rec_best.pth"
    candidates = sorted(glob.glob(ckpt_glob))
    if not candidates:
        raise FileNotFoundError(f"❌ No HG_Rec_best.pth in {ckpt_glob}")
    best_ckpt = candidates[-1]
    print(f"[Stage 4 Arm {args.arm}] Loading best ckpt: {best_ckpt}")

    config = {
        'batch_size': 256,
        'infer_size': 96,
        'lr': 1e-4,
        'device': args.device,
        'num_layers': 6,
        'num_decoder_layers': 4,
        'd_model': 128,
        'd_ff': 1024,
        'num_heads': 6,
        'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'feed_forward_proj': 'relu',
        'max_len': 20,
        'dataset_name': 'Instruments',
        'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
        'codebook_size': [64, 128, 256, 1],
        'code_path': f'_t5_hrqvae_kappa_decouple_arm_{args.arm}.npy',
        'topk_list': [5, 10, 20],
        'beam_size': 20,
    }

    device = torch.device(args.device)
    model = HG_Rec(config)
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size=config['codebook_size'],
        max_len=config['max_len']
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
    print(f"[Stage 4 Arm {args.arm}] Test dataset size: {len(test_dataset)}")

    avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

    print(f"[Stage 4 Arm {args.arm}] Test recalls: {avg_recalls}")
    print(f"[Stage 4 Arm {args.arm}] Test ndcgs: {avg_ndcgs}")

    result_json = f"/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task144_arm_{args.arm}_test_metrics.json"
    os.makedirs(os.path.dirname(result_json), exist_ok=True)
    with open(result_json, 'w') as f:
        json.dump({
            'arm': args.arm,
            'best_ckpt': best_ckpt,
            'test_recalls': avg_recalls,
            'test_ndcgs': avg_ndcgs,
        }, f, indent=2)
    print(f"[Stage 4 Arm {args.arm}] Saved: {result_json}")


if __name__ == "__main__":
    main()