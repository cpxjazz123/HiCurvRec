#!/usr/bin/env python3
# Task #136 TIGER seed=2025 — standalone test eval (final)
import os
import sys
import json
import logging
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))
sys.path.insert(0, str(REPO / "scripts"))

import torch

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.HG_Rec import HG_Rec
from task84_hgrec_stage3_train import evaluate

# Match task136 launcher config inline (HG_Rec takes vars(args) config dict)
DATASET = "Instruments"
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/"
CODE_PATH_SUFFIX = "_curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy"
CODE_PATH = os.path.join(DATASET_PATH, DATASET, DATASET + CODE_PATH_SUFFIX)
CONFIG = {
    'dataset_name': DATASET,
    'dataset_path': DATASET_PATH,
    'code_path': CODE_PATH_SUFFIX,
    'codebook_size': [64, 128, 256, 1],
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'vocab_size': 1025,
    'max_len': 20,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'dropout_rate': 0.0,
    'feed_forward_proj': 'relu',
    'device': 'cuda',
    'mode': 'eval',
    'seed': 2025,
    'beam_size': 20,
    'infer_size': 96,
    'topk_list': [5, 10, 20],
    'early_stop': 20,
    'batch_size': 256,
}
INFER_SIZE = 96
BEAM_SIZE = 20
TOPK_LIST = [5, 10, 20]
DEVICE = 'cuda:0'
CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task136/ckpt_tiger_seed2025/Instruments/Jul-24-2026_13-02-48/HG_Rec_best.pth"
OUTPUT_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task136_tiger_seed2025_test_eval.json"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)
    device = torch.device(DEVICE)
    log.info(f"Loading ckpt: {CKPT_PATH}")

    test_dataset = GenRecDataset(
        os.path.join(DATASET_PATH, DATASET, 'test.parquet'),
        CODE_PATH, mode='evaluation',
        codebook_size=CONFIG['codebook_size'], max_len=CONFIG['max_len'],
        PAD_TOKEN=CONFIG['pad_token_id'],
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=INFER_SIZE, shuffle=False)
    log.info(f"Test dataset: {len(test_dataset)} samples")

    model = HG_Rec(CONFIG).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    log.info(f"Model loaded from {CKPT_PATH}")

    log.info("Running test eval ...")
    test_avg_recalls, test_avg_ndcgs = evaluate(
        model, test_dataloader, TOPK_LIST, BEAM_SIZE, device,
    )
    log.info(f"Test Dataset: {test_avg_recalls}")
    log.info(f"Test Dataset: {test_avg_ndcgs}")

    result = {
        'task': 'Task #136 TIGER seed=2025 test eval (standalone)',
        'ckpt_path': CKPT_PATH,
        'test_recalls': test_avg_recalls,
        'test_ndcgs': test_avg_ndcgs,
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Saved: {OUTPUT_JSON}")
    log.info(f"R@10 = {test_avg_recalls.get('Recall@10', 'N/A')}")
    log.info(f"NDCG@10 = {test_avg_ndcgs.get('NDCG@10', 'N/A')}")

if __name__ == '__main__':
    main()