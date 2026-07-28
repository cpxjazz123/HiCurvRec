#!/usr/bin/env python3
"""Task #137 Stage 4 — FreeCurvHRQVAE codebook (curv0.5) test eval.

Fork of task136_tiger_test_eval.py, adapted for Task #137 curv0.5 codebook
(_t5_hrqvae_poincare_curv0.5.npy).

R11.3 自主决策:
- ckpt_path: from --ckpt_path arg (Stage 3 output)
- code_suffix: _t5_hrqvae_poincare_curv0.5 (Task #137 A-arm Stage 2 output)
- output_json: verdicts/task137_curv0_5_test_eval.json
- CONFIG: 复用 Task #136 内联 CONFIG (HG_Rec takes config dict)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))
sys.path.insert(0, str(REPO / "scripts"))

import torch

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.HG_Rec import HG_Rec
from task84_hgrec_stage3_train import evaluate


DATASET = "Instruments"
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
INFER_SIZE = 96
BEAM_SIZE = 20
TOPK_LIST = [5, 10, 20]
DEVICE = "cuda:0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--code_suffix", required=True,
                   help="e.g. _t5_hrqvae_poincare_curv0.5.npy")
    p.add_argument("--output_json", required=True)
    p.add_argument("--log_file", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(args.log_file),
                                  logging.StreamHandler()])
    log = logging.getLogger(__name__)
    device = torch.device(DEVICE)
    log.info(f"Loading ckpt: {args.ckpt_path}")

    code_path = os.path.join(DATASET_PATH, DATASET, DATASET + args.code_suffix)

    CONFIG = {
        "dataset_name": DATASET,
        "dataset_path": DATASET_PATH,
        "code_path": args.code_suffix,
        "codebook_size": CODEBOOK_SIZE,
        "num_layers": 6,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "vocab_size": 1025,
        "max_len": MAX_LEN,
        "pad_token_id": PAD_TOKEN,
        "eos_token_id": PAD_TOKEN,
        "dropout_rate": 0.0,
        "feed_forward_proj": "relu",
        "device": "cuda",
        "mode": "eval",
        "seed": 42,
        "beam_size": BEAM_SIZE,
        "infer_size": INFER_SIZE,
        "topk_list": TOPK_LIST,
        "early_stop": 20,
        "batch_size": 256,
    }

    test_dataset = GenRecDataset(
        os.path.join(DATASET_PATH, DATASET, "test.parquet"),
        code_path, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=INFER_SIZE, shuffle=False)
    log.info(f"Test dataset: {len(test_dataset)} samples")

    model = HG_Rec(CONFIG).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    log.info(f"Model loaded from {args.ckpt_path}")

    log.info("Running test eval ...")
    test_avg_recalls, test_avg_ndcgs = evaluate(
        model, test_dataloader, TOPK_LIST, BEAM_SIZE, device,
    )
    log.info(f"Test Dataset: {test_avg_recalls}")
    log.info(f"Test Dataset: {test_avg_ndcgs}")

    result = {
        "task": "Task #137 A-arm (curv0.5) Stage 4 test eval",
        "ckpt_path": args.ckpt_path,
        "code_suffix": args.code_suffix,
        "test_recalls": test_avg_recalls,
        "test_ndcgs": test_avg_ndcgs,
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Saved: {args.output_json}")
    log.info(f"R@10 = {test_avg_recalls.get('Recall@10', 'N/A')}")
    log.info(f"NDCG@10 = {test_avg_ndcgs.get('NDCG@10', 'N/A')}")


if __name__ == "__main__":
    main()