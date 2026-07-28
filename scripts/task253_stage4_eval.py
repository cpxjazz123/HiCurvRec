#!/usr/bin/env python3
"""Task #253 Stage 4 — Möbius residual SID test eval.

Fork of task137_test_eval_free_curv.py, adapted for Task #253 mobius_residual SID
(_t5_rqvae_mobius_residual_issue13_gate2.npy).

R11.3 自主决策:
- ckpt_path: Task #253 Stage 3 best_ckpt (HG_Rec_best.pth)
- code_suffix: _t5_rqvae_mobius_residual_issue13_gate2
- output_json: verdicts/task253_stage4_eval.json
- CONFIG: 复用 task137 内联 CONFIG
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
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
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
                   help="e.g. _t5_rqvae_mobius_residual_issue13_gate2.npy")
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

    # HG_Rec config (HG_Rec.py 完整签名)
    model = HG_Rec(config={
        "num_layers": 6,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "vocab_size": 1025,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "feed_forward_proj": "gated-gelu",
    })
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    log.info("Model loaded")

    code_path = os.path.join(DATASET_PATH, DATASET + args.code_suffix)
    log.info(f"Code path: {code_path}")

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(DATASET_PATH, "test.parquet"),
        code_path=code_path,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN,
    )
    test_loader = GenRecDataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    log.info(f"Test dataset: {len(test_dataset)} samples")

    metrics = evaluate(model, test_loader, TOPK_LIST, BEAM_SIZE, device)
    # metrics = (recall_dict, ndcg_dict)
    recall_dict, ndcg_dict = metrics
    log.info(f"Recall: {recall_dict}")
    log.info(f"NDCG: {ndcg_dict}")

    def _to_float(v):
        return v.item() if hasattr(v, 'item') else v

    out = {
        "recall": {k: _to_float(v) for k, v in recall_dict.items()},
        "ndcg": {k: _to_float(v) for k, v in ndcg_dict.items()},
    }
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()