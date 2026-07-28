#!/usr/bin/env python3
"""Task #149 Stage 4 — Heterogeneous κ HRQ-VAE (3 layers) test eval.

Fork of task137_test_eval_free_curv.py, adapted for Task #149 heterokappa codebook
(_t5_hrqvae_heterokappa.npy).

R11.3 自主决策:
- ckpt_path: from --ckpt_path arg (Stage 3 best NDCG@20 ckpt, currently epoch 49)
- code_suffix: _t5_hrqvae_heterokappa (Task #149 Stage 2 output, M=1 per layer,
  L0 κ=-0.127, L1 κ=0.0, L2 κ=+0.163)
- output_json: verdicts/task149_heterokappa_test_eval.json
- Goal: 验证 Goal #2 下游指标合理范围 (paper R@10=0.1315 ±25%)
- 中间版: 即使 Stage 3 后续变, 也能用现有 best ckpt 拿 test set 真实指标
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
    p.add_argument("--code_suffix", default="_t5_hrqvae_heterokappa",
                   help="default: _t5_hrqvae_heterokappa (Task #149)")
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
    log.info(f"Code suffix: {args.code_suffix}")

    code_path = os.path.join(DATASET_PATH, DATASET, DATASET + args.code_suffix + ".npy")
    log.info(f"Code file: {code_path}")

    CONFIG = {
        "dataset_name": DATASET,
        "dataset_path": DATASET_PATH,
        "code_path": args.code_suffix + ".npy",
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
        "batch_size": 256,
        "infer_size": INFER_SIZE,
        "lr": 1e-4,
        "topk_list": TOPK_LIST,
        "beam_size": BEAM_SIZE,
        "device": DEVICE,
    }

    # Load best ckpt
    model = HG_Rec(CONFIG)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    log.info("Ckpt loaded successfully")

    # Test dataset
    test_dataset = GenRecDataset(
        dataset_path=os.path.join(DATASET_PATH, DATASET, "test.parquet"),
        code_path=code_path,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=INFER_SIZE, shuffle=False)
    log.info(f"Test dataset size: {len(test_dataset)}")

    # Evaluate on TEST
    avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, TOPK_LIST, BEAM_SIZE, device)
    log.info(f"Test recalls: {avg_recalls}")
    log.info(f"Test ndcgs: {avg_ndcgs}")

    # JSON output
    result = {
        "best_ckpt": args.ckpt_path,
        "code_suffix": args.code_suffix,
        "test_recalls": avg_recalls,
        "test_ndcgs": avg_ndcgs,
        "paper_R@10": 0.1315,
        "paper_R@5": 0.0844,
        "paper_NDCG@10": 0.1074,
        "paper_NDCG@5": 0.0721,
        "paper_source": "HG-Rec paper Table 1 Instruments Musical_Instruments dataset",
        "goal2_target": "R@10 ∈ [0.0986, 0.1644] (±25% paper)",
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Test metrics saved: {args.output_json}")
    log.info("===== [Task #149 Stage 4] Done =====")


if __name__ == "__main__":
    main()