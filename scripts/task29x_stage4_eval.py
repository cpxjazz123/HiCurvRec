#!/usr/bin/env python3
"""Task #290/#291/#292 Stage 4 — Parameterized Test evaluation.

Loads best ckpt, evaluates on TEST split, saves JSON to {output_json}.

Usage:
    python3 scripts/task29x_stage4_eval.py \
        --task_id 290 --stage3_ckpt <path> --code_path <path> \
        --output_json <path>
"""
import argparse
import importlib.util as ilu
import json
import os
import sys

import torch

HGREC = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec"
sys.path.insert(0, HGREC)

# Reuse HG-Rec fork's evaluate/recall/ndcg functions (R11.3 决策: 复用代码 vs 复制)
_s3_spec = ilu.spec_from_file_location(
    "task84_s3_train_fork",
    "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py",
)
_s3_mod = ilu.module_from_spec(_s3_spec)
_s3_spec.loader.exec_module(_s3_mod)
evaluate = _s3_mod.evaluate

from model.HG_Rec import HG_Rec  # noqa: E402
from data.dataset import GenRecDataset  # noqa: E402
from data.dataloader import GenRecDataLoader  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True,
                        help="绝对路径或相对 HG-Rec/dataset/Instruments/<basename> 的 filename")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/")
    parser.add_argument("--dataset_name", type=str, default="Instruments")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--beam_size", type=int, default=20)
    args = parser.parse_args()

    config = {
        "batch_size": 256,
        "infer_size": args.batch_size,
        "lr": 1e-4,
        "device": args.device,
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
        "feed_forward_proj": "relu",
        "max_len": 20,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "codebook_size": [64, 128, 256, 1],
        "topk_list": [5, 10, 20],
        "beam_size": args.beam_size,
    }

    print(f"[Task #{args.task_id}] Stage 4 test eval")
    print(f"  ckpt = {args.stage3_ckpt}")
    print(f"  code_path = {args.code_path}")
    print(f"  output_json = {args.output_json}")

    device = torch.device(args.device)
    model = HG_Rec(config)
    state_dict = torch.load(args.stage3_ckpt, map_location="cpu")
    # 兼容纯 state_dict 和 {'state_dict': ...} 两种格式
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # code_path 是 .npy 文件名 (相对 HG-Rec/dataset/Instruments/), GenRecDataset 会拼成 dataset_name + filename
    # 我们的产物: Instruments_t5_hrqvae_<quantizer>.npy (例如 Instruments_t5_hrqvae_fsk.npy)
    # GenRecDataset 期望参数: code_path 会被拼成 {dataset_name}{code_path}
    # dataset_name='Instruments' + code_path='_t5_hrqvae_fsk.npy' → 'Instruments_t5_hrqvae_fsk.npy' ✓
    code_filename = "_" + os.path.basename(args.code_path).replace("Instruments_", "", 1)
    print(f"  derived code_filename = {code_filename}")

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(config["dataset_path"], config["dataset_name"], "test.parquet"),
        code_path=os.path.join(config["dataset_path"], config["dataset_name"], config["dataset_name"] + code_filename),
        mode="evaluation",
        codebook_size=config["codebook_size"],
        max_len=config["max_len"],
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config["infer_size"], shuffle=False)
    print(f"  test set size = {len(test_dataset)}")

    avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config["topk_list"], config["beam_size"], device)

    result = {
        "task_id": args.task_id,
        "best_ckpt": args.stage3_ckpt,
        "code_path": args.code_path,
        "test_recalls": avg_recalls,
        "test_ndcgs": avg_ndcgs,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Stage 4] Test metrics saved: {args.output_json}")
    print(f"[Stage 4] Test recalls: {avg_recalls}")
    print(f"[Stage 4] Test ndcgs:   {avg_ndcgs}")


if __name__ == "__main__":
    main()