"""Evaluate an HG-Rec checkpoint with RecBole3.0's vanilla TIGER protocol."""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("hgrec_train", ROOT / "train_HG-Rec.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        required=True,
        help="HG_Rec_best.pth produced by train_HG-Rec.py",
    )
    parser.add_argument(
        "--sid_file",
        "--sid_npy",
        dest="sid_file",
        type=str,
        default="./dataset/Amazon_2023_Instruments/item_sids_recbole.json",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./dataset/Amazon_2023_Instruments/test_recbole.parquet",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="./test_final.json",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_exclude_history", action="store_true")
    parser.add_argument("--n_user_tokens", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_user_tokens <= 0:
        raise ValueError("RecBole TIGER requires --n_user_tokens >= 1")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = (
        torch.device(args.device)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else torch.device("cpu")
    )
    dataset = train_module.GenRecDataset(
        dataset_path=args.test_file,
        code_path=args.sid_file,
        mode="evaluation",
        codebook_size=[256, 256, 256, 1],
        max_len=args.max_len,
        n_user_tokens=args.n_user_tokens,
    )
    config = {
        "num_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "activation_function": "relu",
        "feed_forward_proj": "relu",
        "vocab_size": dataset.vocab_size,
        "pad_token_id": 0,
        "eos_token_id": dataset.eos_token,
        "decoder_start_token_id": 0,
        "sid_length": dataset.n_digit,
        "max_token_seq_len": dataset.max_token_seq_len,
    }
    model = train_module.HG_Rec(config).to(device)
    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"loaded {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = train_module.GenRecDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sample_collator=train_module.make_sample_collator(dataset, include_seen=True),
        max_token_seq_len=dataset.max_token_seq_len,
        include_seen=True,
        **loader_options,
    )
    recalls, ndcgs = train_module.evaluate(
        model,
        loader,
        dataset,
        [5, 10],
        args.beam_size,
        device,
        exclude_history=not args.no_exclude_history,
        use_bf16=args.bf16,
    )
    result = {
        "best_checkpoint": os.path.basename(args.ckpt),
        "n_eval": len(dataset),
        **{f"test_{key}": value for key, value in recalls.items()},
        **{f"test_{key}": value for key, value in ndcgs.items()},
    }
    print(json.dumps(result, indent=2), flush=True)
    output_path = Path(args.test_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
