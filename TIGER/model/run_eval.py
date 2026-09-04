"""run_eval.py — TIGER Stage 4 Eval (test_R@5/10/20 + NDCG@5/10/20).

单卡即可, 加载 best ckpt 在 test set 上评估. 输出 final_metrics.json + test_metrics.
"""
import os
import sys
import argparse
import json

_TIGER_ROOT = "/fs04/ar57/wenyu/GeneRec/TIGER"
_MODEL_DIR = os.path.join(_TIGER_ROOT, "model")
sys.path.insert(0, _TIGER_ROOT)
sys.path.insert(0, _MODEL_DIR)

# === R51+ ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from dataset import GenRecDataset
from main import TIGER as TIGERClass, set_seed, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--code_path", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infer_size", type=int, default=96)
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--d_kv", type=int, default=64)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--vocab_size", type=int, default=1025)
    p.add_argument("--pad_token_id", type=int, default=0)
    p.add_argument("--eos_token_id", type=int, default=0)
    p.add_argument("--feed_forward_proj", default="relu")
    p.add_argument("--beam_size", type=int, default=20)
    p.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
    return p.parse_args()


def main():
    args = parse_args()

    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = vars(args)
    tiger_model = TIGERClass(config).to(device)
    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    tiger_model.load_state_dict(state)
    tiger_model.eval()

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, "test.parquet"),
        code_path=args.code_path, mode="evaluation", max_len=args.max_len,
    )

    def collate_fn(batch, pad_token=0):
        """Collate fn — must match run_t5_ddp.py collate_fn exactly (PAD bug fix).
        dataset.py already pads history to max_len sublists of length 3 each.
        """
        histories = []
        targets = []
        attention_masks = []

        for item in batch:
            flat_hist = []
            for sublist in item['history']:
                sublist_list = sublist.tolist() if hasattr(sublist, 'tolist') else list(sublist)
                flat_hist.extend(sublist_list)
            attn = [0 if t == pad_token else 1 for t in flat_hist]
            histories.append(flat_hist)
            attention_masks.append(attn)

            tgt = item['target']
            tgt_list = tgt.tolist() if hasattr(tgt, 'tolist') else list(tgt)
            targets.append(tgt_list)

        return {
            'history': torch.tensor(histories, dtype=torch.int64),
            'target': torch.tensor(targets, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.int64),
        }

    test_loader = DataLoader(test_dataset, batch_size=args.infer_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn, pin_memory=True)

    recalls, ndcgs = evaluate(tiger_model, test_loader, args.topk_list, args.beam_size, device)
    print(f"[Stage4] test_recalls={recalls}")
    print(f"[Stage4] test_ndcgs={ndcgs}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"test_recalls": recalls, "test_ndcgs": ndcgs, "n_eval": len(test_dataset), "beam_size": args.beam_size}, f, indent=2, default=str)
    print(f"[Stage4] saved {args.output_path}")


if __name__ == "__main__":
    main()