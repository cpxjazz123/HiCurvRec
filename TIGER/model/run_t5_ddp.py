"""run_t5_ddp.py — TIGER Stage 3 T5 DDP 4 卡 wrapper.

包装 TIGER model/main.py 为 DDP 4 卡 + R51+ 6 项确定性约束.
DDP 切分 DataLoader (DistributedSampler), model 用 DistributedDataParallel.
Stage 3 + Stage 4 一起跑 (训练期 best ckpt 在 test set 上评估, 输出 final metrics).

数据格式 (修复 dataset.py PAD bug 后):
- 每个 sample: history = list of 20 sublists, 每个 sublist = 3 tokens (PAD) 或 3 个 code tokens (真实 item)
- target = 1 sublist of 3 tokens
- flatten history = 20 * 3 = 60 tokens (所有 sample 长度一致, 因为 dataset.py 已经 pad 到 max_len)
- flatten target = 3 tokens
"""
import os
import sys
import argparse
import json

_TIGER_ROOT = "/fs04/ar57/wenyu/GeneRec/TIGER"
_MODEL_DIR = os.path.join(_TIGER_ROOT, "model")
sys.path.insert(0, _TIGER_ROOT)
sys.path.insert(0, _MODEL_DIR)

# === R51+ 6 项确定性约束 ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

from dataset import GenRecDataset
from main import TIGER, set_seed, train, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--code_path", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--infer_size", type=int, default=96)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
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
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--beam_size", type=int, default=20)
    p.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
    return p.parse_args()


def collate_fn(batch, pad_token=0):
    """Collate function for GenRecDataset samples.

    After dataset.py PAD bug fix (PAD len=3 not 4), all samples have:
    - history: list of max_len (=20) sublists, each of length 3
    - target: list of length 3

    Flatten history to 1D tensor of shape (B, max_len*3=60), attention_mask (B, 60).
    Target to tensor of shape (B, 3).

    No padding needed because dataset.py already left-padded to max_len sublists.
    """
    histories = []
    targets = []
    attention_masks = []

    for item in batch:
        flat_hist = []
        for sublist in item['history']:
            sublist_list = sublist.tolist() if hasattr(sublist, 'tolist') else list(sublist)
            flat_hist.extend(sublist_list)
        # flat_hist length = max_len * 3 = 60 (dataset.py already padded to max_len)
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


def main():
    args = parse_args()

    # === DDP ===
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # === R51+ seed ===
    seed = args.seed + rank
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")

    # === 模型 ===
    config = vars(args)
    tiger_model = TIGER(config).to(device)
    tiger_model = DDP(tiger_model, device_ids=[device.index], find_unused_parameters=False)

    # === 数据 (DistributedSampler) ===
    train_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, "train.parquet"),
        code_path=args.code_path, mode="train", max_len=args.max_len,
    )
    valid_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, "valid.parquet"),
        code_path=args.code_path, mode="evaluation", max_len=args.max_len,
    )
    test_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, "test.parquet"),
        code_path=args.code_path, mode="evaluation", max_len=args.max_len,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=0, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.infer_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.infer_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    optimizer = optim.Adam(tiger_model.parameters(), lr=args.lr)

    # === 训练循环 ===
    best_ndcg = 0.0
    early_stop_counter = 0
    final_metrics = None

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train(tiger_model.module, train_loader, optimizer, device)
        avg_recalls, avg_ndcgs = evaluate(tiger_model.module, valid_loader, args.topk_list, args.beam_size, device)

        if rank == 0:
            print(f"[DDP Stage3] epoch={epoch} loss={train_loss:.4f} valid_NDCG@20={avg_ndcgs['NDCG@20']:.4f}")

        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0
            test_recalls, avg_test_ndcgs = evaluate(tiger_model.module, test_loader, args.topk_list, args.beam_size, device)
            final_metrics = {"test_recalls": test_recalls, "test_ndcgs": avg_test_ndcgs, "best_ndcg20": best_ndcg, "epoch": epoch}
            if rank == 0:
                torch.save(tiger_model.module.state_dict(), args.save_path)
                print(f"[DDP Stage3] best NDCG@20={best_ndcg:.4f} → saved {args.save_path}")
                print(f"[DDP Stage3] test_recalls={test_recalls}")
                print(f"[DDP Stage3] test_ndcgs={avg_test_ndcgs}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop:
                if rank == 0:
                    print(f"[DDP Stage3] EARLY_STOP at epoch={epoch}")
                break

    # === rank 0 写 last ckpt + final_metrics ===
    if rank == 0:
        torch.save(tiger_model.module.state_dict(), args.save_path.replace(".pth", "_last.pth"))
        print(f"[DDP Stage3] last epoch ckpt saved")
        if final_metrics is not None:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            with open(args.save_path.replace(".pth", "_final_metrics.json"), "w") as f:
                json.dump(final_metrics, f, indent=2, default=str)
            print(f"[DDP Stage3] final metrics saved")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()