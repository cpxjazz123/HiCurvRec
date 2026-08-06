"""纯 T5 stage3 产物评估 — 基线 beam20 协议, 对 test.parquet 全量.

与 train_HG-Rec.py evaluate 完全一致:
  HG_Rec.generate(num_beams=20, num_return_sequences=20, max_length=5)
  preds = preds[:, 1:] -> (B, 20, 4);  R@K = target 4-token 出现在 top-k beam.

R30: 所有超参 + 路径 + GPU + 校验都通过 argparse 传入, 无任何 env var 读取.
launcher (.sh) 必须传 --ckpt_path / --sid_npy / --product_dir; 其他 arg 走默认值.
实验变体: 复制此脚本为新文件改默认值, 不复用同一脚本 + env toggle.

当前变体默认值 (Issue #41 + hyp 系列常用):
  BATCH_SIZE=96, SEED=42, MAX_LEN=20, BEAM_SIZE=20, TOP_K=[5,10,20].
"""
import os
import sys
import json
import hashlib
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

from HG_Rec import HG_Rec          # noqa: E402
from dataset import GenRecDataset  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402

# ──────────────────────────────────────────────────────────────
# argparse (R30 严格: 无 env var 读取)
# ──────────────────────────────────────────────────────────────
_argparser = argparse.ArgumentParser(description="Stage4 pure T5 eval (R30 strict, no env var)")
_argparser.add_argument("--ckpt_path", type=str, required=True, help="HG_Rec_best.pth 路径")
_argparser.add_argument("--sid_npy", type=str, required=True, help="stage2 SID npy 路径")
_argparser.add_argument("--product_dir", type=str, required=True, help="产物目录 (verdict 落这里)")
_argparser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
_argparser.add_argument("--tag", type=str, default="task", help="方向标签 (taskA/taskB), 写进 verdict")
_argparser.add_argument("--expected_sid_sha", type=str, default="", help="校验 SID 文件 sha256; 不传则跳过")
_argparser.add_argument("--eval_parquet", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet",
                        help="评估集路径 (也可传 valid.parquet 做交叉验证)")
_args = _argparser.parse_args()

CKPT_PATH = _args.ckpt_path
SID_NPY = _args.sid_npy
PRODUCT_DIR = Path(_args.product_dir)
DEVICE = _args.device
TAG = _args.tag
EXPECTED_SID_SHA = _args.expected_sid_sha
EVAL_PARQUET = _args.eval_parquet

# 超参 (R30 硬编码 — 变体需 fork 脚本)
BATCH_SIZE = 96
SEED = 42

CODEBOOK_SIZE = [64, 128, 256, 1]
CONFIG = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
TOP_K = [5, 10, 20]
BEAM_SIZE = 20
MAX_LEN = 20

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_PATH = PRODUCT_DIR / "eval_test.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            if preds[i, j].tolist() == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sid_sha = sha256_of(SID_NPY)
    if EXPECTED_SID_SHA:
        if sid_sha != EXPECTED_SID_SHA:
            raise ValueError(
                f"SID sha mismatch: got {sid_sha}, expected {EXPECTED_SID_SHA} ({SID_NPY})")

    model = HG_Rec(CONFIG)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu", weights_only=False))
    model.to(DEVICE)
    model.eval()

    ds = GenRecDataset(
        dataset_path=EVAL_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    loader = GenRecDataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n = len(ds)

    recalls = {f"R@{k}": [] for k in TOP_K}
    ndcgs = {f"NDCG@{k}": [] for k in TOP_K}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids = batch["history"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["target"].to(DEVICE)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())
            if (bi + 1) % 50 == 0:
                print(f"  {bi+1}/{len(loader)} batches done", flush=True)

    result = {k: sum(v) / len(v) for k, v in recalls.items()}
    result.update({k: sum(v) / len(v) for k, v in ndcgs.items()})
    result["n_eval"] = n
    result["t_eval_s"] = round(time.time() - t0, 1)
    result["ckpt"] = CKPT_PATH
    result["sid_sha256"] = sid_sha
    result["eval_parquet"] = EVAL_PARQUET
    result["tag"] = TAG
    result["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(VERDICT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"=== {TAG} test R@5/10/20 = {result['R@5']:.4f}/{result['R@10']:.4f}/{result['R@20']:.4f} "
          f"NDCG@5/10/20 = {result['NDCG@5']:.4f}/{result['NDCG@10']:.4f}/{result['NDCG@20']:.4f}")
    print(f"=== verdict: {VERDICT_PATH}")


if __name__ == "__main__":
    main()
