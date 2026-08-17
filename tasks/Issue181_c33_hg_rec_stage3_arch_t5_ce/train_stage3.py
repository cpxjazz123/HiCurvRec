"""C33 Stage3 trainer — HG-Rec 风格 T5ForConditionalGeneration (序列级 CE).

直接复用 HG-Rec 的 model/hg_rec.py (T5ForConditionalGeneration) 和 data/dataset.py (GenRecDataset).
差异 (C33 vs 原版 HG-Rec):
  - codebook_size: [256,256,256] (C28) 而非 HG-Rec [64,128,256,1]
  - vocab_size: 769 (C28 3 层平铺) 而非 1025 (HG-Rec 4 层)
  - SID 来源: C28 sids_c28_curriculum_m3.npy 而非 HG-Rec 重训

R36 严格: C33 是几何变换 (per-hierarchy CE → 序列级 CE + 异构 vocab 平铺),
不是参数调优. 与 C31 (HG-Rec arch 适配但保留 per-hierarchy CE) 是正交改进点.

R44: 数据集从 /home/wlia0047/ar57/wenyu/GeneRec/dataset/ 读取 (HG-Rec 也读同一目录)
R46: HG-Rec 代码已复制到 _lib/ 目录 (复制自 baseline/HG-Rec/), 唯一改动是把 codebook/vocab 改成 C28 的 3 层
R32: 单卡 python3 -u 直跑 (HG-Rec 原版单卡, 我们保留单卡 R35/R42 不强制 DDP, 因为 HG-Rec baseline 也是单卡)
R5: seed=42
R11: 直跑, 不等用户拍板
"""
import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

# HG-Rec code 来自 _lib/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_lib"))

from data.dataset import GenRecDataset  # noqa: E402
from data.dataloader import GenRecDataLoader  # noqa: E402
from model.hg_rec import HG_Rec  # noqa: E402
from model.utils import ensure_dir, get_local_time, set_color  # noqa: E402


# === C33 硬编码超参 (R30/R43: 路径参数可传, 数值超参必须硬编码) ===
SEED = 42
MAX_LEN = 20
CODEBOOK_SIZE = [256, 256, 256]  # C28: 3 层同构
N_LAYERS = len(CODEBOOK_SIZE)
VOCAB_SIZE = sum(CODEBOOK_SIZE) + 1  # 769 (PAD=0)
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 0

# HG-Rec 标准 T5 架构
D_MODEL = 128
D_FF = 1024
NUM_HEADS = 6
D_KV = 64
NUM_LAYERS = 6
NUM_DECODER_LAYERS = 4
DROPOUT_RATE = 0.1

# 训练超参 (HG-Rec 原版)
BATCH_SIZE = 256
INFER_SIZE = 96
NUM_EPOCHS = 200
LR = 1e-4
EARLY_STOP = 20  # R41: hardcoded 20

# 评估
TOPK_LIST = [5, 10, 20]
BEAM_SIZE = 20  # R35: 单 ckpt + beam=20
MAX_GEN_LEN = 5  # HG-Rec 标准: 4 layer code + 1 decoder_start

# 路径 (R44 约束: 与 HG-Rec baseline 同一份 parquet, 保证可比)
# HG-Rec parquet 位于 HG-Rec/dataset/Instruments/, 这是 HG-Rec test R@10=0.1074 用的版本
DATASET_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/dataset/Instruments/Instruments_c28_sids_for_hgrec.npy"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c33_train"
CKPT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c33_ckpt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
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


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def evaluate(model, loader, topk_list, beam_size, device, return_per_sample=False):
    """R35b 严格: 每样本恰好计数一次, 返回 (avg_recalls, avg_ndcgs, [可选 per_sample 列表])."""
    model.eval()
    recalls = {f"Recall@{k}": [] for k in topk_list}
    ndcgs = {f"NDCG@{k}": [] for k in topk_list}
    per_sample_records = [] if return_per_sample else None
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
            )
            preds = preds[:, 1:]  # 去 decoder_start_token_id (=PAD)
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            for k in topk_list:
                recalls[f"Recall@{k}"].append(recall_at_k(pos_index, k))
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k))
            if return_per_sample:
                # 收集每样本 R@10 / NDCG@20 (与 R35b 一致)
                for bi in range(input_ids.shape[0]):
                    per_sample_records.append({
                        "user": -1,  # 单卡无 rank, 用户 id 暂不重要
                        "recall@10": float(recall_at_k(pos_index[bi:bi+1], 10).item()),
                        "ndcg@20": float(ndcg_at_k(pos_index[bi:bi+1], 20).item()),
                    })
    avg_recalls = {k: torch.cat(v).mean().item() for k, v in recalls.items()}
    avg_ndcgs = {k: torch.cat(v).mean().item() for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs, per_sample_records


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    # C33 HG-Rec config (与原版 HG-Rec 一致, 只改 vocab_size/codebook_size)
    config = {
        "num_layers": NUM_LAYERS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "num_heads": NUM_HEADS,
        "d_kv": D_KV,
        "dropout_rate": DROPOUT_RATE,
        "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "decoder_start_token_id": PAD_TOKEN_ID,
        "feed_forward_proj": "relu",
    }
    model = HG_Rec(config).to(device)
    print(f"[model] HG_Rec created", flush=True)
    print(model.n_parameters, flush=True)

    # 数据集
    train_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "train.parquet"),
        code_path=CODE_PATH,
        mode="train",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN_ID,
    )
    valid_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "valid.parquet"),
        code_path=CODE_PATH,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN_ID,
    )
    test_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "test.parquet"),
        code_path=CODE_PATH,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN_ID,
    )
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}", flush=True)

    train_loader = GenRecDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = GenRecDataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=2)
    test_loader = GenRecDataLoader(test_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    ensure_dir(LOG_DIR)
    ensure_dir(CKPT_DIR)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "C33_train.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    best_ndcg = 0.0
    best_epoch = 0
    early_stop_counter = 0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_ckpt.pt")
    last_ckpt_path = os.path.join(CKPT_DIR, "last_ckpt.pt")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        t_train = time.time() - t0
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] train_loss={train_loss:.4f} time={t_train:.1f}s", flush=True)
        logging.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

        t1 = time.time()
        avg_recalls, avg_ndcgs, _ = evaluate(model, valid_loader, TOPK_LIST, BEAM_SIZE, device)
        t_eval = time.time() - t1
        print(f"[Epoch {epoch+1}] valid {avg_recalls} {avg_ndcgs} eval_time={t_eval:.1f}s", flush=True)
        logging.info(f"Epoch {epoch+1} valid: {avg_recalls} {avg_ndcgs}")

        cur_ndcg = avg_ndcgs[f"NDCG@20"]
        if cur_ndcg > best_ndcg:
            best_ndcg = cur_ndcg
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[Best@{epoch+1}] saved best_ckpt.pt valid_NDCG@20={cur_ndcg:.4f}", flush=True)
        else:
            early_stop_counter += 1
            print(f"[NoImprove@{epoch+1}] counter={early_stop_counter}/{EARLY_STOP}", flush=True)
            if early_stop_counter >= EARLY_STOP:
                print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                break
        # 每个 epoch 也存 last (R12)
        torch.save(model.state_dict(), last_ckpt_path)

    # 训练结束: 加载 best_ckpt, 跑最终 test eval (HG-Rec 风格)
    print(f"\n=== 训练结束: 加载 best_ckpt 跑最终 test eval ===", flush=True)
    print(f"  best_ckpt epoch={best_epoch+1} valid_NDCG@20={best_ndcg:.4f}", flush=True)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_recalls, test_ndcgs, _ = evaluate(model, test_loader, TOPK_LIST, BEAM_SIZE, device)
    print(f"\n=== TEST FINAL (beam={BEAM_SIZE}) ===", flush=True)
    for k, v in test_recalls.items():
        print(f"  {k}: {v:.4f}", flush=True)
    for k, v in test_ndcgs.items():
        print(f"  {k}: {v:.4f}", flush=True)
    logging.info(f"TEST FINAL: {test_recalls} {test_ndcgs}")


if __name__ == "__main__":
    main()
