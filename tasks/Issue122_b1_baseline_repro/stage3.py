#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #122 b1 baseline_repro Stage3 — 复现 baseline v3e+LR=4e-4+v15 capmatch SID.

Issue122 lineage v1/v2/v3/v4 全部 NO-GO. v4 根因: monkey-patch T5 forward + tuple unpack
破坏 cross-entropy gradient 流. 现在恢复 baseline v3e+LR=4e-4 fork (c4289aa) 完整路径
(原生 T5 forward, 无 monkey-patch, 无 adapter) + Issue122 Stage2 v4 SID (v15 capmatch 等价)
验证能复现 baseline R@10=0.10+.

vs baseline v3e fork (c4289aa):
- 完全移除 monkey-patch, 用原生 T5ForConditionalGeneration.forward
- 完全移除 adapter (alpha=0 都不需要, 直接 baseline 路径)
- DDP 4-card (R42)
- LR=4e-4, BATCH_SIZE=256, cosine schedule, bf16 autocast (跟 baseline 一致)
- EARLY_STOP=20 (R41)
- 自包含 dataset/ + _lib/ (R40+R44)

vs Issue122 stage3 v4:
- 关键差异: 无 types.MethodType(adapter_forward, t5_model) monkey-patch
- 关键差异: 无 tuple return (用原生 CausalLMOutputWithPast)
- 关键差异: 无 DDP find_unused_parameters=True (默认 False, 跟 baseline 一致)
"""
import os
import sys
import json
import time
import math
import random
import hashlib
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# R40+R44 自包含: 用本任务 _lib/
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TASK_DIR = REPO / "tasks/Issue122_b1_baseline_repro"
DATASET_DIR = REPO / "dataset"
LIB_DIR = TASK_DIR / "_lib"
sys.path.insert(0, str(LIB_DIR))

STAGE2_DIR = TASK_DIR / "stage2"
STAGE3_DIR = TASK_DIR / "stage3"
STAGE4_DIR = TASK_DIR / "stage4"

# ──────────────────────────────────────────────────────────────
# 训练超参 (跟 baseline v3e+LR=4e-4 fork c4289aa 完全一致)
# ──────────────────────────────────────────────────────────────
NUM_EPOCHS = 100
EARLY_STOP = 20  # R41
BATCH_SIZE = 256
INFER_SIZE = 96
LR = 4e-4
WEIGHT_DECAY = 0.0  # baseline v3e fork 用 weight_decay=0.0 (无 L2 显式)
LABEL_SMOOTHING = 0.0  # baseline v3e fork 用 label_smoothing=0.0 (T5 默认)
DROPOUT = 0.10  # baseline v3e fork 用 0.10
MAX_LEN = 20
SEED = 42
DETERMINISTIC = True
BF16 = True

# baseline v3e fork T5_CONFIG (跟 c4289aa CONFIG 完全一致)
T5_CONFIG = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=DROPOUT, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)

# baseline v3e fork 评估超参
TOP_K = [5, 10, 20]
BEAM_SIZE = 20

# baseline v3e fork cosine schedule (warmup_epochs=20 → 我们的实现用 warmup_steps)
LR_WARMUP_EPOCHS = 20
LR_MIN_FACTOR = 0.01  # baseline v3e fork 末期 LR = LR * 0.01

EARLY_STOP_METRIC = "NDCG@20"  # baseline v3e fork 一致

PAD_TOKEN_ID = 0
SID_OFFSETS = [1, 65, 193, 449]


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [Issue122-b1-stage3] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GenRecDataset(Dataset):
    """跟 Issue122 stage3.py GenRecDataset 完全一致."""
    def __init__(self, dataset_path, code_path, mode="train", codebook_size=1024, max_len=20):
        df = pd.read_parquet(dataset_path)
        self.history = df["history"].tolist()
        self.target = df["target"].tolist()
        self.code = np.load(code_path)
        self.mode = mode
        self.max_len = max_len

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        hist_ids = list(self.history[idx])[:self.max_len]
        target_raw = self.target[idx]
        hist_tokens = []
        for item_id in hist_ids:
            sid = self.code[int(item_id) - 1]
            hist_tokens.extend(int(x) for x in sid)
        if isinstance(target_raw, (int, np.integer)):
            target_sid = [int(x) for x in self.code[int(target_raw) - 1]]
        else:
            target_sid = list(target_raw)
        return {"history": hist_tokens, "target": target_sid}


def collate_fn(batch, pad_token=PAD_TOKEN_ID):
    histories = [b["history"] for b in batch]
    targets = [b["target"] for b in batch]
    max_h = max(len(h) for h in histories)
    padded_hist = []
    attn_mask = []
    for h in histories:
        if len(h) < max_h:
            h_padded = list(h) + [pad_token] * (max_h - len(h))
        else:
            h_padded = list(h[:max_h])
        padded_hist.append(h_padded)
        attn_mask.append([1 if t != pad_token else 0 for t in h_padded])
    max_t = max(len(t) for t in targets)
    padded_targets = []
    for t in targets:
        if len(t) < max_t:
            padded_targets.append(list(t) + [pad_token] * (max_t - len(t)))
        else:
            padded_targets.append(list(t[:max_t]))
    return {
        "history": torch.tensor(padded_hist, dtype=torch.long),
        "target": torch.tensor(padded_targets, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }


def calculate_pos_index(preds, labels):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    return (preds == labels.unsqueeze(1)).all(dim=-1)


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0))
    return dcg[:, :k].sum(dim=1)


def train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if BF16
                    else torch.nullcontext())
    for step, batch in enumerate(train_loader):
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        with autocast_ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * input_ids.shape[0]
        n += input_ids.shape[0]
        if step % 200 == 0:
            log(f"  ep{epoch+1} step{step}/{len(train_loader)} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s")
    return total_loss / max(n, 1)


def evaluate(model, eval_loader, device, is_main=False):
    """跟 baseline v3e+LR=4e-4 fork (c4289aa) 路径一致.
    baseline fork 用 HG_Rec.generate 包装显式传 max_length=5 + num_return_sequences=num_beams.
    原生 T5 generate 默认 max_length=21 会跟 labels (B, 4) 维度不匹配 → 必须显式 max_length=5.
    """
    model.eval()
    gen_model = model.module if hasattr(model, "module") else model
    recalls = {f"R@{k}": [] for k in TOP_K}
    ndcgs = {f"NDCG@{k}": [] for k in TOP_K}
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            # baseline v3e+LR=4e-4 fork pf_generate 等价: max_new_tokens=4 + start token
            # generate 返回 (B*BEAM, max_new_tokens+1=5). 但 transformers 5.x 在 beam 提前 emit EOS
            # 时会截断 output (而非 pad), 导致 batch 间 shape 不一致.
            # pad_token_id=0 (跟 eos 一致), 用 min_new_tokens=4 强制至少 4 token + pad 兜底到 5.
            preds = gen_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                num_beams=BEAM_SIZE, num_return_sequences=BEAM_SIZE,
                max_new_tokens=4, min_new_tokens=4,
                pad_token_id=PAD_TOKEN_ID, eos_token_id=PAD_TOKEN_ID,
            )
            # 兜底: pad 到 5 columns (含 start), 取 [:, 1:5] 强制 4 token
            if preds.shape[1] < 5:
                pad_cols = 5 - preds.shape[1]
                padding = torch.full(
                    (preds.shape[0], pad_cols), PAD_TOKEN_ID,
                    dtype=preds.dtype, device=preds.device,
                )
                preds = torch.cat([preds, padding], dim=1)
            if batch_idx == 0 and is_main:
                log(f"  eval[0] preds.shape={tuple(preds.shape)} labels.shape={tuple(labels.shape)}")
            preds = preds[:, 1:5]  # 排除 start + 取 token 1..4
            if batch_idx == 0 and is_main:
                log(f"  eval[0] after [:,1:5] shape={tuple(preds.shape)}")
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            if batch_idx == 0 and is_main:
                log(f"  eval[0] after reshape shape={tuple(preds.shape)}")
            pos_index = calculate_pos_index(preds, labels)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k))
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k))
    metrics = {}
    for k in TOP_K:
        metrics[f"R@{k}"] = float(torch.cat(recalls[f"R@{k}"]).mean())
        metrics[f"NDCG@{k}"] = float(torch.cat(ndcgs[f"NDCG@{k}"]).mean())
    return metrics


def main():
    from transformers import T5Config, T5ForConditionalGeneration
    from torch.nn.parallel import DistributedDataParallel as DDP

    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_mode = world_size > 1
    if ddp_mode:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(SEED + rank)
    is_main = (rank == 0)
    if is_main:
        STAGE3_DIR.mkdir(parents=True, exist_ok=True)
        log(f"DDP world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")

    # 加载 SID (Issue122 Stage2 v4 = v15 capmatch 等价)
    sid_npy = STAGE2_DIR / "sid_output.npy"
    if not sid_npy.exists():
        raise FileNotFoundError(f"Stage2 SID 缺失: {sid_npy}. 先跑 stage2.py")
    sid = np.load(sid_npy)
    sha_sid = sha256_file(sid_npy)
    if is_main:
        log(f"sid_output.npy shape={sid.shape} dtype={sid.dtype} SHA256={sha_sid}")

    # 构造 T5 (原生, 无 monkey-patch)
    t5config = T5Config(**T5_CONFIG)
    t5_model = T5ForConditionalGeneration(t5config)
    t5_model = t5_model.to(device)
    if is_main:
        log(f"T5 params={sum(p.numel() for p in t5_model.parameters()):,}")

    # 数据集
    train_ds = GenRecDataset(DATASET_DIR / "train.parquet", sid_npy, mode="train", codebook_size=1024, max_len=MAX_LEN)
    valid_ds = GenRecDataset(DATASET_DIR / "valid.parquet", sid_npy, mode="evaluation", codebook_size=1024, max_len=MAX_LEN)
    if is_main:
        log(f"n_train={len(train_ds)} n_valid={len(valid_ds)}")

    if ddp_mode:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
        per_rank_batch = BATCH_SIZE // world_size
        per_rank_infer = INFER_SIZE // world_size
        train_loader = DataLoader(train_ds, batch_size=per_rank_batch, sampler=train_sampler,
                                  num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=per_rank_infer, sampler=valid_sampler,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # DDP 包装 (默认 find_unused_parameters=False, 跟 baseline c4289aa 一致)
    if ddp_mode:
        t5_model = DDP(t5_model, device_ids=[local_rank])

    # Optimizer + Scheduler (跟 baseline v3e fork 一致)
    optimizer = optim.AdamW(t5_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(total_steps * (LR_WARMUP_EPOCHS / NUM_EPOCHS)))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return LR_MIN_FACTOR + (1.0 - LR_MIN_FACTOR) * cos_factor

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if is_main:
        log(f"optimizer AdamW lr={LR} wd={WEIGHT_DECAY}, cosine schedule warmup_steps={warmup_steps}/{total_steps}")

    # 训练循环
    best_valid_metric = 0.0
    best_epoch = -1
    best_valid_r10 = 0.0
    no_improve_count = 0
    trace = []
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        if ddp_mode:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(t5_model, train_loader, optimizer, device, epoch, scheduler)
        if ddp_mode:
            valid_sampler.set_epoch(epoch)
        metrics = evaluate(t5_model, valid_loader, device, is_main=is_main)
        if is_main:
            log(f"ep {epoch+1}/{NUM_EPOCHS} train_loss={train_loss:.4f} elapsed={time.time()-t0:.0f}s")
            log(f"  valid: R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} R@20={metrics['R@20']:.4f} "
                f"NDCG@5={metrics['NDCG@5']:.4f} NDCG@10={metrics['NDCG@10']:.4f} NDCG@20={metrics['NDCG@20']:.4f}")
            trace.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})

        # early stop
        cur_metric = metrics[EARLY_STOP_METRIC]
        if cur_metric > best_valid_metric:
            best_valid_metric = cur_metric
            best_valid_r10 = metrics["R@10"]
            best_epoch = epoch + 1
            no_improve_count = 0
            if is_main:
                ckpt_path = STAGE3_DIR / "HG_Rec_best.pth"
                base_model = t5_model.module if hasattr(t5_model, "module") else t5_model
                torch.save({
                    "model_state_dict": base_model.state_dict(),
                    "epoch": epoch + 1,
                    "best_valid_R10": best_valid_r10,
                    "best_valid_metric": best_valid_metric,
                    "config": T5_CONFIG,
                }, ckpt_path)
                log(f"  new best {EARLY_STOP_METRIC}={best_valid_metric:.4f} (R@10={best_valid_r10:.4f}), saved {ckpt_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP:
                if is_main:
                    log(f"EARLY_STOP={EARLY_STOP} reached at ep{epoch+1}")
                break

    # 写 verdict
    best_metrics = {}
    for tr in trace:
        if tr.get("epoch") == best_epoch:
            best_metrics = tr
            break
    if not best_metrics and trace:
        best_metrics = trace[-1]
    if is_main:
        ckpt_path = STAGE3_DIR / "HG_Rec_best.pth"
        ckpt_sha = sha256_file(ckpt_path) if ckpt_path.exists() else None
        verdict = {
            "issue_iid": "122_b1_baseline_repro",
            "stage": "Stage3 baseline_repro training complete",
            "config": T5_CONFIG,
            "training": {
                "num_epochs_run": epoch + 1,
                "best_epoch": best_epoch,
                "best_valid_R10": best_metrics.get("R@10", 0.0),
                "best_valid_NDCG20": best_metrics.get("NDCG@20", 0.0),
                "early_stop_metric": EARLY_STOP_METRIC,
                "ddp_world_size": world_size,
                "batch_size_global": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "early_stop": EARLY_STOP,
            },
            "sid_path": str(sid_npy),
            "sid_sha256": sha_sid,
            "ckpt_path": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "trace": trace,
            "elapsed_s": round(time.time() - t0, 1),
            "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(STAGE3_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        with open(STAGE3_DIR / "trace.json", "w") as f:
            json.dump(trace, f, indent=2)
        with open(STAGE3_DIR / "_TRAINING_PID", "w") as f:
            f.write(str(os.getpid()))
        log(f"verdict: {STAGE3_DIR / 'verdict.json'}")
        log("DONE")

    if ddp_mode:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()