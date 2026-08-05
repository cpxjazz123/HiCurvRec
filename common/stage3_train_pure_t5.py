"""纯 T5 基线 stage3 训练 — 与 HG-Rec/train_HG-Rec.py 完全一致的方法.

R30: 所有超参硬编码在脚本顶部常量区. launcher (.sh) 仅负责 GPU 选择 + 路径
(SID_NPY / PRODUCT_DIR / DEVICE) + TAG/EXPECTED_SID_SHA 校验, 不传任何超参.
实验变体: 复制此脚本为新文件改常量, 不复用同一脚本 + env toggle.

用指定 SID npy (taskA/taskB stage2 新 SID) 作为 item-to-code 映射,
T5 随机初始化从头训练, 无 adapter 注入, 监控 valid NDCG@20 (beam20), early stop.

当前变体默认值 (Issue #41 任务 + hyp 系列常用):
  NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=256, INFER_SIZE=96,
  SEED=42, LR=1e-4, MAX_LEN=20, NUM_WORKERS=0, STAGE3_BF16=True,
  TORCH_COMPILE=False.

产物 (PRODUCT_DIR/):
  HG_Rec_best.pth    最佳 NDCG@20 ckpt
  trace.json         每 epoch train_loss / R@5/10/20 / NDCG@5/10/20 / best
  verdict.json       SID sha + config + 最终指标 + best epoch
  _TRAINING_PID      PID (R12)
"""
import os
import sys
import json
import hashlib
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

from HG_Rec import HG_Rec          # noqa: E402
from dataset import GenRecDataset  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402

# ---------------------------------------------------------------- config (R30 硬编码)
# 路径 / DDP infrastructure 保留 env var (launcher 控制, R30 豁免)
SID_NPY = os.environ["SID_NPY"]                       # R30: 路径 — launcher 必传
PRODUCT_DIR = Path(os.environ["PRODUCT_DIR"])         # R30: 路径 — launcher 必传
DEVICE = os.environ.get("DEVICE", "cuda:0")           # R30: GPU 选择 — launcher 必传
TAG = os.environ.get("TAG", "task")                   # R30: metadata 标签
EXPECTED_SID_SHA = os.environ.get("EXPECTED_SID_SHA", "")  # R30: SID 校验 (可选, 预期内缺失)

# 超参 (R30 硬编码 — 变体需 fork 脚本)
NUM_EPOCHS = 200
EARLY_STOP = 20
BATCH_SIZE = 256
INFER_SIZE = 96
SEED = 42
LR = 1e-4
MAX_LEN = 20
NUM_WORKERS = 0

# 训练加速 (2026-08-03): bf16 autocast — T5 训练/beam 解码标准实践, forward 转 bf16 计算
# (参数保持 fp32, backward 后 optimizer 在 fp32 权重更新, 数值影响极小). 默认开.
STAGE3_BF16 = True
# v29 加速 (2026-08-04): torch.compile — PyTorch 2.x 内置, A100+ 推荐 reduce-overhead
_TORCH_COMPILE = False
_TRAIN_COMPILED = False

# 多卡加速 (2026-08-03): DDP 数据并行. torchrun 启动自动注入 WORLD_SIZE/RANK/LOCAL_RANK.
# 保留为 env var (DDP infrastructure, R30 豁免).
# 全局 batch 严格保持 BATCH_SIZE (每卡 BATCH_SIZE//WORLD_SIZE, 梯度 all-reduce 平均) → 与单卡
# batch 语义数值等价 (T5 无 batch norm). 非 DDP (WORLD_SIZE=1) 走原 GenRecDataLoader 路径不变.
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DDP_MODE = WORLD_SIZE > 1


def _collate_fn(batch, pad_token=0):
    # 与 HG-Rec/data/dataloader.py GenRecDataLoader.collate_fn 逐字节一致. DDP 需要 sampler,
    # 但 GenRecDataLoader 不接受 sampler 参数 → 复刻 collate 以构造标准 DataLoader (仅 DDP 用).
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    flattened_histories = torch.stack(
        [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
    )
    flattened_targets = torch.stack(
        [torch.tensor(target, dtype=torch.int64) for target in targets]
    )
    attention_masks = torch.stack(
        [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
    )
    return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}

CODEBOOK_SIZE = [64, 128, 256, 1]          # 基线 stage2 结构
CONFIG = dict(                               # 基线 T5 (task84, encoder 6 + decoder 4)
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
DATA_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
TRAIN_PARQUET = os.environ.get("TRAIN_PARQUET", os.path.join(DATA_ROOT, "train.parquet"))
VALID_PARQUET = os.environ.get("VALID_PARQUET", os.path.join(DATA_ROOT, "valid.parquet"))
TOP_K = [5, 10, 20]
BEAM_SIZE = 20

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_PATH = PRODUCT_DIR / "train_pure_t5.log"
CKPT_PATH = PRODUCT_DIR / "HG_Rec_best.pth"
TRACE_PATH = PRODUCT_DIR / "trace.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_pos_index(preds, labels, maxk=20):
    # preds: (B, maxk, seq_len) 每 beam 生成的 token 序列; labels: (B, seq_len) SID code.
    # 加速 (2026-08-03): 向量化 "beam 序列与 target code 全等" 判定 (原逻辑对每个 (i,j) 逐 token
    # Python 比较 → 每 epoch ~50 万次循环), 数值完全等价.
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"
    # labels (B, seq_len) → (B,1,seq_len) 与 preds (B,maxk,seq_len) 广播, all(dim=-1) 判全等
    pos_index = (preds == labels.unsqueeze(1)).all(dim=-1)  # (B, maxk)
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if STAGE3_BF16 else torch.nullcontext())
    # v29 加速 (2026-08-04): torch.compile — env TORCH_COMPILE=1 启用
    if _TORCH_COMPILE and not _TRAIN_COMPILED:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            globals()['_TRAIN_COMPILED'] = True
            log("[v29] torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            log(f"[v29] torch.compile failed: {e}")
            globals()['_TRAIN_COMPILED'] = True  # 不再试
    for batch in train_loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        # 加速: bf16 forward (T5 标准混合精度, 参数 fp32, 仅前向计算转 bf16)
        with autocast_ctx:
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.shape[0]
        n += input_ids.shape[0]
    # DDP: all-reduce 各卡 loss (按样本数加权平均, 全局一致)
    if DDP_MODE:
        t = torch.tensor([total_loss, float(n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, n = t[0].item(), int(t[1].item())
    return total_loss / n


def evaluate(model, eval_loader, device):
    model.eval()
    # DDP 包装后 generate 在 model.module 上 (forward 走 DDP.__call__, generate 是原模型方法)
    gen_model = model.module if DDP_MODE else model
    recalls = {f"R@{k}": [] for k in TOP_K}
    ndcgs = {f"NDCG@{k}": [] for k in TOP_K}
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if STAGE3_BF16 else torch.nullcontext())
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            with autocast_ctx:
                preds = gen_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())
    out_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    out_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    # DDP: 每卡 eval 分片 local mean, all-reduce SUM / WORLD_SIZE = 全局 mean
    if DDP_MODE:
        for k in TOP_K:
            rk = f"R@{k}"
            t = torch.tensor([out_recalls[rk], out_ndcgs[f"NDCG@{k}"]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out_recalls[rk], out_ndcgs[f"NDCG@{k}"] = t[0].item() / WORLD_SIZE, t[1].item() / WORLD_SIZE
    return out_recalls, out_ndcgs


def main():
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    # DDP: init_process_group (env://), 每卡 device 由 LOCAL_RANK 决定, rank 0 为主进程
    if DDP_MODE:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(LOCAL_RANK)
        device = f"cuda:{LOCAL_RANK}"
        is_main = (RANK == 0)
    else:
        device = DEVICE
        is_main = True
    if is_main:
        with open(TRAINING_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    set_seed(SEED)

    # ---- SID 校验 (EXPECTED_SID_SHA 不设则跳过 = 预期内缺失) ----
    sid_sha = sha256_of(SID_NPY)
    if EXPECTED_SID_SHA:
        if sid_sha != EXPECTED_SID_SHA:
            raise ValueError(
                f"SID sha mismatch: got {sid_sha}, expected {EXPECTED_SID_SHA} ({SID_NPY})")
        if is_main:
            log(f"[SHA256] SID_NPY OK: {sid_sha}")

    if is_main:
        log(f"config: TAG={TAG} SID={SID_NPY} sha={sid_sha} epochs={NUM_EPOCHS} "
            f"early_stop={EARLY_STOP} batch={BATCH_SIZE} lr={LR} seed={SEED} device={device} "
            f"world_size={WORLD_SIZE} bf16={STAGE3_BF16}")

    model = HG_Rec(CONFIG)
    if is_main:
        log(model.n_parameters.rstrip())
    model.to(device)
    if DDP_MODE:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    valid_ds = GenRecDataset(
        dataset_path=VALID_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    if is_main:
        log(f"n_train(windowed)={len(train_ds)} n_valid={len(valid_ds)}")

    # DDP: DistributedSampler 分片 + 标准 DataLoader (collate 复刻自 GenRecDataLoader);
    # 全局 batch 严格保持 (每卡 BATCH_SIZE//WORLD_SIZE). 非 DDP 走原 GenRecDataLoader 不变.
    if DDP_MODE:
        if BATCH_SIZE % WORLD_SIZE != 0:
            raise ValueError(f"BATCH_SIZE={BATCH_SIZE} 必须被 WORLD_SIZE={WORLD_SIZE} 整除 (DDP 全局 batch 严格保持)")
        if INFER_SIZE % WORLD_SIZE != 0:
            raise ValueError(f"INFER_SIZE={INFER_SIZE} 必须被 WORLD_SIZE={WORLD_SIZE} 整除 (DDP eval 分片)")
        train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE // WORLD_SIZE, sampler=train_sampler,
                                  num_workers=NUM_WORKERS, collate_fn=_collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=INFER_SIZE // WORLD_SIZE, sampler=valid_sampler,
                                  num_workers=NUM_WORKERS, collate_fn=_collate_fn)
    else:
        train_loader = GenRecDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        valid_loader = GenRecDataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    best_ndcg = 0.0
    best_epoch = -1
    early_stop_counter = 0
    trace = []

    for epoch in range(NUM_EPOCHS):
        if DDP_MODE:
            train_sampler.set_epoch(epoch)  # 每 epoch 不同 shuffle (所有 rank 一致)
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, device, epoch)
        t_train = time.time() - t0

        t0 = time.time()
        recalls, ndcgs = evaluate(model, valid_loader, device)
        t_eval = time.time() - t0

        if is_main:
            row = {
                "epoch": epoch, "train_loss": train_loss,
                **recalls, **ndcgs,
                "t_train": round(t_train, 1), "t_eval": round(t_eval, 1),
            }
            trace.append(row)
            log(f"Epoch {epoch+1}/{NUM_EPOCHS} loss={train_loss:.4f} "
                f"R@10={recalls['R@10']:.4f} NDCG@20={ndcgs['NDCG@20']:.4f} "
                f"(train {t_train:.0f}s, eval {t_eval:.0f}s)")

            if ndcgs["NDCG@20"] > best_ndcg:
                best_ndcg = ndcgs["NDCG@20"]
                best_epoch = epoch
                early_stop_counter = 0
                torch.save(model.module.state_dict() if DDP_MODE else model.state_dict(), CKPT_PATH)
                log(f"[BEST] NDCG@20={best_ndcg:.4f} saved {CKPT_PATH}")
            else:
                early_stop_counter += 1
                log(f"no improv ({early_stop_counter}/{EARLY_STOP})")
        # DDP: rank 0 的 early stop 决策 broadcast 到所有 rank (否则非主卡继续跑, 卡死)
        if DDP_MODE:
            stop_flag = torch.tensor(1 if is_main and early_stop_counter >= EARLY_STOP else 0, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                if is_main:
                    log("early stop triggered")
                break
        elif early_stop_counter >= EARLY_STOP:
            log("early stop triggered")
            break

    if is_main:
        with open(TRACE_PATH, "w") as f:
            json.dump(trace, f, indent=2)
        verdict = {
            "tag": TAG, "sid_npy": SID_NPY, "sid_sha256": sid_sha,
            "best_epoch": best_epoch, "best_ndcg20": best_ndcg,
            "final_epoch": trace[-1]["epoch"] if trace else None,
            "best_trace": trace[best_epoch] if 0 <= best_epoch < len(trace) else None,
            "config": {**CONFIG, "lr": LR, "batch_size": BATCH_SIZE,
                       "num_epochs": NUM_EPOCHS, "early_stop": EARLY_STOP, "seed": SEED,
                       "world_size": WORLD_SIZE, "bf16": STAGE3_BF16},
            "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(VERDICT_PATH, "w") as f:
            json.dump(verdict, f, indent=2)
        log(f"[VERDICT] {VERDICT_PATH} best_epoch={best_epoch} best_NDCG@20={best_ndcg:.4f}")
        log("DONE")
    if DDP_MODE:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
