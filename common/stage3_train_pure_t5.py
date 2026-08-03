"""纯 T5 基线 stage3 训练 — 与 HG-Rec/train_HG-Rec.py 完全一致的方法.

用指定 SID npy (taskA/taskB stage2 新 SID) 作为 item-to-code 映射,
T5 随机初始化从头训练, 无 adapter 注入, 监控 valid NDCG@20 (beam20), early stop.

环境变量 (与 taskX_stage3.py 风格一致):
  SID_NPY           必填 — stage2 新 SID npy 路径
  PRODUCT_DIR       必填 — 产物目录 (taskX/_history/<tag>/)
  DEVICE            默认 cuda:0
  TAG               默认 "task" — 方向标签 (taskA/taskB), 写进 verdict
  EXPECTED_SID_SHA  可选 — 校验 SID npy 文件字节 sha256; 不设则跳过 (预期内缺失)
  NUM_EPOCHS        默认 200 (基线)
  EARLY_STOP        默认 20  (基线)
  BATCH_SIZE        默认 256 (基线)
  INFER_SIZE        默认 96  (基线 eval batch)
  SEED              默认 42  (项目 R5)
  LR                默认 1e-4 (基线)
  MAX_LEN           默认 20  (基线)
  NUM_WORKERS       默认 0 (基线默认 4, 但 fork + CUDA 不稳, 用 0 数据量小不受影响)

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

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

from HG_Rec import HG_Rec          # noqa: E402
from dataset import GenRecDataset  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402

# ---------------------------------------------------------------- config
SID_NPY = os.environ["SID_NPY"]
PRODUCT_DIR = Path(os.environ["PRODUCT_DIR"])
DEVICE = os.environ.get("DEVICE", "cuda:0")
TAG = os.environ.get("TAG", "task")
EXPECTED_SID_SHA = os.environ.get("EXPECTED_SID_SHA", "")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "200"))
EARLY_STOP = int(os.environ.get("EARLY_STOP", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
INFER_SIZE = int(os.environ.get("INFER_SIZE", "96"))
SEED = int(os.environ.get("SEED", "42"))
LR = float(os.environ.get("LR", "1e-4"))
MAX_LEN = int(os.environ.get("MAX_LEN", "20"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

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
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"
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


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in train_loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.shape[0]
        n += input_ids.shape[0]
    return total_loss / n


def evaluate(model, eval_loader, device):
    model.eval()
    recalls = {f"R@{k}": [] for k in TOP_K}
    ndcgs = {f"NDCG@{k}": [] for k in TOP_K}
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())
    return {k: sum(v) / len(v) for k, v in recalls.items()}, \
           {k: sum(v) / len(v) for k, v in ndcgs.items()}


def main():
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    set_seed(SEED)

    # ---- SID 校验 (EXPECTED_SID_SHA 不设则跳过 = 预期内缺失) ----
    sid_sha = sha256_of(SID_NPY)
    if EXPECTED_SID_SHA:
        if sid_sha != EXPECTED_SID_SHA:
            raise ValueError(
                f"SID sha mismatch: got {sid_sha}, expected {EXPECTED_SID_SHA} ({SID_NPY})")
        log(f"[SHA256] SID_NPY OK: {sid_sha}")

    log(f"config: TAG={TAG} SID={SID_NPY} sha={sid_sha} epochs={NUM_EPOCHS} "
        f"early_stop={EARLY_STOP} batch={BATCH_SIZE} lr={LR} seed={SEED} device={DEVICE}")

    model = HG_Rec(CONFIG)
    log(model.n_parameters.rstrip())
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    valid_ds = GenRecDataset(
        dataset_path=VALID_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    log(f"n_train(windowed)={len(train_ds)} n_valid={len(valid_ds)}")

    train_loader = GenRecDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = GenRecDataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    best_ndcg = 0.0
    best_epoch = -1
    early_stop_counter = 0
    trace = []

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, DEVICE, epoch)
        t_train = time.time() - t0

        t0 = time.time()
        recalls, ndcgs = evaluate(model, valid_loader, DEVICE)
        t_eval = time.time() - t0

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
            torch.save(model.state_dict(), CKPT_PATH)
            log(f"[BEST] NDCG@20={best_ndcg:.4f} saved {CKPT_PATH}")
        else:
            early_stop_counter += 1
            log(f"no improv ({early_stop_counter}/{EARLY_STOP})")
            if early_stop_counter >= EARLY_STOP:
                log("early stop triggered")
                break

    with open(TRACE_PATH, "w") as f:
        json.dump(trace, f, indent=2)
    verdict = {
        "tag": TAG, "sid_npy": SID_NPY, "sid_sha256": sid_sha,
        "best_epoch": best_epoch, "best_ndcg20": best_ndcg,
        "final_epoch": trace[-1]["epoch"] if trace else None,
        "best_trace": trace[best_epoch] if 0 <= best_epoch < len(trace) else None,
        "config": {**CONFIG, "lr": LR, "batch_size": BATCH_SIZE,
                   "num_epochs": NUM_EPOCHS, "early_stop": EARLY_STOP, "seed": SEED},
        "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2)
    log(f"[VERDICT] {VERDICT_PATH} best_epoch={best_epoch} best_NDCG@20={best_ndcg:.4f}")
    log("DONE")


if __name__ == "__main__":
    main()
