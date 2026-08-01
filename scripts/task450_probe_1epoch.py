#!/usr/bin/env python3
"""Issue #150 Task #450 Probe 1 epoch — 测实际 epoch 耗时, 决定 long train epoch 规模."""
import sys
import os
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task450"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

import importlib.util as _ilu
_t440_spec = _ilu.spec_from_file_location("task440_module",
    "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task440_issue150_zero_centered_linear_layernorm.py")
_t440_mod = _ilu.module_from_spec(_t440_spec)
_t440_spec.loader.exec_module(_t440_mod)

SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS_PROBE = 1
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)

train_ds = GenRecDataset(dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
                         codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN)
n_samples = len(train_ds)
print(f"train_ds size: {n_samples}", flush=True)

all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
all_targets = np.zeros((n_samples, 4), dtype=np.int64)
for i in range(n_samples):
    s = train_ds[i]
    all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
    all_targets[i] = np.asarray(s["target"], dtype=np.int64)
all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()
print(f"preload done: {all_histories.shape}, {all_targets.shape}", flush=True)

t5_state_dict = _t440_mod.load_t5_state_dict(T5_CKPT)
t5_config = _t440_mod.get_t5_config()
model_wrapper = _t440_mod.HG_Rec_with_ZeroCenteredLayerNormAdapter(
    t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4).to(DEVICE)

conditioner_params = [p for n, p in model_wrapper.adapter.named_parameters()]
layernorm_params = list(model_wrapper.first_input_ln.parameters())
optimizer = torch.optim.Adam([
    {"params": conditioner_params, "lr": LR_CONDITIONER},
    {"params": layernorm_params, "lr": LR_LAYERNORM},
])

n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
rng = torch.Generator().manual_seed(42 + 8)

print(f"probe 1 epoch: {n_batches} batches", flush=True)
t_start = time.time()
for epoch in range(NUM_EPOCHS_PROBE):
    epoch_losses = []
    epoch_indices = torch.randperm(n_samples, generator=rng).to(DEVICE)
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_samples)
        batch_indices = epoch_indices[start:end]
        history_tensor_b = all_histories_t[batch_indices]
        target_tensor_b = all_targets_t[batch_indices]
        attention_mask_b = (history_tensor_b != PAD_TOKEN).long()
        B = history_tensor_b.shape[0]
        L_flat = MAX_LEN * 4
        digit_values = history_tensor_b.float()
        layer_idx = torch.arange(L_flat, device=DEVICE) % 4
        layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
        pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
        pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
        padding_flag = (digit_values == PAD_TOKEN).float()
        sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
        kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        loss, _, alpha_val = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                          labels=target_tensor_b, sid_meta=sid_meta, kappa_meta=kappa_meta)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if batch_idx % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  [epoch {epoch}, batch {batch_idx}/{n_batches}] loss={loss.item():.4f}, elapsed={elapsed:.1f}s", flush=True)
t_end = time.time()
elapsed_per_epoch = t_end - t_start
print(f"\n[Probe 完成] 1 epoch 耗时: {elapsed_per_epoch:.1f}s = {elapsed_per_epoch/60:.1f}min", flush=True)
print(f"[Estimated] 200 epoch 耗时: {elapsed_per_epoch * 200 / 3600:.1f}h", flush=True)
print(f"[Estimated] 100 epoch 耗时: {elapsed_per_epoch * 100 / 3600:.1f}h", flush=True)
print(f"[Estimated] 50 epoch 耗时: {elapsed_per_epoch * 50 / 3600:.1f}h", flush=True)

# 落 probe 结果
with open("/home/wlia0047/ar57/wenyu/GeneRec/products/task450_issue150_stage3_long_train_200ep/probe_1epoch.json", "w") as f:
    json.dump({"n_samples": n_samples, "n_batches": n_batches, "batch_size": BATCH_SIZE,
               "elapsed_per_epoch_sec": elapsed_per_epoch,
               "est_200epoch_h": elapsed_per_epoch * 200 / 3600,
               "est_100epoch_h": elapsed_per_epoch * 100 / 3600,
               "est_50epoch_h": elapsed_per_epoch * 50 / 3600,
               "loss_start": epoch_losses[0], "loss_end": epoch_losses[-1]},
              f, indent=2)
print(f"[Probe 落盘] products/task450_issue150_stage3_long_train_200ep/probe_1epoch.json", flush=True)
