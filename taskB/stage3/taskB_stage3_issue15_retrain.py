#!/usr/bin/env python3
"""Issue #15 [方向B Step 4] Stage3 重训 (从 0 开始, 含 learnable kappa_logits/mixing_logits).

Per #15 spec:
- Step 1 修了 taskB_stage3_mixed_curv_recontinue.py 加 nn.Parameter kappa_logits / mixing_logits
- 新 ckpt 不存在 (旧 best_adapter.pt 没这些 params)
- 必须从 0 重训才能产生满足 precheck 的 ckpt
- 用现有 taskB_stage3_issue193_long_run 训练循环逻辑, 但 START_EPOCH=0, 不 load PRIOR_CKPT

输出位置: taskB/stage3/taskB_stage3_issue15_retrain/
- adapter.pt (per-epoch save)
- best_adapter.pt (best val_R@10 save)
- val_trace.json
- _TRAINING_PID
- log via taskB/_logs/task_issue15_retrain.log
"""
import os
import sys
import json
import hashlib
import time
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ============================================================================
# Config (per Issue #15 spec + Issue #193 baseline config)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
START_EPOCH = 0  # 重训从 0 开始 (旧 ckpt 不含新 nn.Parameter)
NUM_EPOCHS = 60  # 跟 #193 同上限 (epoch 60 已 train_R@10 收敛, 60 epoch 是 #193 验证上限)
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
LR_NEW_PARAMS = 1e-3  # kappa_logits / mixing_logits 学习率

# Mid-train canary 配置
CANARY_EPOCH = 30  # 50% point
CANARY_N = 200
DECISION_BASELINE_R10 = 0.1020

# Val + early stop 配置
VAL_SPLIT_RATIO = 0.10
VAL_EVAL_N = 1000
EARLY_STOP_PATIENCE = 10

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/train.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_ckpt/HG_Rec_best.pth"
# NOTE: 不再 load PRIOR_CKPT, 因为旧 ckpt 不含 kappa_logits / mixing_logits

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue15_retrain")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue15_retrain.log"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
CANARY_VERDICT_PATH = PRODUCT_DIR / "canary_verdict.json"
STAGE4_VERDICT_PATH = PRODUCT_DIR / "stage4_verdict.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def get_layer_ranges(codebook_size):
    cum = [0]
    for k in codebook_size[:-1]:
        cum.append(cum[-1] + k)
    cum.append(cum[-1] + codebook_size[-1])
    return [(cum[i] + 1, cum[i + 1]) for i in range(len(codebook_size))]


def load_wrapper_cls():
    spec = importlib.util.spec_from_file_location(
        "t471",
        "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_mixed_curv_recontinue.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HG_Rec_with_BoundedWeightedMixedAdapter, mod.get_t5_config


def autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges):
    """Issue #191 修复: 自回归 4-forwards + 闭区间 mask."""
    B = history_tensor.shape[0]
    predicted = torch.zeros(B, 4, dtype=torch.long, device=history_tensor.device)
    cur_history = history_tensor.clone()
    cur_mask = attention_mask.clone()
    for layer_i in range(4):
        out, _, _ = model_wrapper(
            cur_history, attention_mask=cur_mask,
            sid_meta=torch.zeros(B, cur_history.shape[1], 4, device=history_tensor.device),
            curvature_meta=model_wrapper.adapter.build_curvature_meta(B),
            labels=cur_history,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        next_logits = logits[:, -1, :]  # (B, vocab_size)
        lo, hi = layer_ranges[layer_i]
        mask = torch.full_like(next_logits, float("-inf"))
        mask[:, lo:hi + 1] = 0.0
        masked_logits = next_logits + mask
        next_token = masked_logits.argmax(dim=-1)  # (B,)
        predicted[:, layer_i] = next_token
        cur_history = torch.cat([cur_history, next_token.unsqueeze(1)], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
    return predicted


def load_data():
    from dataset import GenRecDataset
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n = len(train_ds)
    n_val = int(n * VAL_SPLIT_RATIO)
    n_train = n - n_val
    histories = []
    targets = []
    for i in range(n):
        s = train_ds.data[i]
        histories.append(s["history"])
        targets.append(s["target"])
    return histories[:n_train], targets[:n_train], histories[n_train:], targets[n_train:]


def pad_batch(histories, targets):
    B = len(histories)
    max_L = max(len(h) for h in histories)
    history_padded = np.zeros((B, max_L * 4), dtype=np.int64)
    target_padded = np.zeros((B, 4), dtype=np.int64)
    for i, (h, t) in enumerate(zip(histories, targets)):
        flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
        history_padded[i, :len(flat)] = flat
        target_padded[i] = np.asarray(t, dtype=np.int64).flatten()
    return history_padded, target_padded


def check_mixing_non_degenerate(model_wrapper, eps=1e-3):
    """Precheck: mixing_logits 不能退化为单分支 (e.g. 1 component > 0.99)."""
    with torch.no_grad():
        m = model_wrapper.adapter.get_mixing_per_layer()  # (3, 3)
    per_layer_max = m.max(dim=-1).values.cpu().tolist()
    non_degenerate = all(p < 1.0 - eps for p in per_layer_max)
    return {
        "mixing_per_layer_max_weight": per_layer_max,
        "mixing_non_degenerate": bool(non_degenerate),
        "kappa_per_layer": model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist(),
    }


def run_train():
    # Write TRAINING_PID
    pid = os.getpid()
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(pid) + "\n")

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Issue #15 方向B Step 4 重训] START_EPOCH={START_EPOCH}, NUM_EPOCHS={NUM_EPOCHS}")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[预期] Issue #157 SID hash: {EXPECTED_SID_SHA}")
    log_lines.append(f"[Hash 一致] {sid_sha == EXPECTED_SID_SHA}")
    log_lines.append(f"[NOTE] 从 epoch 0 重训, 不 load PRIOR_CKPT (旧 ckpt 没 kappa_logits / mixing_logits)")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash 不一致, 立即 STOP")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    WrapperCls, get_t5_config = load_wrapper_cls()
    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]
    t5_config = get_t5_config()
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper.to(DEVICE)

    train_hist, train_tgt, val_hist, val_tgt = load_data()
    log_lines.append(f"\n[Data] train={len(train_hist)}, val={len(val_hist)}")

    val_histories_t = []
    val_targets_t = []
    for h, t in zip(val_hist[:VAL_EVAL_N], val_tgt[:VAL_EVAL_N]):
        flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
        val_histories_t.append(flat)
        val_targets_t.append(np.asarray(t, dtype=np.int64).flatten())
    # pad to common length
    max_L = max(len(h) for h in val_histories_t)
    val_hist_arr = np.zeros((len(val_histories_t), max_L), dtype=np.int64)
    for i, h in enumerate(val_histories_t):
        val_hist_arr[i, :len(h)] = h
    val_tgt_arr = np.stack(val_targets_t, axis=0)
    val_histories_t = torch.from_numpy(val_hist_arr).to(DEVICE)
    val_targets_t = torch.from_numpy(val_tgt_arr).to(DEVICE)

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)

    # Param groups: 不同 lr for new params (kappa_logits, mixing_logits) vs conditioner/LN
    new_param_ids = set()
    new_param_ids.add(id(model_wrapper.adapter.kappa_logits))
    new_param_ids.add(id(model_wrapper.adapter.mixing_logits))
    conditioner_params = []
    ln_params = []
    new_params = []
    for name, p in model_wrapper.named_parameters():
        if id(p) in new_param_ids:
            new_params.append(p)
        elif "first_input_ln" in name or "ln" in name.lower():
            ln_params.append(p)
        else:
            conditioner_params.append(p)
    optim = torch.optim.AdamW([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": ln_params, "lr": LR_LAYERNORM},
        {"params": new_params, "lr": LR_NEW_PARAMS},
    ])

    best_val_r10 = 0.0
    best_epoch = -1
    patience_counter = 0
    early_stop_triggered = False
    val_trace = []

    log_lines.append(f"\n[Train] starting {NUM_EPOCHS - START_EPOCH} epochs from epoch {START_EPOCH}")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    t_start = time.time()
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        model_wrapper.train()
        perm = np.random.permutation(len(train_hist))
        total_loss = 0.0
        cond_grad_sum = 0.0
        ln_grad_sum = 0.0
        new_grad_sum = 0.0
        n_batches = 0
        nan_inf_detected = False

        for batch_start in range(0, len(train_hist), BATCH_SIZE):
            batch_idx = perm[batch_start:batch_start + BATCH_SIZE]
            batch_h = [train_hist[i] for i in batch_idx]
            batch_t = [train_tgt[i] for i in batch_idx]
            history_arr, target_arr = pad_batch(batch_h, batch_t)
            history_tensor = torch.from_numpy(history_arr).to(DEVICE)
            target_tensor = torch.from_numpy(target_arr).to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()

            B = history_tensor.shape[0]
            sid_meta = torch.zeros(B, history_tensor.shape[1], 4, device=DEVICE)
            curvature_meta = model_wrapper.adapter.build_curvature_meta(B)

            out, _, alpha = model_wrapper(
                history_tensor, attention_mask=attention_mask,
                sid_meta=sid_meta, curvature_meta=curvature_meta,
                labels=target_tensor,
            )
            loss = out.loss if hasattr(out, "loss") else out[0]
            if not torch.isfinite(loss):
                nan_inf_detected = True
                log_lines.append(f"  [epoch {epoch+1}] NaN/Inf detected at batch {n_batches}, abort")
                break

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            for name, p in model_wrapper.named_parameters():
                if p.grad is not None:
                    if id(p) in new_param_ids:
                        new_grad_sum += p.grad.abs().mean().item()
                    elif "first_input_ln" in name or "ln" in name.lower():
                        ln_grad_sum += p.grad.abs().mean().item()
                    else:
                        cond_grad_sum += p.grad.abs().mean().item()
            n_batches += 1

        if nan_inf_detected:
            break

        avg_loss = total_loss / max(n_batches, 1)
        avg_cond_grad = cond_grad_sum / max(n_batches, 1)
        avg_ln_grad = ln_grad_sum / max(n_batches, 1)
        avg_new_grad = new_grad_sum / max(n_batches, 1)
        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1 - START_EPOCH + 1) * (NUM_EPOCHS - epoch - 1)
        msg = f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, new_param_grad={avg_new_grad:.4e}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

        # Per-epoch ckpt (R12: 删旧留新)
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
        }, ADAPTER_CKPT_PATH)

        # Val eval (per-epoch)
        val_r10, val_in_range_pct = run_val_eval(model_wrapper, layer_ranges, val_histories_t, val_targets_t, val_n=VAL_EVAL_N)
        mixing_audit = check_mixing_non_degenerate(model_wrapper)
        log_lines.append(f"  [val @ epoch {epoch+1}] val_R@10={val_r10:.4f}, in-range={val_in_range_pct*100:.1f}%, mixing_audit_max_weight={mixing_audit['mixing_per_layer_max_weight']}")
        val_trace.append({
            "epoch": epoch + 1, "val_r10": val_r10, "val_in_range_pct": val_in_range_pct,
            "val_n": VAL_EVAL_N,
            "kappa_per_layer": mixing_audit["kappa_per_layer"],
            "mixing_per_layer_max_weight": mixing_audit["mixing_per_layer_max_weight"],
            "mixing_non_degenerate": mixing_audit["mixing_non_degenerate"],
            "timestamp_unix": time.time(),
        })

        if val_r10 > best_val_r10:
            best_val_r10 = val_r10
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "adapter_state_dict": model_wrapper.adapter.state_dict(),
                "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                "epoch": epoch + 1,
                "val_r10": val_r10,
            }, BEST_ADAPTER_CKPT_PATH)
            log_lines.append(f"  [BEST] new best at epoch {epoch+1}, val_R@10={val_r10:.4f}")
        else:
            patience_counter += 1

        val_msg = f"  [val @ epoch {epoch+1}] val_R@10={val_r10:.4f}, best={best_val_r10:.4f} @ epoch {best_epoch}, patience={patience_counter}/{EARLY_STOP_PATIENCE}"
        print(val_msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(val_msg + "\n")

        # Save val_trace (per epoch, so progress is never lost)
        with open(VAL_TRACE_PATH, "w") as f:
            json.dump({"val_trace": val_trace, "best_val_r10": best_val_r10, "best_epoch": best_epoch,
                       "early_stop_triggered": early_stop_triggered}, f, indent=2)

        if patience_counter >= EARLY_STOP_PATIENCE:
            early_stop_triggered = True
            es_msg = f"  [EarlyStop TRIGGERED @ epoch {epoch+1}] patience={patience_counter} >= {EARLY_STOP_PATIENCE}"
            print(es_msg, flush=True)
            with open(LOG_PATH, "a") as f:
                f.write(es_msg + "\n")
            break

    # Save final state
    summary_path = PRODUCT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "issue": 15,
            "step": "Step 4 Stage3 retrain (含 learnable kappa_logits / mixing_logits)",
            "best_val_r10": best_val_r10,
            "best_epoch": best_epoch,
            "early_stop_triggered": early_stop_triggered,
            "total_epochs_run": len(val_trace),
            "elapsed_min": (time.time() - t_start) / 60,
            "val_trace": val_trace,
        }, f, indent=2)
    print(f"\n[Training complete] best_val_r10={best_val_r10} @ epoch {best_epoch}, early_stop={early_stop_triggered}", flush=True)


def run_val_eval(model_wrapper, layer_ranges, val_histories_t, val_targets_t, val_n=1000, batch_size=32):
    model_wrapper.eval()
    n = min(val_n, val_histories_t.shape[0])
    val_r10 = 0.0
    in_range_count = 0
    sample_idx = torch.randperm(val_histories_t.shape[0], generator=torch.Generator().manual_seed(SEED + 13))[:n]
    histories = val_histories_t[sample_idx]
    targets = val_targets_t[sample_idx]
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            B = batch_end - batch_start
            history_tensor = histories[batch_start:batch_end]
            target_tensor = targets[batch_start:batch_end]
            attention_mask = (history_tensor != PAD_TOKEN).long()
            predicted = autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges)
            for i in range(B):
                pred = predicted[i].cpu().tolist()
                tgt = target_tensor[i].cpu().tolist()
                preds_all.append(pred[:4])
                targets_all.append(tgt[:4])
                for layer_i, token_id in enumerate(pred[:4]):
                    lo, hi = layer_ranges[layer_i]
                    if lo <= token_id <= hi:
                        in_range_count += 1
    r10 = float(np.mean([1.0 if p == t else 0.0 for p, t in zip(preds_all, targets_all)]))
    total_tokens = len(preds_all) * 4
    in_range_pct = in_range_count / total_tokens if total_tokens > 0 else 0.0
    return r10, in_range_pct


if __name__ == "__main__":
    # Clean old PID file
    if TRAINING_PID_FILE.exists():
        TRAINING_PID_FILE.unlink()
    run_train()