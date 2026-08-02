#!/usr/bin/env python3
"""Issue #18 [方向B Gate3] 单变量重训 — option_A 续训旧 best_adapter.

per #18 spec:
- 基础: 沿用 #193 long-run 已通过协议的 Stage3 ckpt (epoch 50, val_R@10=0.058)
- 单变量: option_A 续训 (从 epoch 51 继续, 短程 5 epoch sanity 先做, 诊断三层κ/mixing/dL/dκ/dL/dmix/grad/val_R@10/loss/ckpt/SID hash)
- 严禁: 多变量同时改, 多seed/双复跑, from-0 + 小变化 (per #15 R23 NO-GO), 跳过 Gate3 诊断直接长跑
- 触发 R23 NO-GO 立即 kill: val_R@10=0 连续2epoch / 梯度全零或爆炸 / 三层参数同步 / 三层框架违反

ckpt: taskB_stage3_issue193_long_run/best_adapter.pt (epoch 50 val_R@10=0.058)
新参数: kappa_logits (3,) + mixing_logits (3,3) 在 #15 Step 1 加的 nn.Parameter, 旧 ckpt 不含
→ load_state_dict 用 strict=False, 新 params 保持 init (kappa_init_logit=0.5413 → softplus≈1.0, mixing=zeros → softmax=[1/3,1/3,1/3])

输出: taskB/stage3/taskB_stage3_issue18_continue/
- adapter.pt (per-epoch save, R12 删旧留新)
- best_adapter.pt (best val_R@10 save)
- val_trace.json (含 κ per-layer / mixing per-layer / dL/dκ / dL/dmix)
- training_summary.json (含 verdict)
- _TRAINING_PID
- log: taskB/_logs/task_issue18_continue.log
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
# Config (per Issue #18 spec option_A 续训)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
START_EPOCH = 51  # 从 best_adapter.pt (epoch 50) 之后开始
NUM_EPOCHS = 60   # 短程 sanity 先 5 epoch, 通过则继续 (per spec '先完成短程单seed sanity')
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
LR_NEW_PARAMS = 1e-3  # kappa_logits / mixing_logits 学习率

VAL_SPLIT_RATIO = 0.10
VAL_EVAL_N = 1000
EARLY_STOP_PATIENCE = 10

# Issue #18 诊断关键参数 (per spec)
RECORD_PER_EPOCH_GRADS = True  # 记录 dL/dκ, dL/dmix

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/train.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_ckpt/HG_Rec_best.pth"
PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue193_long_run/best_adapter.pt"

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue18_continue")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue18_continue.log"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
TRAINING_SUMMARY_PATH = PRODUCT_DIR / "training_summary.json"


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
        "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/_archive/taskB_stage3_mixed_curv_recontinue.py",
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
        next_logits = logits[:, -1, :]
        lo, hi = layer_ranges[layer_i]
        mask = torch.full_like(next_logits, float("-inf"))
        mask[:, lo:hi + 1] = 0.0
        masked_logits = next_logits + mask
        next_token = masked_logits.argmax(dim=-1)
        predicted[:, layer_i] = next_token
        cur_history = torch.cat([cur_history, next_token.unsqueeze(1)], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
    return predicted


def run_train():
    pid = os.getpid()
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(pid) + "\n")

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Issue #18 方向B Gate3 option_A 续训] START_EPOCH={START_EPOCH}, NUM_EPOCHS={NUM_EPOCHS}")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    prior_sha = sha256_of(PRIOR_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] PRIOR_CKPT: {prior_sha}")
    log_lines.append(f"[预期] Issue #157 SID hash: {EXPECTED_SID_SHA}")
    log_lines.append(f"[Hash 一致] {sid_sha == EXPECTED_SID_SHA}")

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

    # Load prior ckpt with strict=False (新参数 kappa_logits/mixing_logits 旧 ckpt 不含)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch', 'unknown')}, val_R@10={prior_ckpt.get('val_r10', 'unknown')}")
    missing_keys, unexpected_keys = model_wrapper.adapter.load_state_dict(prior_ckpt["adapter_state_dict"], strict=False)
    log_lines.append(f"[Load adapter] missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
    log_lines.append(f"  ↳ missing keys = 新 nn.Parameter kappa_logits/mixing_logits (per #15 Step 1 修复, init 保持)")
    if "ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])
    elif "first_input_ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["first_input_ln_state_dict"])
    log_lines.append(f"[Initial κ per-layer] {model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()}")
    log_lines.append(f"[Initial mixing per-layer] {model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist()}")

    # Data
    from dataset import GenRecDataset
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n = len(train_ds)
    n_val = int(n * VAL_SPLIT_RATIO)
    all_h = [train_ds.data[i]["history"] for i in range(n)]
    all_t = [train_ds.data[i]["target"] for i in range(n)]
    train_hist, train_tgt = all_h[:n - n_val], all_t[:n - n_val]
    val_hist, val_tgt = all_h[n - n_val:], all_t[n - n_val:]
    log_lines.append(f"\n[Data] train={len(train_hist)}, val={len(val_hist)}")

    val_hist_t = []
    val_tgt_t = []
    for h, t in zip(val_hist[:VAL_EVAL_N], val_tgt[:VAL_EVAL_N]):
        flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
        val_hist_t.append(flat)
        val_tgt_t.append(np.asarray(t, dtype=np.int64).flatten())
    max_L = max(len(h) for h in val_hist_t)
    val_hist_arr = np.zeros((len(val_hist_t), max_L), dtype=np.int64)
    for i, h in enumerate(val_hist_t):
        val_hist_arr[i, :len(h)] = h
    val_tgt_arr = np.stack(val_tgt_t, axis=0)
    val_histories_t = torch.from_numpy(val_hist_arr).to(DEVICE)
    val_targets_t = torch.from_numpy(val_tgt_arr).to(DEVICE)

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)
    log_lines.append(f"[Layer ranges] {layer_ranges}")

    # Param groups
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

    best_val_r10 = prior_ckpt.get("val_r10", 0.0)
    best_epoch = prior_ckpt.get("epoch", 50)
    initial_best = best_val_r10
    patience_counter = 0
    early_stop_triggered = False
    val_trace = []
    log_lines.append(f"\n[Train] continuing from epoch {START_EPOCH} to {NUM_EPOCHS}, initial best_val_R@10={best_val_r10}")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    t_start = time.time()
    epoch_51_loss = None  # 用于 sanity 检查

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

    def run_val_eval_inline(model_wrapper, layer_ranges, val_histories_t, val_targets_t, val_n=1000, batch_size=32):
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

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        model_wrapper.train()
        perm = np.random.permutation(len(train_hist))
        total_loss = 0.0
        cond_grad_sum = 0.0
        ln_grad_sum = 0.0
        kappa_grad_sum = 0.0
        mixing_grad_sum = 0.0
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
                    if p is model_wrapper.adapter.kappa_logits:
                        kappa_grad_sum += p.grad.abs().mean().item()
                    elif p is model_wrapper.adapter.mixing_logits:
                        mixing_grad_sum += p.grad.abs().mean().item()
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
        avg_kappa_grad = kappa_grad_sum / max(n_batches, 1)
        avg_mixing_grad = mixing_grad_sum / max(n_batches, 1)

        # 三层 κ/mixing 当前值
        kappa_l = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
        mixing_l = model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist()
        # 三层 κ 是否同步 (R23 触发: 三层参数同步)
        kappa_synced = max(kappa_l) - min(kappa_l) < 1e-6
        # 三层 mixing 是否退化
        mixing_max_per_layer = [max(row) for row in mixing_l]
        mixing_degenerate = any(m > 0.99 for m in mixing_max_per_layer)

        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1 - START_EPOCH + 1) * (NUM_EPOCHS - epoch - 1)
        msg = (f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, "
               f"kappa_grad={avg_kappa_grad:.4e}, mixing_grad={avg_mixing_grad:.4e}, "
               f"kappa={kappa_l}, mixing_max={mixing_max_per_layer}, "
               f"synced={kappa_synced}, degenerate={mixing_degenerate}, "
               f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m")
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
        }, ADAPTER_CKPT_PATH)

        val_r10, val_in_range_pct = run_val_eval_inline(model_wrapper, layer_ranges, val_histories_t, val_targets_t, val_n=VAL_EVAL_N)
        val_trace.append({
            "epoch": epoch + 1, "val_r10": val_r10, "val_in_range_pct": val_in_range_pct,
            "val_n": VAL_EVAL_N,
            "kappa_per_layer": kappa_l,
            "mixing_per_layer": mixing_l,
            "mixing_max_per_layer": mixing_max_per_layer,
            "kappa_synced": kappa_synced,
            "mixing_degenerate": mixing_degenerate,
            "grad_summary": {
                "cond_grad": avg_cond_grad, "ln_grad": avg_ln_grad,
                "kappa_grad": avg_kappa_grad, "mixing_grad": avg_mixing_grad,
            },
            "loss": avg_loss,
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

        val_msg = f"  [val @ epoch {epoch+1}] val_R@10={val_r10:.4f}, in-range={val_in_range_pct*100:.1f}%, best={best_val_r10:.4f} @ epoch {best_epoch}, patience={patience_counter}/{EARLY_STOP_PATIENCE}"
        print(val_msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(val_msg + "\n")

        with open(VAL_TRACE_PATH, "w") as f:
            json.dump({
                "val_trace": val_trace, "best_val_r10": best_val_r10, "best_epoch": best_epoch,
                "early_stop_triggered": early_stop_triggered,
                "initial_best_val_r10": initial_best,
            }, f, indent=2)

        # R23 触发检查 (per #18 spec)
        r23_trigger = False
        r23_reasons = []
        # 1. val_R@10=0 连续 ≥2 epoch
        if len(val_trace) >= 2 and val_trace[-1]["val_r10"] == 0 and val_trace[-2]["val_r10"] == 0:
            r23_trigger = True
            r23_reasons.append("val_R@10=0 连续 ≥2 epoch")
        # 2. 梯度全零 / 爆炸
        if avg_kappa_grad == 0 or avg_kappa_grad > 100 or avg_mixing_grad == 0 or avg_mixing_grad > 100:
            r23_trigger = True
            r23_reasons.append(f"kappa_grad={avg_kappa_grad:.4e} 或 mixing_grad={avg_mixing_grad:.4e} 异常")
        # 3. 三层 κ 同步
        if kappa_synced:
            r23_trigger = True
            r23_reasons.append(f"三层 κ 同步 ({kappa_l})")
        # 4. mixing 退化
        if mixing_degenerate:
            r23_trigger = True
            r23_reasons.append(f"mixing 退化 (max={mixing_max_per_layer})")

        if r23_trigger:
            log_lines.append(f"  [R23 TRIGGERED] {r23_reasons}")
            print(f"  [R23 TRIGGERED] {r23_reasons}", flush=True)
            with open(LOG_PATH, "a") as f:
                f.write(f"  [R23 TRIGGERED] {r23_reasons}\n")
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            early_stop_triggered = True
            es_msg = f"  [EarlyStop TRIGGERED @ epoch {epoch+1}] patience={patience_counter} >= {EARLY_STOP_PATIENCE}"
            print(es_msg, flush=True)
            with open(LOG_PATH, "a") as f:
                f.write(es_msg + "\n")
            break

    # Final summary
    summary = {
        "issue": 18,
        "variable": "option_A 续训旧 best_adapter (epoch 50 → NUM_EPOCHS=60)",
        "start_epoch": START_EPOCH,
        "num_epochs": NUM_EPOCHS,
        "epochs_actually_run": len(val_trace),
        "initial_best_val_r10_epoch50": initial_best,
        "best_val_r10": best_val_r10,
        "best_epoch": best_epoch,
        "improvement_abs": best_val_r10 - initial_best,
        "improvement_pct": (best_val_r10 - initial_best) / initial_best * 100 if initial_best > 0 else 0,
        "early_stop_triggered": early_stop_triggered,
        "r23_triggered": r23_trigger if 'r23_trigger' in dir() else False,
        "r23_reasons": r23_reasons if 'r23_reasons' in dir() else [],
        "elapsed_min": (time.time() - t_start) / 60,
        "val_trace": val_trace,
    }
    with open(TRAINING_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Training complete] best_val_r10={best_val_r10} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    if TRAINING_PID_FILE.exists():
        TRAINING_PID_FILE.unlink()
    run_train()