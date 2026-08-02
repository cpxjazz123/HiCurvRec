#!/usr/bin/env python3
"""Issue #17 [方向A Step 5] 单变量实验: option_A_extend_epoch.

Per #17 spec Step 5: 选 1 个变量做单变量实验, 预先写 pass/fail 判据, 禁止一次改多变量.
本脚本: 续训 taskA_stage3_issue192_long_run/best_adapter.pt (epoch 48, val_R@10=0.058) 从 epoch 49 → 199.

预定义 pass/fail 判据 (写在本脚本 docstring + training_summary):
- ✅ 单变量提升 PASS: 任一 epoch val_R@10 > 0.07 (baseline 0.058 当前 best + 20%)
- ❌ 单变量提升 FAIL: 199 epoch 跑完 best val_R@10 ≤ 0.07
- 中间结果每 epoch 落盘 val_trace.json + adapter.pt + best_adapter.pt

严禁条款:
- 不改 curvature_meta / 协议 (跟原 #192 long-run 完全一致)
- 不解冻 T5
- 不改 κ / mixing / conditioner 结构
- 只改 epoch 数 (从 48 → 199)

输出位置: taskA/stage3/taskA_stage3_issue17_step5/
- adapter.pt (per-epoch save, R12 删旧留新)
- best_adapter.pt (best val_R@10 save)
- val_trace.json
- training_summary.json (含 pass/fail verdict)
- _TRAINING_PID
- log via taskA/_logs/task_issue17_step5.log
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
# Config (per Issue #17 spec Step 5 option_A_extend_epoch)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"  # run_in_background 启动时会通过 CUDA_VISIBLE_DEVICES 切到 GPU 1
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
START_EPOCH = 49  # 从 best_adapter.pt (epoch 48) 之后开始
NUM_EPOCHS = 199  # 跟 #192 long-run 目标一致
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_MAX = 0.5

# 单变量判据 (per #17 spec)
PASS_THRESHOLD = 0.07  # val_R@10 > 0.07 算单变量提升 PASS (baseline best 0.058 + 20%)

# Val + early stop 配置 (跟 #192 long-run 一致)
VAL_SPLIT_RATIO = 0.10
VAL_EVAL_N = 1000
EARLY_STOP_PATIENCE = 10

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/train.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"
PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"  # 加载 epoch 48 best

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue17_step5")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue17_step5.log"
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
        "t470",
        "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_kappa_scale_recontinue.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HG_Rec_with_BoundedAdapter, mod.get_t5_config


def autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges):
    """跟 #192 long-run / Stage4 同路径: 自回归 4-forwards + 闭区间 mask."""
    B = history_tensor.shape[0]
    predicted = torch.zeros(B, 4, dtype=torch.long, device=history_tensor.device)
    cur_history = history_tensor.clone()
    cur_mask = attention_mask.clone()
    for layer_i in range(4):
        out, _, _ = model_wrapper(
            cur_history, attention_mask=cur_mask,
            sid_meta=torch.zeros(B, cur_history.shape[1], 4, device=history_tensor.device),
            kappa_meta=torch.zeros(B, 3, device=history_tensor.device),
            scale_meta=torch.ones(B, 3, device=history_tensor.device),
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
    log_lines.append(f"[Issue #17 方向A Step 5 option_A_extend_epoch] START_EPOCH={START_EPOCH}, NUM_EPOCHS={NUM_EPOCHS}, PASS_THRESHOLD={PASS_THRESHOLD}")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    prior_sha = sha256_of(PRIOR_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] PRIOR_CKPT (epoch 48 best): {prior_sha}")
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

    # Load prior ckpt (epoch 48 best, val_R@10=0.058)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    if "adapter_state_dict" in prior_ckpt:
        model_wrapper.adapter.load_state_dict(prior_ckpt["adapter_state_dict"])
    else:
        model_wrapper.adapter.load_state_dict(prior_ckpt)
    if "ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])
    elif "first_input_ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["first_input_ln_state_dict"])
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch', 'unknown')}, val_R@10={prior_ckpt.get('val_r10', 'unknown')}, alpha_value={prior_ckpt.get('alpha_value', 'unknown')}")

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

    conditioner_params = list(model_wrapper.adapter.parameters())
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    best_val_r10 = prior_ckpt.get("val_r10", 0.0)  # 从 epoch 48 的 0.058 开始
    best_epoch = prior_ckpt.get("epoch", 48)
    initial_best = best_val_r10
    patience_counter = 0
    early_stop_triggered = False
    val_trace = []
    log_lines.append(f"\n[Train] continuing from epoch {START_EPOCH} to {NUM_EPOCHS}, initial best_val_R@10={best_val_r10}")
    log_lines.append(f"[PASS_THRESHOLD] val_R@10 > {PASS_THRESHOLD} = 单变量提升 PASS")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    t_start = time.time()
    pass_achieved = False
    pass_epoch = None

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
        n_batches = 0
        for batch_start in range(0, len(train_hist), BATCH_SIZE):
            batch_idx = perm[batch_start:batch_start + BATCH_SIZE]
            batch_h = [train_hist[i] for i in batch_idx]
            batch_t = [train_tgt[i] for i in batch_idx]
            history_arr, target_arr = pad_batch(batch_h, batch_t)
            history_tensor = torch.from_numpy(history_arr).to(DEVICE)
            target_tensor = torch.from_numpy(target_arr).to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            B = history_tensor.shape[0]
            out, _, alpha = model_wrapper(
                history_tensor, attention_mask=attention_mask,
                sid_meta=torch.zeros(B, history_tensor.shape[1], 4, device=DEVICE),
                kappa_meta=torch.zeros(B, 3, device=DEVICE),
                scale_meta=torch.ones(B, 3, device=DEVICE),
                labels=target_tensor,
            )
            loss = out.loss if hasattr(out, "loss") else out[0]
            if not torch.isfinite(loss):
                log_lines.append(f"  [epoch {epoch+1}] NaN/Inf at batch {n_batches}, abort")
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for p in conditioner_params:
                if p.grad is not None:
                    cond_grad_sum += p.grad.abs().mean().item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        avg_cond_grad = cond_grad_sum / max(n_batches, 1)
        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1 - START_EPOCH + 1) * (NUM_EPOCHS - epoch - 1)
        msg = f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, α={alpha.item():.6e}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
            "alpha_value": alpha.item() if hasattr(alpha, "item") else float(alpha),
        }, ADAPTER_CKPT_PATH)

        val_r10, val_in_range_pct = run_val_eval_inline(model_wrapper, layer_ranges, val_histories_t, val_targets_t, val_n=VAL_EVAL_N)
        val_trace.append({
            "epoch": epoch + 1, "val_r10": val_r10, "val_in_range_pct": val_in_range_pct,
            "val_n": VAL_EVAL_N,
            "timestamp_unix": time.time(),
        })

        if val_r10 > best_val_r10:
            best_val_r10 = val_r10
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "adapter_state_dict": model_wrapper.adapter.state_dict(),
                "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                "epoch": epoch + 1,
                "val_r10": val_r10,
                "alpha_value": alpha.item() if hasattr(alpha, "item") else float(alpha),
            }, BEST_ADAPTER_CKPT_PATH)
            log_lines.append(f"  [BEST] new best at epoch {epoch+1}, val_R@10={val_r10:.4f}")
            if val_r10 > PASS_THRESHOLD and not pass_achieved:
                pass_achieved = True
                pass_epoch = epoch + 1
                log_lines.append(f"  [PASS_THRESHOLD] val_R@10={val_r10:.4f} > {PASS_THRESHOLD} @ epoch {epoch+1} ✅ 单变量提升达成")
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
                "pass_threshold": PASS_THRESHOLD, "pass_achieved": pass_achieved, "pass_epoch": pass_epoch,
                "initial_best_val_r10": initial_best,
            }, f, indent=2)

        if patience_counter >= EARLY_STOP_PATIENCE:
            early_stop_triggered = True
            es_msg = f"  [EarlyStop TRIGGERED @ epoch {epoch+1}] patience={patience_counter} >= {EARLY_STOP_PATIENCE}"
            print(es_msg, flush=True)
            with open(LOG_PATH, "a") as f:
                f.write(es_msg + "\n")
            break

    # Final summary
    improvement = best_val_r10 - initial_best
    improvement_pct = improvement / initial_best * 100 if initial_best > 0 else 0
    pass_fail = "PASS" if (pass_achieved or best_val_r10 > PASS_THRESHOLD) else "FAIL"
    summary = {
        "issue": 17,
        "step": "Step 5 single-variable experiment — option_A_extend_epoch",
        "start_epoch": START_EPOCH,
        "num_epochs": NUM_EPOCHS,
        "epochs_actually_run": len(val_trace),
        "initial_best_val_r10_epoch48": initial_best,
        "best_val_r10": best_val_r10,
        "best_epoch": best_epoch,
        "improvement_abs": improvement,
        "improvement_pct": improvement_pct,
        "pass_threshold": PASS_THRESHOLD,
        "pass_achieved": pass_achieved,
        "pass_epoch": pass_epoch,
        "single_variable_verdict": pass_fail,
        "early_stop_triggered": early_stop_triggered,
        "elapsed_min": (time.time() - t_start) / 60,
        "val_trace": val_trace,
    }
    with open(TRAINING_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Training complete] best_val_r10={best_val_r10} @ epoch {best_epoch}, single_variable_verdict={pass_fail}", flush=True)


if __name__ == "__main__":
    if TRAINING_PID_FILE.exists():
        TRAINING_PID_FILE.unlink()
    run_train()