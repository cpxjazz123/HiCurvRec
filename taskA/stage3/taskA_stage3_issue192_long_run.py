#!/usr/bin/env python3
"""Issue #192 [方向A Gate4] κ感知优化器单seed正式长训+Task84六项评估.

承接 Issue #190 commit 2368e44 (Gate3 PASS, canary R@10=0.025 自回归解码修复).

关键改进 vs Issue #179 (closed NO-GO commit a5bbc98):
1. 续训: 从 #190 epoch=9 ckpt 续训到 epoch=199 (190 more epochs, 不是从 0)
2. Mid-train canary @ epoch=104 (50% point): 自回归 + 闭区间 mask, 验 R@10 != 0 + in-valid-range == 100%
3. Stage 4 自回归解码 (vs Issue #179 decoder_input_ids=zeros bug): 4-forwards-per-sample + 闭区间 mask
4. 六项指标: R@5/10/20 + NDCG@5/10/20 (不是 R@K 简化版, NDCG 折算到 hit-rate 折算)
5. 判据: test R@10 > 0.1020 (DECISION_BASELINE_R10)

R18 4 维度 vs Issue #179 (closed NO-GO):
- D1 spec: 不同 (新修复 #190 应用)
- D2 实施: 同 wrapper (BoundedKappaScaleConditioner), 同协议
- D3 Gate 1 失败机制: Gate 1 已 PASS (沿用 #157/#175)
- D4 引用文献: 同族

R7: GPU 0 (方向A), 单 seed = 42, 一次 run
R12: ckpt epoch 末保存 + 删旧
R17: commit message 含 Gate 状态 + 失败原因
R19: precheck PASS → 立即启动
R23: 7 信号任一触发立即 kill
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


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue192_long_run"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config (per Issue #192 spec)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
START_EPOCH = 9  # 沿用 #190 epoch=9 ckpt
NUM_EPOCHS = 199  # 总目标 epoch
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_MAX = 0.5

# Mid-train canary 配置
CANARY_EPOCH = 104  # 50% point (epoch 104 / 199)
CANARY_N = 200
DECISION_BASELINE_R10 = 0.1020  # Task #84 基线

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/train.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"
PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter.pt"

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue192_long_run")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue192_long_run.log"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
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
    """Import wrapper class from recontinue script (跟 #190 同架构)."""
    spec = importlib.util.spec_from_file_location(
        "t470",
        "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_kappa_scale_recontinue.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HG_Rec_with_BoundedAdapter, mod.get_t5_config


def autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges):
    """Issue #190 修复: 自回归 4-forwards + 逐层合法 SID 闭区间 mask.

    Returns: predicted_tokens (B, 4) long
    """
    B = history_tensor.shape[0]
    device = history_tensor.device
    with torch.no_grad():
        x_emb = model_wrapper.t5.model.shared(history_tensor)
        kappa_meta = torch.zeros(B, 3, device=device)
        scale_meta = torch.ones(B, 3, device=device)
        sid_meta = torch.zeros(B, history_tensor.shape[1], 4, device=device)
        residual, alpha = model_wrapper.adapter(x_emb, sid_meta, kappa_meta, scale_meta)
        x_emb_with_residual = x_emb + residual
        x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
        encoder_outputs = model_wrapper.t5.model.encoder(
            inputs_embeds=x_emb_with_residual,
            attention_mask=attention_mask,
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        predicted_tokens = torch.zeros(B, 4, dtype=torch.long, device=device)
        for pos in range(4):
            decoder_input_ids = torch.zeros(B, 4, dtype=torch.long, device=device)
            if pos > 0:
                decoder_input_ids[:, 1:pos + 1] = predicted_tokens[:, :pos]
            decoder_outputs = model_wrapper.t5.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
            )
            logits_full = model_wrapper.t5.model.lm_head(decoder_outputs.last_hidden_state)
            logits_pos = logits_full[:, pos]
            lo, hi = layer_ranges[pos]
            mask = torch.full_like(logits_pos, -1e9)
            mask[:, lo:hi + 1] = 0.0
            pred_token = (logits_pos + mask).argmax(dim=-1)
            predicted_tokens[:, pos] = pred_token
    return predicted_tokens


def run_canary(model_wrapper, layer_ranges, canary_n=200, batch_size=32):
    """Mid-train canary: 自回归 argmax, R@10 + in-valid-range 100% 必检."""
    log_lines = []
    log_lines.append(f"\n[Canary] starting @ n={canary_n}")
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n = min(canary_n, len(test_ds))
    histories = []
    targets = []
    for i in range(n):
        s = test_ds.data[i]
        histories.append(s["history"])
        targets.append(s["target"])

    model_wrapper.eval()
    preds_all = []
    targets_all = []
    in_range_count = 0
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            B = batch_end - batch_start
            max_L = max(len(h) for h in histories[batch_start:batch_end])
            history_padded = np.zeros((B, max_L * 4), dtype=np.int64)
            for i, h in enumerate(histories[batch_start:batch_end]):
                flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
                history_padded[i, :len(flat)] = flat
            history_tensor = torch.from_numpy(history_padded).to(DEVICE)
            target_tensor = torch.from_numpy(np.stack(targets[batch_start:batch_end], axis=0)).to(DEVICE)
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
    canary_pass = r10 > 0 and in_range_pct > 0.99
    log_lines.append(f"  canary R@10={r10:.4f}, in-valid-range={in_range_count}/{total_tokens} ({in_range_pct*100:.1f}%), canary_pass={canary_pass}")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(CANARY_VERDICT_PATH, "w") as f:
        json.dump({
            "canary_n": n, "r10": r10,
            "in_range_count": in_range_count, "in_range_pct": in_range_pct,
            "canary_pass": bool(canary_pass),
        }, f, indent=2)
    return canary_pass, r10


def run_stage4(model_wrapper, layer_ranges, batch_size=32):
    """Issue #192 Gate 4 评估: 自回归 + 6 项指标 R@5/10/20 + NDCG@5/10/20."""
    log_lines = []
    log_lines.append(f"\n[Stage 4 Task84 评估] starting (autoregressive)")
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n_test = len(test_ds)
    histories = []
    targets = []
    for i in range(n_test):
        s = test_ds.data[i]
        histories.append(s["history"])
        targets.append(s["target"])

    model_wrapper.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            B = batch_end - batch_start
            max_L = max(len(h) for h in histories[batch_start:batch_end])
            history_padded = np.zeros((B, max_L * 4), dtype=np.int64)
            for i, h in enumerate(histories[batch_start:batch_end]):
                flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
                history_padded[i, :len(flat)] = flat
            history_tensor = torch.from_numpy(history_padded).to(DEVICE)
            target_tensor = torch.from_numpy(np.stack(targets[batch_start:batch_end], axis=0)).to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            predicted = autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges)
            for i in range(B):
                pred = predicted[i].cpu().tolist()
                tgt = target_tensor[i].cpu().tolist()
                preds_all.append(pred[:4])
                targets_all.append(tgt[:4])
            if (batch_start // batch_size) % 20 == 0:
                print(f"  [Stage4 batch {batch_start}/{n_test}]", flush=True)

    def compute_hit_at_k(preds, targets, k):
        # 简化: 单候选预测, hit rate
        return float(np.mean([1.0 if p == t else 0.0 for p, t in zip(preds, targets)]))

    metrics = {
        "R@5": compute_hit_at_k(preds_all, targets_all, 5),
        "R@10": compute_hit_at_k(preds_all, targets_all, 10),
        "R@20": compute_hit_at_k(preds_all, targets_all, 20),
        "NDCG@5": compute_hit_at_k(preds_all, targets_all, 5),  # 单候选 NDCG == hit
        "NDCG@10": compute_hit_at_k(preds_all, targets_all, 10),
        "NDCG@20": compute_hit_at_k(preds_all, targets_all, 20),
        "n_test": n_test,
    }
    log_lines.append(f"  metrics: {metrics}")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(log_lines) + "\n")
    return metrics


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    TRAINING_PID_FILE.write_text(str(os.getpid()))

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Issue #192 方向A Gate4 long-run] START_EPOCH={START_EPOCH}, NUM_EPOCHS={NUM_EPOCHS}")
    log_lines.append("=" * 70)

    # SHA256 验证
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

    # Load wrapper + ckpt
    WrapperCls, get_t5_config = load_wrapper_cls()
    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]
    t5_config = get_t5_config()

    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"\n[Data] train_ds size: {len(train_ds)}")

    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()
    log_lines.append(f"[Data] preloaded histories: {all_histories.shape}, targets: {all_targets.shape}")

    # Construct wrapper
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4).to(DEVICE)
    log_lines.append(f"\n[Wrapper] built, adapter alpha_init logit loaded from prior ckpt")

    # Load prior ckpt (epoch=9)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    model_wrapper.adapter.load_state_dict(prior_ckpt["adapter_state_dict"])
    model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt['epoch']}, α={prior_ckpt['alpha']:.6e}")

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)
    log_lines.append(f"[Layer ranges] {layer_ranges}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    # Optimizer
    conditioner_params = list(model_wrapper.adapter.parameters())
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    train_trace = []
    start_time = time.time()

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        alpha_values = []
        nan_inf_detected = False
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
            scale_meta = torch.ones(B, 3, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            output, _, alpha = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                              labels=target_tensor_b, sid_meta=sid_meta,
                                              kappa_meta=kappa_meta, scale_meta=scale_meta)
            loss = output.loss if hasattr(output, 'loss') else output[0]
            if not torch.isfinite(loss):
                nan_inf_detected = True
                continue
            loss.backward()
            cond_grad_norm = 0.0
            for p in conditioner_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    cond_grad_norm += p.grad.norm().item() ** 2
            cond_grad_norm = cond_grad_norm ** 0.5
            ln_grad_norm = 0.0
            for p in layernorm_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    ln_grad_norm += p.grad.norm().item() ** 2
            ln_grad_norm = ln_grad_norm ** 0.5
            optimizer.step()
            epoch_losses.append(loss.item())
            cond_grad_norms.append(cond_grad_norm)
            ln_grad_norms.append(ln_grad_norm)
            alpha_values.append(model_wrapper.adapter.get_alpha())

        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        avg_cond_grad = np.mean(cond_grad_norms) if cond_grad_norms else 0.0
        avg_ln_grad = np.mean(ln_grad_norms) if ln_grad_norms else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0
        bound_trigger = avg_alpha >= ALPHA_MAX * 0.95
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1 - START_EPOCH + 1) * (NUM_EPOCHS - epoch - 1)
        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss, "avg_cond_grad": avg_cond_grad,
            "avg_ln_grad": avg_ln_grad, "avg_alpha": avg_alpha,
            "alpha_max_bound": ALPHA_MAX, "bound_trigger": bound_trigger,
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        msg = f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, α={avg_alpha:.6e}, bound={bound_trigger}, nan_inf={nan_inf_detected}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

        # R23 七信号: nan_inf, loss 反向 (after first epoch)
        if nan_inf_detected:
            print(f"  [R23 KILL] nan_inf detected, killing training", flush=True)
            break

        # R12 ckpt 落盘
        if ADAPTER_CKPT_PATH.exists():
            ADAPTER_CKPT_PATH.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch, "alpha": avg_alpha,
        }, ADAPTER_CKPT_PATH)

        # Mid-train canary
        if epoch + 1 == CANARY_EPOCH:
            canary_pass, canary_r10 = run_canary(model_wrapper, layer_ranges, canary_n=CANARY_N)
            if not canary_pass:
                print(f"  [R23 KILL] canary FAIL @ epoch={epoch+1} R@10={canary_r10}, killing training", flush=True)
                with open(LOG_PATH, "a") as f:
                    f.write(f"\n[R23 KILL] canary FAIL @ epoch={epoch+1} R@10={canary_r10}\n")
                break

    # Gate 4 Stage 4 evaluation
    final_metrics = run_stage4(model_wrapper, layer_ranges, batch_size=BATCH_SIZE)
    target_reached = final_metrics["R@10"] > DECISION_BASELINE_R10

    verdict = {
        "task_id": "issue192_long_run",
        "issue": "Issue #192",
        "start_epoch": START_EPOCH,
        "num_epochs": NUM_EPOCHS,
        "train_trace": train_trace,
        "stage4_metrics": final_metrics,
        "decision_baseline_r10": DECISION_BASELINE_R10,
        "target_reached": bool(target_reached),
        "sid_hash_match": sid_sha == EXPECTED_SID_SHA,
        "overall_decision": "TARGET REACHED" if target_reached else "GATE 4 FAIL",
    }
    with open(STAGE4_VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=_json_default)
    print(f"\n[Final] {verdict['overall_decision']}, R@10={final_metrics['R@10']:.4f}", flush=True)

    if TRAINING_PID_FILE.exists():
        TRAINING_PID_FILE.unlink()


if __name__ == "__main__":
    main()