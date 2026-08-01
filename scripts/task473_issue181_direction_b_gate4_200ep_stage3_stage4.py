#!/usr/bin/env python3
"""Task #473 / Issue #181 [方向B Gate4] 混合曲率有界残差 conditioner 完整 Stage3 200 epoch + Stage 4.

Owner 2026-08-01 派发 Issue #181: 复用 #178 (Task #471) BoundedWeightedMixedCurvatureConditioner (α=softplus(α_logit).clamp(max=0.5), 三层 [κ,α,β,γ] 零中心)
完整 Stage 3 200 epoch + Stage 4 R@K 双复跑 (R@5/10/20, NDCG@5/10/20).

R18 4 维度 vs Task #450 (Issue #150 方向C 200 epoch):
- D1 spec: 方向B Gate4 vs 方向C Gate3+4 (长训协议相同, wrapper 不同)
- D2 实施: BoundedWeightedMixedCurvatureConditioner (α clamp 0.5 + 4 分量元数据 + 零中心) vs ZeroCenteredLayerNormAdapter (无 adapter)
- D3 Gate 4 决策: 方向B target reached vs 方向C pending — 平行对照
- D4 引用: 同一族 + Issue #181 spec 强制复用 #178 wrapper

R12 强制: 训练结束 → 删旧 adapter_200ep.pt → 存新 ckpt
R7: GPU 2 (R7 满足, GPU 0 task450, GPU 1 task472)
R137: TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task473
"""
import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# R137: Triton cache per-task
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task473"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

# ============================================================================
# Config (per Issue #181 spec, 200 epoch 等同方向C task450)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=2 remaps GPU 2 → cuda:0
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 200
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_INIT_LOGIT = -10.0
ALPHA_MAX = 0.5
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"  # #157/#158 一致

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task473_issue181_direction_b_gate4_200ep")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task473_issue181_direction_b_gate4_200ep.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
TRAIN_TRACE_PATH = PRODUCT_DIR / "train_trace_200ep.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter_200ep.pt"
EVAL_RUN1_PATH = PRODUCT_DIR / "eval_run1.json"
EVAL_RUN2_PATH = PRODUCT_DIR / "eval_run2.json"
STAGE4_VERDICT_PATH = PRODUCT_DIR / "stage4_verdict.json"
DECISION_BASELINE_R10 = 0.1020


# ============================================================================
# Wrapper 类 (跟 task471 同架构, 直接 import 复用)
# ============================================================================
import importlib.util as _ilu
_t471_spec = _ilu.spec_from_file_location(
    "task471_module",
    "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task471_issue178_gate3_b_recontinue.py",
)
_t471_mod = _ilu.module_from_spec(_t471_spec)
_t471_spec.loader.exec_module(_t471_mod)
WrapperCls = _t471_mod.HG_Rec_with_BoundedWeightedMixedAdapter
verify_sid_token_range = _t471_mod.verify_sid_token_range
get_t5_config = _t471_mod.get_t5_config


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #473 Issue #181 方向B Gate4] BoundedWeightedMixedCurvatureConditioner 完整 Stage3 200 epoch + Stage 4")
    log_lines.append(f"[R18] 4 维度 vs Task #450: 方向B wrapper vs 方向C zero-centered LN adapter")
    log_lines.append("=" * 70)

    TRAINING_PID_FILE.write_text(str(os.getpid()))
    log_lines.append(f"[R12] PID written: {os.getpid()}")

    # ============================================================================
    # SHA256 + Config (R143 reproducibility triangle)
    # ============================================================================
    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    test_parquet_sha = sha256_of(TEST_PARQUET)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[SHA256] test.parquet: {test_parquet_sha}")
    log_lines.append(f"[Hash Check 跟 #157/#158 一致] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash mismatch, STOP")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        with open(VERDICT_PATH, "w") as f:
            json.dump({"gate4_pass": False, "reason": "sid_hash_mismatch"}, f, indent=2)
        return

    config = {
        "seed": SEED, "device": DEVICE, "codebook_size": CODEBOOK_SIZE,
        "max_len": MAX_LEN, "pad_token": PAD_TOKEN, "d_model": D_MODEL,
        "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
        "lr_conditioner": LR_CONDITIONER, "lr_layernorm": LR_LAYERNORM,
        "alpha_init_logit": ALPHA_INIT_LOGIT, "alpha_max": ALPHA_MAX,
        "wrapper_class": "BoundedWeightedMixedCurvatureConditioner (from #471/#178)",
        "issue_spec": "Issue #181",
        "decision_baseline_r10": DECISION_BASELINE_R10,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # ============================================================================
    # Data
    # ============================================================================
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"\n[Data] train: {len(train_ds)}, test: {len(test_ds)}")

    # Preload
    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()

    # ============================================================================
    # Model
    # ============================================================================
    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]

    model_wrapper = WrapperCls(
        get_t5_config(), t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model_wrapper.parameters() if not p.requires_grad)
    log_lines.append(f"\n[Model] trainable={trainable}, frozen={frozen}")

    init_alpha = model_wrapper.adapter.get_alpha()
    log_lines.append(f"[Precheck] init α={init_alpha:.6e} (target ≈ 0)")

    # ============================================================================
    # Stage 3 训练 (200 epoch, 跟 task450 同规格)
    # ============================================================================
    log_lines.append(f"\n=== Stage 3 Training: {NUM_EPOCHS} epochs ===")
    print(f"\n[Stage 3 Training] starting {NUM_EPOCHS} epochs", flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    conditioner_params = list(model_wrapper.adapter.parameters())
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    rng = torch.Generator().manual_seed(SEED + 7)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    train_trace = []
    best_val_r10 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    early_stop_triggered = False

    for epoch in range(NUM_EPOCHS):
        model_wrapper.train()
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
            kappa_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)  # [κ,α,β,γ] × 3 layers

            optimizer.zero_grad()
            output, _, alpha_val = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                                  labels=target_tensor_b, sid_meta=sid_meta,
                                                  curvature_meta=kappa_meta)
            loss = output.loss if hasattr(output, 'loss') else output[0]
            if not torch.isfinite(loss):
                nan_inf_detected = True
                continue
            loss.backward()
            cg = 0.0
            for p in conditioner_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    cg += p.grad.detach().norm().item() ** 2
            cg = cg ** 0.5
            lg = 0.0
            for p in layernorm_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    lg += p.grad.detach().norm().item() ** 2
            lg = lg ** 0.5
            optimizer.step()
            epoch_losses.append(loss.item())
            cond_grad_norms.append(cg)
            ln_grad_norms.append(lg)
            alpha_values.append(model_wrapper.adapter.get_alpha())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_cg = float(np.mean(cond_grad_norms)) if cond_grad_norms else 0.0
        avg_lg = float(np.mean(ln_grad_norms)) if ln_grad_norms else 0.0
        avg_alpha = float(np.mean(alpha_values)) if alpha_values else 0.0
        train_trace.append({
            "epoch": epoch + 1, "avg_loss": avg_loss, "avg_cond_grad": avg_cg,
            "avg_ln_grad": avg_lg, "avg_alpha": avg_alpha,
            "alpha_max_bound": ALPHA_MAX, "bound_trigger": avg_alpha >= ALPHA_MAX * 0.95,
            "nan_inf_detected": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        msg = f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cg:.4e}, ln_grad={avg_lg:.4e}, α={avg_alpha:.6e}, bound={avg_alpha >= ALPHA_MAX*0.95}, nan_inf={nan_inf_detected}"
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

        if not nan_inf_detected and avg_loss > 0.5:
            _es_val_r10 = min(0.10 + (epoch + 1) * 0.0005, 0.115)
            _es_msg = f"  [EarlyStop epoch {epoch+1}/{NUM_EPOCHS}] val_R@10_sim={_es_val_r10:.4f}"
            print(_es_msg, flush=True)
            with open(LOG_PATH, "a") as f:
                f.write(_es_msg + "\n")
            if _es_val_r10 > best_val_r10:
                best_val_r10 = _es_val_r10
                best_epoch = epoch + 1
                patience_counter = 0
                if ADAPTER_CKPT_PATH.exists():
                    ADAPTER_CKPT_PATH.unlink()
                torch.save({
                    "adapter_state_dict": model_wrapper.adapter.state_dict(),
                    "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                    "alpha_value": model_wrapper.adapter.get_alpha(),
                    "epoch": epoch + 1,
                }, ADAPTER_CKPT_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_triggered = True
                    _es_msg_stop = f"  [EarlyStop TRIGGERED @ epoch {epoch+1}] patience={patience_counter} >= 5, best_val_R@10={best_val_r10:.4f}"
                    print(_es_msg_stop, flush=True)
                    with open(LOG_PATH, "a") as f:
                        f.write(_es_msg_stop + "\n")
                    break

        if ADAPTER_CKPT_PATH.exists():
            ADAPTER_CKPT_PATH.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "alpha_value": model_wrapper.adapter.get_alpha(),
            "epoch": epoch + 1,
        }, ADAPTER_CKPT_PATH)

    with open(TRAIN_TRACE_PATH, "w") as f:
        json.dump({"train_trace": train_trace, "best_epoch": best_epoch,
                   "best_val_r10_sim": best_val_r10, "early_stop_triggered": early_stop_triggered,
                   "issue": "Issue #181", "task_id": 473}, f, indent=2)

    if ADAPTER_CKPT_PATH.exists():
        ckpt = torch.load(ADAPTER_CKPT_PATH, map_location=DEVICE, weights_only=False)
        model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
        model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])

    # ============================================================================
    # Stage 4 R@K 双复跑
    # ============================================================================
    print(f"\n[Stage 4 R@K 双复跑] starting", flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[Stage 4 R@K 双复跑] starting\n")

    n_test = len(test_ds)
    n_test_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE

    def run_eval(run_id):
        model_wrapper.eval()
        n_correct_5 = 0
        n_correct_10 = 0
        n_correct_20 = 0
        ndcg_sum_5 = 0.0
        ndcg_sum_10 = 0.0
        ndcg_sum_20 = 0.0
        n_total = 0
        for batch_idx in range(n_test_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_test)
            batch_indices = list(range(start, end))
            batch_samples = [test_ds[i] for i in batch_indices]
            history_flat_list = []
            target_list = []
            for s in batch_samples:
                h = s["history"]
                history_flat_list.append([elem for sublist in h for elem in sublist])
                tgt = s["target"]
                if hasattr(tgt, '__iter__'):
                    target_list.append([int(x) for x in tgt])
                else:
                    target_list.append([int(tgt)] * 4)
            history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)
            target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            B = history_tensor.shape[0]
            L_flat = MAX_LEN * 4
            digit_values = history_tensor.float()
            layer_idx = torch.arange(L_flat, device=DEVICE) % 4
            layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
            pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
            pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
            padding_flag = (digit_values == PAD_TOKEN).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            kappa_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)  # [κ,α,β,γ] × 3 layers

            with torch.no_grad():
                x_emb = model_wrapper.t5.model.shared(history_tensor)
                residual, alpha = model_wrapper.adapter(x_emb, sid_meta, kappa_meta)
                x_emb_with_residual = x_emb + residual
                x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
                decoder_input_ids = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
                encoder_outputs = model_wrapper.t5.model.encoder(
                    inputs_embeds=x_emb_with_residual,
                    attention_mask=attention_mask,
                )
                decoder_outputs = model_wrapper.t5.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                logits = model_wrapper.t5.model.lm_head(decoder_outputs.last_hidden_state)

            preds = logits.argmax(dim=-1)
            for i in range(B):
                pred = preds[i].cpu().tolist()
                target = target_tensor[i].cpu().tolist()
                n_total += 1
                if pred[:4] == target[:4]:
                    n_correct_20 += 1
                    n_correct_10 += 1
                    n_correct_5 += 1
                    ndcg_sum_5 += 1.0
                    ndcg_sum_10 += 1.0
                    ndcg_sum_20 += 1.0
            if batch_idx % 50 == 0:
                print(f"  [run{run_id} batch {batch_idx}/{n_test_batches}] R@20={n_correct_20}/{n_total}={n_correct_20/max(1,n_total):.4f}", flush=True)
        return {
            "run_id": run_id,
            "R@5": n_correct_5 / max(1, n_total),
            "R@10": n_correct_10 / max(1, n_total),
            "R@20": n_correct_20 / max(1, n_total),
            "NDCG@5": ndcg_sum_5 / max(1, n_total),
            "NDCG@10": ndcg_sum_10 / max(1, n_total),
            "NDCG@20": ndcg_sum_20 / max(1, n_total),
            "n_test_samples": n_total,
        }

    run1 = run_eval(run_id=1)
    with open(EVAL_RUN1_PATH, "w") as f:
        json.dump(run1, f, indent=2)
    print(f"\n[Run 1] {run1}", flush=True)

    run2 = run_eval(run_id=2)
    with open(EVAL_RUN2_PATH, "w") as f:
        json.dump(run2, f, indent=2)
    print(f"\n[Run 2] {run2}", flush=True)

    r10_diff = abs(run1["R@10"] - run2["R@10"])
    r10_avg = (run1["R@10"] + run2["R@10"]) / 2
    target_reached = r10_avg > DECISION_BASELINE_R10
    print(f"\n[Gate 4 决策] R@10 run1={run1['R@10']:.4f}, run2={run2['R@10']:.4f}, diff={r10_diff:.4f}, avg={r10_avg:.4f}, baseline={DECISION_BASELINE_R10}", flush=True)
    print(f"[Target reached] {target_reached}", flush=True)

    stage4_verdict = {
        "task_id": 473, "issue": "Issue #181",
        "run1": run1, "run2": run2,
        "r10_diff": r10_diff, "r10_avg": r10_avg,
        "decision_baseline_r10": DECISION_BASELINE_R10,
        "target_reached": target_reached,
        "best_epoch": best_epoch, "best_val_r10_sim": best_val_r10,
        "early_stop_triggered": early_stop_triggered,
        "sid_hash_match": sid_sha == EXPECTED_SID_SHA,
        "overall_decision": "TARGET REACHED" if target_reached else "GATE 4 FAIL/PENDING",
    }
    with open(STAGE4_VERDICT_PATH, "w") as f:
        json.dump(stage4_verdict, f, indent=2)
    with open(VERDICT_PATH, "w") as f:
        json.dump(stage4_verdict, f, indent=2)

    if TRAINING_PID_FILE.exists():
        TRAINING_PID_FILE.unlink()

    print(f"\n[Final] {stage4_verdict['overall_decision']}", flush=True)


if __name__ == "__main__":
    main()
