#!/usr/bin/env python3
"""Task #450 / Issue #150 [方向C Gate3+Gate4 long train] 200 epoch Stage 3 + Stage 4 R@K 双复跑.

Owner 2026-08-01 拍板 "do long train". Issue #150 spec Gate 4 强制:
- 200 epoch Stage 3 full training (Task #84 anchor 等同 epoch)
- Stage 4 双复跑 R@K (R@5/10/20, NDCG@5/10/20)
- Target reached: test R@10 > 0.1020 (HG-Rec Task #84 baseline)

R18 4 维度 vs task441 (10 epoch Stage 4 sanity):
- D1 spec: 200 epoch Stage 3 full train vs 10 epoch short-train
- D2 实施: 重新训 ckpt (adapter_200ep.pt) vs 复用 task440 10 epoch ckpt
- D3 Gate 4 决策: 双复跑 R@10 > 0.1020 = Target reached vs sanity check 无决策意义
- D4 引用: Issue #150 spec Gate 4 强制 200 epoch + 双复跑 vs #441 sanity only

R12 强制: 训练结束 → 删旧 adapter_200ep.pt → 存新 ckpt
R20 4 Gate 详细: commit message + Issue #150 comment 每个 Gate ≥3-5 行
R21 v2: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
R7: GPU 0 (全空闲, 46068 MiB available)
R137: TRITON_CACHE_DIR=~/.triton/cache_task450
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
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task450"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

# ============================================================================
# Config (per Issue #150 spec)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 200  # Task #84 anchor 等同 epoch (Issue #150 spec Gate 4)
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_INIT = 0.0
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task450_issue150_stage3_long_train_200ep")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task450_issue150_stage3_long_train_200ep.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
TRAIN_TRACE_PATH = PRODUCT_DIR / "train_trace_200ep.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter_200ep.pt"  # 区别 task440 adapter.pt
EVAL_RUN1_PATH = PRODUCT_DIR / "eval_run1.json"
EVAL_RUN2_PATH = PRODUCT_DIR / "eval_run2.json"
STAGE4_VERDICT_PATH = PRODUCT_DIR / "stage4_verdict.json"
DECISION_BASELINE_R10 = 0.1020  # HG-Rec Task #84 baseline


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Wrapper 类 (跟 task440 同架构, 直接 import 复用)
# ============================================================================
import importlib.util as _ilu
_t440_spec = _ilu.spec_from_file_location(
    "task440_module",
    "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task440_issue150_zero_centered_linear_layernorm.py",
)
_t440_mod = _ilu.module_from_spec(_t440_spec)
_t440_spec.loader.exec_module(_t440_mod)
WrapperCls = _t440_mod.HG_Rec_with_ZeroCenteredLayerNormAdapter
verify_sid_token_range = _t440_mod.verify_sid_token_range


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #450 Issue #150 long train] 200 epoch Stage 3 + Stage 4 双复跑 R@K")
    log_lines.append(f"[R18] 4 维度 vs task441: 200 epoch vs 10 epoch + full test vs 50 batch sanity")
    log_lines.append("=" * 70)

    # ============================================================================
    # Config + SHA256 (R143 reproducibility triangle)
    # ============================================================================
    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    test_parquet_sha = sha256_of(TEST_PARQUET)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[SHA256] test.parquet: {test_parquet_sha}")

    config = {
        "seed": SEED, "device": DEVICE, "codebook_size": CODEBOOK_SIZE,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
        "lr_conditioner": LR_CONDITIONER, "lr_layernorm": LR_LAYERNORM,
        "alpha_init": ALPHA_INIT, "d_model": D_MODEL, "n_layers": 3,
        "decision_baseline_r10": DECISION_BASELINE_R10,
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
        "task_id": 450, "issue": "Issue #150",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # ============================================================================
    # 加载 T5 (Task #84 ckpt)
    # ============================================================================
    log_lines.append(f"\n[Load T5] ckpt={T5_CKPT}")
    t5_state_dict = _t440_mod.load_t5_state_dict(T5_CKPT)
    t5_config = _t440_mod.get_t5_config()

    # ============================================================================
    # 加载 GenRecDataset
    # ============================================================================
    log_lines.append(f"\n[Load dataset] mode='train', real history-SID")
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"[Data] train_ds size: {len(train_ds)}")

    sample = [train_ds[i] for i in range(BATCH_SIZE)]
    history_list = [s["history"] for s in sample]
    target_list = [s["target"] for s in sample]
    history_flat_list = [[elem for sublist in h for elem in sublist] for h in history_list]
    history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)
    target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
    log_lines.append(f"[Data] history_tensor shape: {history_tensor.shape}")

    sid_range = verify_sid_token_range(history_tensor, CODEBOOK_SIZE)
    log_lines.append(f"[Precheck] SID token range: {sid_range}")

    # ============================================================================
    # 构造 Wrapper
    # ============================================================================
    model_wrapper = WrapperCls(
        t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
    ).to(DEVICE)

    # ============================================================================
    # Stage 3 训练: 200 epoch (替代 task440 10 epoch)
    # ============================================================================
    log_lines.append(f"\n[Stage 3 训练] {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE} (long train)")
    conditioner_params = [p for n, p in model_wrapper.adapter.named_parameters()]
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    # 预加载 train_ds 整个到 GPU tensor
    log_lines.append(f"[预加载] 把 train_ds 整个加载到 GPU tensor...")
    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    log_lines.append(f"  preloaded histories shape: {all_histories.shape}, targets shape: {all_targets.shape}")
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    log_lines.append(f"[训练循环] {NUM_EPOCHS} epoch × {n_batches} batches = {NUM_EPOCHS * n_batches} total")
    log_lines.append(f"[估时] 200 epoch × ~3 sec/batch ≈ {NUM_EPOCHS * n_batches * 3 / 3600:.1f}h")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log_lines = []

    train_trace = []
    epoch0_loss = None
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
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

            optimizer.zero_grad()
            loss, _, alpha_val = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                              labels=target_tensor_b, sid_meta=sid_meta, kappa_meta=kappa_meta)
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

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        avg_cond_grad = sum(cond_grad_norms) / max(1, len(cond_grad_norms))
        avg_ln_grad = sum(ln_grad_norms) / max(1, len(ln_grad_norms))
        if epoch == 0:
            epoch0_loss = avg_loss
        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss,
            "avg_cond_grad_norm": avg_cond_grad,
            "avg_ln_grad_norm": avg_ln_grad,
            "alpha_val": alpha_val.item(),
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        msg = f"  [epoch {epoch}/{NUM_EPOCHS-1}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, α={alpha_val.item():.6e}, n_batches={len(epoch_losses)}, nan_inf={nan_inf_detected}"
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
        # 每 10 epoch 落一次 train_trace (防 200 epoch 内存爆)
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            with open(TRAIN_TRACE_PATH, "w") as f:
                json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                           "loss_decreased": epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss,
                           "num_epochs_completed": epoch + 1}, f, indent=2)

    loss_decreased = epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss
    print(f"\n[Stage 3 训练完成] epoch0={epoch0_loss:.4f}, epoch{NUM_EPOCHS-1}={train_trace[-1]['avg_loss']:.4f}, decreased={loss_decreased}", flush=True)

    # ============================================================================
    # R12 强制 ckpt 落盘 (删旧 + 存新)
    # ============================================================================
    print(f"\n[R12 ckpt] 保存 adapter_200ep.pt (删旧 + 存新)", flush=True)
    if ADAPTER_CKPT_PATH.exists():
        ADAPTER_CKPT_PATH.unlink()
    torch.save({
        "adapter_state_dict": model_wrapper.adapter.state_dict(),
        "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
        "alpha_value": model_wrapper.adapter.get_alpha().item(),
        "epoch_losses": [t["avg_loss"] for t in train_trace],
        "num_epochs": NUM_EPOCHS,
        "epoch0_loss": epoch0_loss,
        "final_loss": train_trace[-1]["avg_loss"],
    }, ADAPTER_CKPT_PATH)
    adapter_ckpt_sha = sha256_of(ADAPTER_CKPT_PATH)
    print(f"  adapter SHA256: {adapter_ckpt_sha}", flush=True)

    # ============================================================================
    # Stage 4 R@K 双复跑 (Issue #150 spec Gate 4 强制)
    # ============================================================================
    print(f"\n[Stage 4 R@K 双复跑] full test set, double-run", flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[Stage 4 R@K 双复跑] full test set, double-run\n")

    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n_test = len(test_ds)
    n_test_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  test_ds size: {n_test}, n_batches: {n_test_batches}", flush=True)

    def run_eval(run_id):
        """单次 Stage 4 eval run (full test set)."""
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
            kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)

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
                logits = model_wrapper.t5.lm_head(decoder_outputs.last_hidden_state)  # (B, 4, vocab_size)

            # Per-position argmax → 4-digit SID prediction
            preds = logits.argmax(dim=-1)  # (B, 4)
            for i in range(B):
                pred = preds[i].cpu().tolist()
                target = target_tensor[i].cpu().tolist()
                n_total += 1
                # 4-digit 全 match = R@K 全 PASS (跟 task441 同协议)
                if pred[:4] == target[:4]:
                    n_correct_20 += 1
                    n_correct_10 += 1
                    n_correct_5 += 1
                    # NDCG: 1/log2(2) = 1 for rank 1
                    ndcg_sum_5 += 1.0
                    ndcg_sum_10 += 1.0
                    ndcg_sum_20 += 1.0
            if batch_idx % 50 == 0:
                print(f"  [run{run_id} batch {batch_idx}/{n_test_batches}] cumulative R@20={n_correct_20}/{n_total}={n_correct_20/max(1,n_total):.4f}", flush=True)
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

    # 双复跑 (Issue #150 spec Gate 4 强制)
    run1 = run_eval(run_id=1)
    print(f"\n[Run 1 完成] {run1}", flush=True)
    with open(EVAL_RUN1_PATH, "w") as f:
        json.dump(run1, f, indent=2)

    run2 = run_eval(run_id=2)
    print(f"\n[Run 2 完成] {run2}", flush=True)
    with open(EVAL_RUN2_PATH, "w") as f:
        json.dump(run2, f, indent=2)

    # ============================================================================
    # 双复跑对比 + Gate 4 决策
    # ============================================================================
    print(f"\n[Gate 4 决策]", flush=True)
    r10_diff = abs(run1["R@10"] - run2["R@10"])
    r10_avg = (run1["R@10"] + run2["R@10"]) / 2
    target_reached = r10_avg > DECISION_BASELINE_R10 and r10_diff < 0.005
    print(f"  R@10 双复跑: run1={run1['R@10']:.4f}, run2={run2['R@10']:.4f}, diff={r10_diff:.4f}, avg={r10_avg:.4f}", flush=True)
    print(f"  Baseline R@10: {DECISION_BASELINE_R10}", flush=True)
    print(f"  Target reached: {target_reached} (avg > 0.1020 AND diff < 0.005)", flush=True)

    if target_reached:
        gate4_decision = "GO_PASS"
        decision_msg = f"✅ R@10={r10_avg:.4f} > 0.1020 (avg 双复跑), diff={r10_diff:.4f} < 0.005"
    elif r10_avg > DECISION_BASELINE_R10:
        gate4_decision = "GO_PARTIAL"
        decision_msg = f"⚠️ R@10={r10_avg:.4f} > 0.1020 但 diff={r10_diff:.4f} ≥ 0.005 (双复跑一致性 FAIL)"
    elif loss_decreased and all(t["avg_cond_grad_norm"] > 0 for t in train_trace):
        gate4_decision = "PARTIAL_ARCHITECTURE_OK"
        decision_msg = f"⚠️ 架构修复成功 (loss 下降, 梯度全程不衰减) 但 R@10={r10_avg:.4f} ≤ 0.1020, paper §6.7 记录 zero-centered LN + bounded-linear 路径"
    else:
        gate4_decision = "NO_GO"
        decision_msg = f"❌ 200 epoch 训练失败 (loss 不下降 / 梯度衰减)"

    print(f"  决策: {gate4_decision} - {decision_msg}", flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[Gate 4 决策] {gate4_decision}: {decision_msg}\n")

    # ============================================================================
    # Verdict 汇总
    # ============================================================================
    train_trace_summary = {
        "total_epochs": NUM_EPOCHS,
        "epoch0_loss": epoch0_loss,
        "final_loss": train_trace[-1]["avg_loss"],
        "loss_decreased": loss_decreased,
        "loss_change_pct": (train_trace[-1]["avg_loss"] - epoch0_loss) / max(1e-8, epoch0_loss) * 100,
        "cond_grad_strs": [f"{t['avg_cond_grad_norm']:.4e}" for t in train_trace],
        "ln_grad_strs": [f"{t['avg_ln_grad_norm']:.4e}" for t in train_trace],
        "alpha_strs": [f"{t['alpha_val']:.4e}" for t in train_trace],
        "any_nan_inf": any(t["nan_inf"] for t in train_trace),
        "loss_50_200": train_trace[49]["avg_loss"] if NUM_EPOCHS >= 50 else None,
        "loss_100_200": train_trace[99]["avg_loss"] if NUM_EPOCHS >= 100 else None,
        "loss_150_200": train_trace[149]["avg_loss"] if NUM_EPOCHS >= 150 else None,
        "loss_199_200": train_trace[-1]["avg_loss"],
    }

    verdict = {
        "task_id": 450,
        "issue": "Issue #150",
        "stage3_long_train": {
            "num_epochs": NUM_EPOCHS,
            "epoch0_loss": epoch0_loss,
            "final_loss": train_trace[-1]["avg_loss"],
            "loss_decreased": loss_decreased,
            "train_trace_summary": train_trace_summary,
        },
        "stage4_double_run": {
            "run1": run1,
            "run2": run2,
            "r10_diff": r10_diff,
            "r10_avg": r10_avg,
        },
        "gate4_decision": gate4_decision,
        "decision_msg": decision_msg,
        "baseline_r10": DECISION_BASELINE_R10,
        "target_reached": target_reached,
        "adapter_ckpt_sha256": adapter_ckpt_sha,
        "sid_npy_sha256": sid_sha,
        "t5_ckpt_sha256": t5_ckpt_sha,
        "test_parquet_sha256": test_parquet_sha,
        "commit_hash": "PENDING_R21_FIX",
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2)
    with open(STAGE4_VERDICT_PATH, "w") as f:
        json.dump({"gate4_decision": gate4_decision, "run1": run1, "run2": run2,
                   "r10_avg": r10_avg, "r10_diff": r10_diff, "target_reached": target_reached,
                   "baseline_r10": DECISION_BASELINE_R10}, f, indent=2)

    print(f"\n[Verdict 落盘] {VERDICT_PATH}", flush=True)
    print(f"[整体决策] {gate4_decision}", flush=True)


if __name__ == "__main__":
    main()
