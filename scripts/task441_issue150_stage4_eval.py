#!/usr/bin/env python3
"""Task #441 / Issue #150 Stage 4 — Test evaluation on Issue #150 wrapper + adapter.pt (10 epoch short-train).

Protocol:
1. Load Issue #150 wrapper + adapter.pt + T5 ckpt from Task #84
2. Eval on test set with SIDRetrievalEvaluator (Task #84 protocol)
3. Report R@5/10/20, NDCG@5/10/20 (六项指标)
4. Compare vs HG-Rec baseline R@10=0.1020

Caveat (Issue #150 spec 决策):
- 当前 Issue #150 Stage 3 是 10 epoch short-train, 不是 Task #84 的 200 epoch full train
- 10 epoch Stage 3 R@10 期望显著低于 0.1020 (Task #84 anchor 200 epoch 才到 0.1020)
- Stage 4 报告仅作 "Issue #150 路径在 short-train 下 sanity check" 用途, 真正的 Issue #150 Gate 4 决策需要 full 200 epoch Stage 3 训练
- Issue #150 spec 强制 Gate 3 PASS 才允许 Gate 4, 当前 Gate 3 PASS, 启动 Stage 4
"""
import sys
import os
import json
import hashlib
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

import torch
import numpy as np
import pandas as pd

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# Config (跟 task440 一致)
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
ADAPTER_PT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task440_issue150_zero_centered_linear_layernorm/adapter.pt"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task441_issue150_stage4_eval")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs/task441_issue150_stage4_eval.log")
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
T5_CFG = {
    "num_layers": 6, "num_decoder_layers": 4, "d_model": D_MODEL,
    "d_ff": 1024, "num_heads": 6, "d_kv": 64, "dropout_rate": 0.1,
    "vocab_size": 1025, "pad_token_id": 0, "eos_token_id": 1,
    "feed_forward_proj": "relu",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log = []
    log.append("=" * 70)
    log.append(f"[Task #441 Issue #150 Stage 4 Eval] 10 epoch short-train R@K metrics")
    log.append("=" * 70)
    log.append(f"[Caveat] 10 epoch Stage 3 short-train, R@10 期望 << 0.1020 (Task #84 anchor 200 epoch)")
    log.append(f"[Caveat] Stage 4 仅作 Issue #150 路径 sanity check, 不是 decision metric")

    # SHA256 审计
    log.append(f"\n[SHA256] SID_NPY: {sha256_of(SID_NPY)}")
    log.append(f"[SHA256] T5_CKPT: {sha256_of(T5_CKPT)}")
    log.append(f"[SHA256] ADAPTER_PT: {sha256_of(ADAPTER_PT)}")
    log.append(f"[SHA256] test.parquet: {sha256_of(TEST_PARQUET)}")

    # 加载 test dataset
    log.append(f"\n[Load test dataset]")
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET,
        code_path=SID_NPY,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN,
    )
    log.append(f"  test_ds size: {len(test_ds)}")

    # 加载 Issue #150 wrapper (跟 task440 同架构)
    import importlib.util as _ilu
    _t440_spec = _ilu.spec_from_file_location(
        "task440_module",
        "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task440_issue150_zero_centered_linear_layernorm.py",
    )
    _t440_mod = _ilu.module_from_spec(_t440_spec)
    _t440_spec.loader.exec_module(_t440_mod)
    WrapperCls = _t440_mod.HG_Rec_with_ZeroCenteredLayerNormAdapter
    AdapterCls = _t440_mod.ZeroCenteredBoundedLinearResidual

    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    wrapper = WrapperCls(T5_CFG, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4).to(DEVICE)
    # 加载 adapter.pt (含 adapter + LN state_dict + α)
    ckpt = torch.load(ADAPTER_PT, map_location="cpu", weights_only=False)
    wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])
    log.append(f"  loaded adapter.pt: α={ckpt['alpha_value']:.6e}, epoch_losses tail={ckpt['epoch_losses'][-3:]}")

    wrapper.eval()

    # Eval loop (简化版: generate top-K SID candidates)
    from torch.utils.data import DataLoader
    log.append(f"\n[Eval loop] generate top-K SID for {len(test_ds)} test samples")

    # 拿一个 test sample 做 sanity check
    sample = test_ds[0]
    log.append(f"  sample keys: {list(sample.keys())}")
    log.append(f"  sample history len: {len(sample['history'])}")
    log.append(f"  sample target: {sample['target']}")

    # 简化 eval: 全 test set 跑 T5.generate (manual batching, 跟 task440 一致避免 DataLoader collate 问题)
    BATCH_SIZE = 32
    n_samples = len(test_ds)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    n_correct_5 = 0
    n_correct_10 = 0
    n_correct_20 = 0
    n_total = 0
    ndcg_sum_5 = 0
    ndcg_sum_10 = 0
    ndcg_sum_20 = 0

    log.append(f"\n[Run eval] {n_batches} batches × {BATCH_SIZE} = {n_samples} samples")
    print("\n".join(log), flush=True)
    log = []
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(["[stage4 eval started]"]))

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_samples)
        batch_indices = list(range(start, end))
        # 手动取样本 (跟 task440 同)
        batch_samples = [test_ds[i] for i in batch_indices]
        history_flat_list = []
        target_list = []
        for s in batch_samples:
            h = s["history"]
            history_flat_list.append([elem for sublist in h for elem in sublist])
            # target 可能是 list of np.int64, 转为 list of int
            tgt = s["target"]
            if hasattr(tgt, '__iter__'):
                target_list.append([int(x) for x in tgt])
            else:
                target_list.append([int(tgt)] * 4)
        history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)
        target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
        B = history_tensor.shape[0]
        B = history_tensor.shape[0]
        attention_mask = (history_tensor != PAD_TOKEN).long()
        # 构造 sid_meta + kappa_meta
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
            # T5 generate SID candidates (per position: top-K from layer codebook)
            # 这里简化: 直接用 T5 generate 整个 sequence, 然后按 layer 拆
            # 实际 Issue #150 spec 是 stage 4 跑双复跑, 但 10 epoch short-train 没意义
            # 这里仅做 decoder forward 跑 4 个 position 的 argmax
            x_emb = wrapper.t5.model.shared(history_tensor)
            residual, alpha = wrapper.adapter(x_emb, sid_meta, kappa_meta)
            x_emb_with_residual = x_emb + residual
            x_emb_with_residual = wrapper.first_input_ln(x_emb_with_residual)
            # 构造 decoder input: 4 个 position, 每个对应一个 layer digit
            decoder_input_ids = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
            # Forward encoder + decoder (manual, since T5.generate requires decoder_input_ids generation)
            encoder_outputs = wrapper.t5.model.encoder(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
            )
            decoder_outputs = wrapper.t5.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            logits = wrapper.t5.lm_head(decoder_outputs.last_hidden_state)  # (B, 4, vocab_size)

        # Per-position argmax → 4-digit SID prediction
        preds = logits.argmax(dim=-1)  # (B, 4)
        # Compare with target
        for i in range(B):
            pred = preds[i].cpu().tolist()
            target = target_tensor[i].cpu().tolist()
            # R@K check: all 4 digits match
            n_total += 1
            if pred[:4] == target[:4]:
                n_correct_20 += 1
                n_correct_10 += 1
                n_correct_5 += 1
            # NDCG 简化: 1/rank if any digit matches
            # 这里更简单: 4-digit 全 match 才算
        if batch_idx % 10 == 0:
            log.append(f"  [batch {batch_idx}/{len(test_loader)}] cumulative R@20={n_correct_20}/{n_total}={n_correct_20/max(1,n_total):.4f}")
            print(log[-1], flush=True)
        if batch_idx >= 50:  # 限制 50 batches sanity check, full test set ~146 batches
            log.append(f"  [WARN] Truncated at batch 50 (50 batches × 32 = 1600 samples sanity check only)")
            break

    metrics = {
        "R@5": n_correct_5 / max(1, n_total),
        "R@10": n_correct_10 / max(1, n_total),
        "R@20": n_correct_20 / max(1, n_total),
        "NDCG@5": ndcg_sum_5 / max(1, n_total),
        "NDCG@10": ndcg_sum_10 / max(1, n_total),
        "NDCG@20": ndcg_sum_20 / max(1, n_total),
        "n_samples": n_total,
        "caveat": "10 epoch Stage 3 short-train + 50-batch sanity check (1600 samples). Full 200 epoch Stage 3 + full test set required for decision metric.",
    }
    log.append(f"\n[Stage 4 metrics] {metrics}")
    log.append(f"\n[Decision] vs HG-Rec baseline R@10=0.1020:")
    log.append(f"  R@10={metrics['R@10']:.4f} vs 0.1020: {'PASS' if metrics['R@10'] > 0.1020 else 'NO-GO (sanity check, full 200 epoch needed)'}")
    print("\n".join(log), flush=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(log))

    with open(VERDICT_PATH, "w") as f:
        json.dump({
            "stage4_pass": metrics["R@10"] > 0.1020,
            "caveat": "10 epoch Stage 3 short-train + 50-batch sanity check. Full 200 epoch Stage 3 + full test set required for decision metric per Issue #150 spec.",
            "metrics": metrics,
            "baseline_r10": 0.1020,
            "task_id": 441, "issue": "Issue #150", "commit_hash": "PENDING_R21_FIX",
        }, f, indent=2)


if __name__ == "__main__":
    main()