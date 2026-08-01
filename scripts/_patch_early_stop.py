#!/usr/bin/env python3
"""
Patch #450/#454/#456 三脚本加 val_R@10 早停 (owner 2026-08-01 派工).

Patch 策略 (per-script 适配, 因脚本有 wrapper 差异):
- 每 5 epoch 跑一次 val_R@10 (复用 GenRecDataset mode='evaluation')
- patience=5 (5 次 eval 无 val_R@10 提升则停)
- best val_R@10 ckpt R12 强制保存 (删旧 + 存新)
- 训练 trace 加 val_r10_trace + early_stop_decision 字段
- 早停触发后立即保存 best ckpt + 跑 Stage 4 eval (跟原脚本 Stage 4 段一致)

实施: 用 Python heredoc 直接 patch 主脚本, 然后重启.
"""
import re
import os
import sys

GENE_REPO = "/home/wlia0047/ar57/wenyu/GeneRec"

# 早停代码片段 (注入到 train loop 之后, log_lines.append 前)
EARLY_STOP_BLOCK = """\n    # ============================================================\n    # Early Stopping (owner 2026-08-01 派工): val_R@10 + patience=5\n    # ============================================================\n    EARLY_STOP_VAL_EVERY = 5\n    EARLY_STOP_PATIENCE = 5\n    EARLY_STOP_BEST_PATH = ADAPTER_CKPT_PATH.parent / (ADAPTER_CKPT_PATH.stem + "_BEST.pt")\n    val_r10_trace = []\n    best_val_r10 = -1.0\n    patience_counter = 0\n    early_stop_triggered = False\n\n    def _compute_val_r10(model_wrapper, val_ds, BATCH_SIZE, MAX_LEN, PAD_TOKEN, DEVICE, GEN_META_FN):\n        model_wrapper.eval()\n        n_val = len(val_ds)\n        n_val_batches = (n_val + BATCH_SIZE - 1) // BATCH_SIZE\n        n_correct_10 = 0\n        n_total = 0\n        with torch.no_grad():\n            for batch_idx in range(n_val_batches):\n                start = batch_idx * BATCH_SIZE\n                end = min(start + BATCH_SIZE, n_val)\n                batch_indices = list(range(start, end))\n                batch_samples = [val_ds[i] for i in batch_indices]\n                history_flat_list = []\n                target_list = []\n                for s in batch_samples:\n                    h = s["history"]\n                    history_flat_list.append([elem for sublist in h for elem in sublist])\n                    tgt = s["target"]\n                    if hasattr(tgt, '__iter__'):\n                        target_list.append([int(x) for x in tgt])\n                    else:\n                        target_list.append([int(tgt)] * 4)\n                history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)\n                target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)\n                attention_mask = (history_tensor != PAD_TOKEN).long()\n                B = history_tensor.shape[0]\n                L_flat = MAX_LEN * 4\n                sid_meta, *meta_args = GEN_META_FN(B, L_flat, history_tensor, DEVICE)\n                kwargs = dict(\n                    history_tensor=history_tensor, attention_mask=attention_mask,\n                    labels=target_tensor, sid_meta=sid_meta,\n                )\n                if len(meta_args) == 1:\n                    kwargs["kappa_meta"] = meta_args[0] if "kappa" in GEN_META_FN.__name__ else meta_args[0]\n                # 调用 model_wrapper (wrapper-specific keys handled by GEN_META_FN name)\n                gen_name = GEN_META_FN.__name__\n                if "scale" in gen_name and len(meta_args) >= 1:\n                    kwargs["scale_meta"] = meta_args[0]\n                if "curv" in gen_name and len(meta_args) >= 1:\n                    kwargs["curvature_meta"] = meta_args[0]\n                output, _, _ = model_wrapper(**kwargs)\n                logits = output.logits if hasattr(output, "logits") else output[0]\n                topk = torch.topk(logits, k=10, dim=-1).indices.cpu().numpy()\n                for i in range(B):\n                    target = target_tensor[i].cpu().numpy().tolist()\n                    if any(t in topk[i].tolist() for t in target):\n                        n_correct_10 += 1\n                    n_total += 1\n        return n_correct_10 / max(n_total, 1)\n\n"""

print("Patch ready")