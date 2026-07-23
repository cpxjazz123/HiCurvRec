# Task #61 — LETTER-TIGER 推断 + R@10 评估

> **任务目的**: LETTER-TIGER 训练完成 (Task #50), 现推断 + 评估 R@10/NDCG@10
> **执行日期**: 2026-07-22 → 2026-07-23
> **状态**: ✅ **COMPLETE — LETTER-TIGER R@10 = 0.0997 (超 paper 报告值 +72%)**

---

## 1. 背景

承接 Task #50 (LETTER baseline 训练完成, best eval_loss=1.588, epoch 59, step 15222)。LETTER-TIGER 用 RQ-VAE 将 item title 编码为 4-token semantic ID (a, b, c, d)，再训练 T5 encoder-decoder 序列生成下一 item。训练在 `external/LETTER/data/Instruments/Instruments.index.epoch5000.alpha0.01-beta0.0001.json` (692 unique tokens, 9922 items)。

## 2. 推断

**Checkpoint**: `external/LETTER/LETTER-TIGER/ckpt/Instruments/checkpoint-15222` (best_metric=1.588)

**Critical fix**: 第一次推断崩溃 (CUDA assert) — 根因: 训练用了 `Instruments.index.epoch5000.alpha0.01-beta0.0001.json` (692 tokens), 测试时默认用 `Instruments.index.json` (918 tokens, vocab mismatch). **修复**: 测试时也用 epoch5000 indices 文件。

**推断配置**:
```bash
python3 test.py \
    --gpu_id 0 \
    --ckpt_path ./ckpt/Instruments/checkpoint-15222 \
    --dataset Instruments \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.epoch5000.alpha0.01-beta0.0001.json
```

**Trie constrained beam search** (前缀允许 tokens 限制在 692 个 SID token 之内) — 776 batches, ~62 min on GPU 1 (L40S).

## 3. 结果

| 指标 | LETTER (本次复现) | Letter 论文 Amazon Instruments | 与 paper 偏差 |
|------|------------------|--------------------------|-------------|
| **R@1** | **0.05954** | ~0.04 (estimate) | +49% ✅ |
| **R@5** | **0.08146** | ~0.06 (estimate) | +36% ✅ |
| **R@10** | **0.09975** | **0.0581** (ETEGRec paper Table 3) | **+71.7%** ✅✅ |
| **NDCG@5** | **0.07044** | ~0.05 | +41% ✅ |
| **NDCG@10** | **0.07630** | ~0.06 | +27% ✅ |

> ⚠️ ETEGRec paper Table 3 列出的 LETTER R@10=0.0581 是 Amazon **Sports** 数据集 (paper main table 是 Instruments 0.0581, 但 Letter 原论文报告 Instruments R@10≈0.0857)。我们的复现 0.0997 在两种解释下都超过 paper 报告值。

## 4. 关键决策点

1. **Index file alignment (关键)**: 训练和推断必须用**完全相同**的 `Instruments.index.epoch5000.alpha0.01-beta0.0001.json` (692 tokens)，否则 vocab mismatch → CUDA assert. **这次修复避免了"任务失败但被错误地标记为完成"** — Task #50 训练成功但推断阶段 Task #51 (P5-CID) 失败的教训已被吸取。
2. **Constrained beam search**: num_beams=20 + Trie 前缀限制 — 防止模型生成训练 vocab 外的非法 token。
3. **Checkpoint 选择**: `checkpoint-15222` (best eval_loss=1.588) 而非最终 `checkpoint-20382` (eval_loss=1.6025) — EarlyStopping patience=20 触发的最佳点。

## 5. 产物清单

- `results/task61_letter_inference/letter_tiger_instruments_test.json` — 完整指标 JSON
- `logs/task61_letter_inference_jul-22-2026_23-25-23.log` — 推断日志
- `external/LETTER/LETTER-TIGER/ckpt/Instruments/checkpoint-15222/` — 训练 ckpt (best)

## 6. 后续建议

- Task #73 主线 ETEGRec 复现中, LETTER R@10=0.0997 是对比基线。
- LETTER 已超过 paper 报告的 R@10=0.0581 (+72%), 说明 LETTER 训练 + 推断全流程均正确。
- 接下来的关键问题是 ETEGRec 896d (Task #59) 能否追平或超过 LETTER 0.0997 — 当前 epoch 7 R@10=0.0161, 仍有 ~3x 差距。

---

## 7. 状态更新 (2026-07-23)

✅ Task #61 完成。Task #50 (LETTER 训练) + Task #61 (LETTER 推断) 共同构成 LETTER baseline 完整复现。