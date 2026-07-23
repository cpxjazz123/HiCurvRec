# Task #78 Result — TIGER T5 Training on Musical_Instruments (paper Table 2 #9)

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成 — Stage 3 TIGER T5 训练完成, best checkpoint at step 51500
> **下游任务**: paper Table 2 baseline #9 (TIGER) 在 Musical_Instruments 数据集上

---

## 1. 任务目标

在 Musical_Instruments 数据集上训练 LETTER-TIGER (t5-small backbone), 作为 paper Table 2
baseline #9 (TIGER) 的复现。Stage 4 inference 在 Task #84 完成 (hit@10=0.0591, NDCG@10=0.0450)。

## 2. 训练配置

| 项 | 值 |
|----|----|
| Backbone | t5-small (LETTER-TIGER) |
| 数据集 | Musical_Instruments (5-core, 24588 items) |
| SID 长度 | 3 (num_hierarchies=3) |
| SID 来源 | `products/task78_phonism_rqvae_768d/rqvae_ckpt/...` (phonism RQ-VAE) |
| 最大步数 | ~53050 steps |
| Best checkpoint | step 51500 |
| 训练时长 | ~1h 41min |

## 3. 关键产物

- Best ckpt: `external/LETTER/LETTER-TIGER/ckpt/Instruments_tiger/checkpoint-51500/`
- 模型权重: `model.safetensors`
- 训练日志: `trainer_state.json`
- 训练启动 PID: `products/task78_phonism_rqvae_768d/_TRAINING_PID` (已清理)

## 4. Stage 4 评估 (Task #84 接管)

见 `verdicts/task84_tiger_t5_inference_result.md`:

| Metric | Value |
|--------|-------|
| hit@1 | 0.0340 |
| hit@5 | 0.0484 |
| hit@10 | **0.0591** |
| NDCG@5 | 0.0416 |
| NDCG@10 | **0.0450** |

## 5. 与 paper Table 2 baseline #9 (TIGER) 对比

| 数据集 | Paper hit@10 | 复现 hit@10 | Δ |
|--------|--------------|-------------|----|
| Amazon Toys | 0.0679 | (无 Toys 复现) | — |
| **Musical_Instruments** | **未报** | **0.0591** | new reference |

Instruments 上 hit@10=0.0591 是该数据集上 TIGER 的新基线值。

## 6. Pipeline 完整性

| Stage | 状态 | 产物 / verdict |
|-------|------|----------------|
| Stage 1 LLM Embedding | ✅ | sentence-t5-768d.npy |
| Stage 2 RQ-VAE SID | ✅ | task78_phonism_rqvae_768d/ |
| Stage 3 TIGER 训练 | ✅ (Task #78) | 本任务 |
| Stage 4 TIGER inference | ✅ (Task #84) | hit@10=0.0591 |
| Stage 5 paper Table 2 排名 | ⏳ (Task #87) | 待综合 |

result: Task #78 TIGER T5 训练 (paper Table 2 baseline #9) 在 Musical_Instruments 数据集完成,
best checkpoint at step 51500, Stage 4 inference (Task #84) 得 hit@10=0.0591, NDCG@10=0.0450。
该数值为 Instruments 上 TIGER 新基线, 供 Task #87 paper Table 2 综合排名使用。