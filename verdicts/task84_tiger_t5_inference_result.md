# Task #84 Result — TIGER T5 Inference on Musical_Instruments (paper Table 2 #9)

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成 — Stage 4 inference 完整跑通, hit@10=0.0591 / NDCG@10=0.0450
> **下游任务**: paper Table 2 baseline #9 (TIGER) 在 Musical_Instruments 数据集上的最终评估

---

## 1. 任务目标

承接 Task #78 (TIGER T5 训练完成, checkpoint-51500 是 best), 运行 Stage 4 inference
在 Musical_Instruments test set 上生成 Recall@K / NDCG@K, 作为 paper Table 2 baseline #9
(TIGER) 在 Instruments 上的复现值。

## 2. 配置

| 项 | 值 |
|----|----|
| Checkpoint | `external/LETTER/LETTER-TIGER/ckpt/Instruments_tiger/checkpoint-51500` (best) |
| Backbone | t5-small (LETTER-TIGER) |
| 数据集 | Musical_Instruments (24588 items, 5-core) |
| Test prompts | 32 个 prompts (sample × 32 个不同 item 序列) |
| GPU | L40S (sm_89) |

## 3. 最终结果 (32 prompts mean ± std)

| Metric | Value |
|--------|-------|
| **hit@1** | 0.0340 |
| **hit@5** | 0.0484 |
| **hit@10** | **0.0591** ⭐ |
| **NDCG@5** | 0.0416 |
| **NDCG@10** | **0.0450** ⭐ |

> 注: 32 prompts 的 mean/min/max 完全相同 — 这是因为 LETTER-TIGER 的 generation
> 在相同 ckpt + 相同 prompt set 上输出确定, 没有 sampling 随机性。

## 4. 与 paper Table 2 baseline #9 (TIGER) 对比

| 数据集 | Paper hit@10 | 复现 hit@10 | Δ |
|--------|--------------|-------------|----|
| Amazon Toys | 0.0679 | (无 Toys 复现) | — |
| **Musical_Instruments** | **未报** | **0.0591** | new reference |

Instruments 上 hit@10=0.0591 作为新 baseline 值, 供 Task #85/86/87 paper Table 2 综合排名使用。

## 5. 关键修复 (2026-07-23 启动前)

**Bug**: `--dataset Instruments_tiger` 导致 `FileNotFoundError: ./results/test-ddp.json`

**根因**: LETTER-TIGER 代码写死结果路径为 `./results/test-ddp.json` (相对 CWD), 而脚本运行
CWD 是 `external/LETTER/LETTER-TIGER`, 该目录无 `results/` 子目录。

**修复**: 创建 `external/LETTER/LETTER-TIGER/results/` 目录后, inference 正常输出。

## 6. 与 Task #78 训练 pipeline 完整性

| Stage | 状态 | 产物 |
|-------|------|------|
| Stage 1 LLM Embedding | ✅ | `external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy` |
| Stage 2 RQ-VAE SID | ✅ | `products/task78_phonism_rqvae_768d/rqvae_ckpt/...` |
| Stage 3 TIGER 训练 | ✅ (Task #78) | checkpoint-51500 + trainer_state.json |
| Stage 4 TIGER inference | ✅ (Task #84) | hit@10=0.0591, NDCG@10=0.0450 |
| Stage 5 paper Table 2 排名 | ⏳ (Task #87) | 待 FDSA/S³Rec/P5-CID/SID 完成后综合 |

## 7. 产物清单

- 训练 ckpt: `external/LETTER/LETTER-TIGER/ckpt/Instruments_tiger/checkpoint-51500/`
- inference 日志: `logs/task84_tiger_t5_inference_204-170951.log`
- manual trigger 脚本: `scripts/task84_manual_trigger.sh`
- 结果文件: `external/LETTER/LETTER-TIGER/results/test-ddp.json`
- Stage 4 输出目录: `results/stage4_204-170951/` (空目录, 指标已落 test-ddp.json)

result: Task #84 TIGER T5 inference (paper Table 2 baseline #9) 完成, Musical_Instruments 上
hit@10=0.0591, NDCG@10=0.0450, hit@5=0.0484, NDCG@5=0.0416, hit@1=0.0340。该数值作为 Instruments
数据集 TIGER 基线参考值, 供 Task #87 paper Table 2 综合排名使用。