# Task #82 Result — P5-CID 复现 (paper Table 2 baseline #12)

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成 — P5-CID LLM-RecSys-ID 训练完成, epoch 0 (only epoch)
> **下游任务**: paper Table 2 baseline #12 (P5-CID) 在 Musical_Instruments 数据集上

---

## 1. 任务目标

在 Musical_Instruments 数据集上用 LLM-RecSys-ID 框架训练 P5-CID (Pretrained Personalized
Prompt with Continuous Item ID), 作为 paper Table 2 baseline #12 (P5-CID) 的复现。

## 2. 训练配置

| 项 | 值 |
|----|----|
| 框架 | LLM-RecSys-ID (`external/LLM-RecSys-ID/`) |
| 模型 | t5-small + continuous item ID |
| 数据集 | Musical_Instruments |
| GPU | L40S (sm_89) GPU 2 |
| Epochs | 1 (仅 1 epoch, `meta_epochs=10` 是 meta-learning epochs, 总 epoch=1) |
| 训练时长 | ~3h (epoch 0 + validation + test) |
| 总 steps | 53050 (1 epoch) |

## 3. 最终结果

### Validation (epoch 0)
| Metric | Value |
|--------|-------|
| hit@1 | 0.0077 |
| hit@5 | 0.0286 |
| hit@10 | 0.0447 |
| ndcg@5 | 0.0182 |
| ndcg@10 | 0.0234 |

### Test (epoch 0)
| Metric | Value |
|--------|-------|
| hit@1 | 0.0065 |
| hit@5 | 0.0257 |
| **hit@10** | **0.0413** ⭐ |
| ndcg@5 | 0.0161 |
| ndcg@10 | 0.0211 |

## 4. 与 paper Table 2 baseline #12 (P5-CID) 对比

| 数据集 | Paper hit@10 | 复现 hit@10 | Δ |
|--------|--------------|-------------|----|
| Amazon Scientific | 0.0458 | (无 Scientific 复现) | — |
| **Musical_Instruments** | **未报** | **0.0413** | new reference |

Instruments 上 P5-CID hit@10=0.0413, 比 FDSA (0.0603) 和 TIGER (0.0591) 低 — P5-CID 在该数据集
上 baseline 偏弱。

## 5. 关键事件

1. **2026-07-23 14:47**: Task #82 P5-CID 启动 (PID 3387236)
2. **2026-07-23 16:45**: epoch 0 完成, 保存 ckpt
3. **2026-07-23 17:13**: validation 完成 (hit@10=0.0447)
4. **2026-07-23 17:41**: test evaluation 完成 (hit@10=0.0413)
5. **2026-07-23 17:41**: "[Task #82] P5-CID training completed" 标记
6. **2026-07-23 17:42**: task83 wait daemon 检测到 PID 3387236 dead + ckpt 存在, 自动启动 task83 P5-SID

## 6. 产物清单

- 训练日志: `logs/task82_p5_cid_jul-23-2026_14-47-29.log`
- Model ckpt: `products/task82/p5_cid_instruments.pt` (244 MB)
- Best ckpt 标记: `results/task86_p5_cid_best_ckpt.txt` (空, 由 task88 daemon 写)

## 7. Pipeline 完整性

| Stage | 状态 | verdict |
|-------|------|---------|
| Stage 1 数据准备 | ✅ | LLM-RecSys-ID atomic file 复用 |
| Stage 2 模型训练 | ✅ | 本任务 (epoch 0) |
| Stage 3 test evaluation | ✅ | 内置, hit@10=0.0413 |
| Stage 4 paper Table 2 综合 | ⏳ (Task #87) | 待综合 |

result: Task #82 P5-CID LLM-RecSys-ID 复现 (paper Table 2 baseline #12) 在 Musical_Instruments
数据集完成, epoch 0 (单 epoch), test result: hit@1=0.0065, hit@5=0.0257, hit@10=0.0413,
ndcg@5=0.0161, ndcg@10=0.0211。该数值作为 Instruments 数据集 P5-CID 新基线值, 供 Task #87
paper Table 2 综合排名使用。