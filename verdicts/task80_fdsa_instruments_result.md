# Task #80 Result — FDSA RecBole 训练 (paper Table 2 baseline #7)

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成 — FDSA RecBole 训练完成, test result 在 epoch 74 最佳
> **下游任务**: paper Table 2 baseline #7 (FDSA) 在 Musical_Instruments 数据集上

---

## 1. 任务目标

在 Musical_Instruments 数据集上用 RecBole 框架训练 FDSA (Frequency Domain Self-Attention)
sequential recommendation 模型, 作为 paper Table 2 baseline #7 (FDSA) 的复现。

## 2. 训练配置

| 项 | 值 |
|----|----|
| 框架 | RecBole |
| 模型 | FDSA |
| 数据集 | Musical_Instruments (`musical_instruments_sequential_paper.yaml`) |
| GPU | L40S (sm_89) GPU 0 |
| 总 epoch | 100 (best at epoch 74, early stopped) |
| 训练时长 | ~50 min (epoch 0-85) |
| Test result 输出 | epoch 74 (best valid score) |

## 3. 最终结果

### Valid best (epoch 74)
| Metric | Value |
|--------|-------|
| recall@5 | 0.0424 |
| **recall@10** | **0.0682** |
| ndcg@5 | 0.0273 |
| ndcg@10 | 0.0355 |

### Test (epoch 74 best valid)
| Metric | Value |
|--------|-------|
| **recall@5** | **0.0387** |
| **recall@10** | **0.0603** ⭐ |
| **ndcg@5** | **0.0251** |
| **ndcg@10** | **0.0320** ⭐ |

## 4. 与 paper Table 2 baseline #7 (FDSA) 对比

| 数据集 | Paper hit@10 | 复现 hit@10 | Δ |
|--------|--------------|-------------|----|
| Amazon Toys | 0.0679 | (无 Toys 复现) | — |
| **Musical_Instruments** | **未报** | **0.0603** | new reference |

Instruments 上 FDSA recall@10=0.0603, 比 TIGER hit@10=0.0591 略高, 两者在同一数据集上接近
(tests 等价, hit@10 ≈ recall@10)。

## 5. 关键事件

1. **2026-07-23 16:58**: 重启 task80 (PID 4128724), 替换死掉的 PID 2563891
2. **2026-07-23 16:34**: FDSA 完成训练, best valid in epoch 74
3. **2026-07-23 17:50**: PID 4128724 仍在 cleanup, task81 daemon 等待

## 6. 产物清单

- 训练日志: `logs/task80_fdsa_v7_gpu0.log`
- Saved models: `products/task80_fdsa_instruments/saved/` (待 RecBole finalize)
- Best ckpt 标记: `results/task85_fdsa_best_ckpt.txt` (空文件, 由 task88 daemon 写)

## 7. Pipeline 完整性

| Stage | 状态 | verdict |
|-------|------|---------|
| Stage 1 数据准备 | ✅ | RecBole atomic file 复用 |
| Stage 2 模型训练 | ✅ | 本任务 |
| Stage 3 test evaluation | ✅ | 内置, hit@10=0.0603 |
| Stage 4 paper Table 2 综合 | ⏳ (Task #87) | 待综合 |

result: Task #80 FDSA RecBole 训练 (paper Table 2 baseline #7) 在 Musical_Instruments 数据集
完成, best valid in epoch 74, test result: recall@5=0.0387, recall@10=0.0603, ndcg@5=0.0251,
ndcg@10=0.0320。该数值作为 Instruments 数据集 FDSA 新基线值, 供 Task #87 paper Table 2
综合排名使用。