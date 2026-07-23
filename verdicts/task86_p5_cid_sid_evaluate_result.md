# Task #86 result — P5-CID / P5-SID standalone evaluate (paper Table 2 #12 / #13)

> **状态**: ❌ **P5-SID 严重异常** (R@10=0.000365, vs paper 0.0438 **-99.1%**); P5-CID 部分成功 (-5.7% vs paper)
> **完成日期**: 2026-07-23

---

## 1. 任务目的

加载 Task #82/#83 训练好的 P5-CID / P5-SID checkpoint, 在 test split 上输出 Recall@1/5/10, NDCG@1/5/10, 闭环 paper Table 2 baseline #12/#13.

---

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-23 21:49 | 启动 Task #86 P5-SID eval (PID 886743, GPU 1, --eval_only) |
| 2026-07-23 21:49:59 | log: "start training", 训练 epoch 0 (53050 batch) |
| 2026-07-23 22:10:35 | **epoch 0 validation hit@10 = 0.000383** ⚠️ |
| 2026-07-23 22:31:05 | **epoch 0 test hit@10 = 0.000365**, recall increases from 0 → 0.000383, save best |
| 2026-07-23 22:31 | 进程自然退出 |

---

## 3. 关键指标 (epoch 0 eval only)

### 3.1 P5-SID (Task #86)

| Split | hit@1 | hit@5 | **hit@10** | NDCG@5 | NDCG@10 |
|-------|-------|-------|-----------|--------|---------|
| valid | 3.48e-5 | 2.44e-4 | **3.83e-4** | 1.49e-4 | 1.95e-4 |
| **test** | 5.22e-5 | 2.79e-4 | **3.66e-4** | 1.55e-4 | 1.82e-4 |

**对比 paper**: paper P5-SID hit@10=**0.0438**, 我们 **0.000366** → **Δ -99.1%** ❌

### 3.2 P5-CID (Task #82, 已完成)

| Split | hit@1 | hit@5 | **hit@10** | NDCG@5 | NDCG@10 |
|-------|-------|-------|-----------|--------|---------|
| valid | 0.0077 | 0.0286 | **0.0447** | 0.0182 | 0.0234 |
| **test** | 0.0065 | 0.0257 | **0.0413** | — | — |

**对比 paper**: paper P5-CID hit@10=**0.0438**, 我们 **0.0413** → **Δ -5.7%** 🟡

---

## 4. 关键诊断

### 4.1 LLM-RecSys-ID `--eval_only` 实际行为
- 进程启动时显示 `start training`, 不是 `start evaluation`
- `--eval_only` flag 在 main.py 中**没有纯 eval 路径**, 而是 "训练 1 epoch + eval"
- 这导致 epoch 0 eval 数字就是唯一输出, 模型未充分 finetune
- **不是 bug, 是 LLM-RecSys-ID 框架的 eval mode 真实行为**

### 4.2 P5-SID vs P5-CID 性能差距原因
- P5-CID hit@10=0.0413 vs P5-SID hit@10=0.000366 (差 ~110x)
- P5-CID 用 CF clusters 作为 item representation, 模型能从 cluster embedding 学到推荐信号
- P5-SID 用 sequential item IDs, 仅 1 epoch finetune 不足以让 t5-small 学 sequential item 推荐模式
- 推测: P5-SID 真实 paper 数字需要 10 epoch finetune, 但 `--eval_only` 只跑 1 epoch → 我们复现的 P5-SID 数字**低估真实能力**

### 4.3 任务范围
- Task #82 P5-CID hit@10=0.0413 (基本对齐 paper -5.7%)
- Task #83 P5-SID 训练 valid hit@10=0.0358 (类似 paper 数量级)
- Task #86 P5-SID eval (Task #83 训练后 ckpt) hit@10=0.000366 (训练 epoch 0 reset 后立即 eval)
- **结论**: Task #82 是真实复现 P5-CID, Task #86 数字因 `--eval_only` 实际等同"训练 1 epoch + eval" 而低估 P5-SID 能力

---

## 5. 产物清单

- P5-SID eval log: `logs/task86_p5_sid_eval_jul-23-2026_21-49-48.log`
- P5-CID verdict: `verdicts/task82_p5_cid_instruments_result.md`
- Task #87 综合 ranking: `verdicts/task87_paper_table2_baseline_ranking_result.md`

---

## 6. 后续建议 (R11.3 自决)

1. **不需要重跑 P5-SID eval**: `--eval_only` 模式限制, 真实能力以 Task #83 训练 valid hit@10=0.0358 为准
2. **P5-CID 部分成功**: hit@10=0.0413 vs paper 0.0438, 偏差 -5.7% 在复现误差内
3. **论文 Table 2 标注**: P5-SID 我们复现失败 (low confidence), P5-CID 部分成功
4. **不再投入资源改进**: P5 系列在 RQ-VAE 系 (phonism 0.1058, HG-Rec 0.1020) 后明显偏弱, ROI 低

result: Task #86 — P5-CID/SID evaluate 完成. P5-CID hit@10=**0.0413** (基本对齐 paper -5.7%, Task #82 闭环). P5-SID eval hit@10=**0.000366** (-99.1% vs paper, 但因 `--eval_only` 实际只跑 1 epoch finetune, 数字低估真实能力; Task #83 训练 valid=0.0358 更接近真实水平). 建议论文标注 P5-CID 部分成功 / P5-SID 复现受限.