# Task #83 — P5-SID 复现 (paper Table 2 baseline #13)

> **任务目的**: 复现 LLM-RecSys-ID P5-SID 模型在 Musical_Instruments 数据集上的训练. 用作 paper Table 2 baseline #13 (sequential item representation).

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成 (Stage 3 训练完成 epoch 0, Stage 4 eval 由 Task #86 接管)

---

## 1. 背景

承接 Task #82 P5-CID (paper Table 2 #12) 的 LLM-RecSys-ID 框架, 同样在 Musical_Instruments 上跑 P5-SID (paper Table 2 #13, 用 sequential item representation 替代 collaborative-filtering clusters).

P5-SID 与 P5-CID 唯一区别:
- item_representation='remapped_sequential' (vs P5-CID 的 'remapped')
- cluster_number=55, cluster_size=100 (P5-CID 用 100+500)

---

## 2. 实验设计

**变量**: 改用 sequential item representation (vs CF cluster)
**保持不变**:
- LLM-RecSys-ID t5-small backbone
- Musical_Instruments 5-core 数据集
- 单 epoch 训练 (与 P5-CID 一致)
- seed=42

**启动命令**: `bash scripts/task83_p5_sid_train.sh` (后台 wait_then_launch 等 task82)

---

## 3. 决策触发

| 指标条件 | R@10 区间 | 决策 |
|----------|----------|------|
| ≥ 0.0438 (paper) | ≥ 0.0438 | ⭐ 复现成功 |
| 0.02-0.0438 | 0.02-0.0438 | 🟡 部分复现 |
| < 0.02 | < 0.02 | ❌ 复现失败 |

---

## 4. 关键产物

- best ckpt: `products/task83/p5_sid_instruments.pt` (243 MB, epoch 0 only)
- valid recall: **0.0358** at epoch 0 (log 17:42 batch)
- 训练日志: `logs/task83_p5_sid_jul-23-2026_17-42-31.log` (53000 steps in 6.5h)

---

## 5. 完成度

- [x] Stage 3 训练 (1 epoch, R12 ckpt 落盘)
- [x] Task #86 standalone eval 闭环 (P5-SID test metrics)

result: Task #83 P5-SID 训练完成 (epoch 0 only, valid recall=0.0358). 与 P5-CID 一致单 epoch. 实际 test metrics 由 Task #86 evaluate 输出.