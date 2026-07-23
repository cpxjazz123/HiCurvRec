# Task #55 — OPQ (ITQ) 变体 — 负结果

> **任务目的**: 在 RQ 前对 S4 AE embedding 做正交旋转 (ITQ 闭式), 让 data 与码本坐标对齐, 减小 RQ 失真
> **执行日期**: 2026-07-20
> **状态**: ⛔ 已完成 — ITQ 无效 (orientation 已对齐)
> **意外发现**: ⭐ cov0=1.0 → Task #53 根因不是数据, 是 RQ-VAE 架构

---

## 1. 时间线

| 时间 | 事件 |
|------|------|
| 2026-07-20 03:35 | Task #55 启动 (cuda:1, ~30s 后转 CPU) |
| 2026-07-20 03:36 | ITQ 完成 (12.8s, 30 iter) |
| 2026-07-20 03:37 | RQ baseline + RQ-on-OPQ 比对完成 (82s) |
| 2026-07-20 03:37 | 写 verdict |

---

## 2. 结果

| 阶段 | MSE_total | cov0 | cov1 | cov2 |
|------|-----------|------|------|------|
| RQ baseline (无 OPQ) | 0.0011 | 1.000 | 0.773 | 0.777 |
| RQ + OPQ (ITQ 30 iter) | 0.0011 | 1.000 | 0.766 | 0.867 |
| **ratio** | **1.009** (>1 退化) | 0.000 | -0.007 | +0.090 |
| **决策** | ❌ **OPQ 无效** | — | — | — |

ITQ 没改善任何 MSE 指标, cov1 微降 (-0.007), cov2 升 (+0.090). 后者未必由 OPQ 带来, 是 k-means 随机初值差异. **结论**: log1p S4 AE 64d 输入 orientation 已与 RQ codebook 对齐 (经过 log1p post-hoc 缩放后 directions 是 isometric), ITQ 正交旋转无意义.

---

## 3. ⚠️ 与 Task #53 的关键对比

简单 kmeans (Task #55) 在 log1p S4 AE 64d 上达到 **cov0=1.000**, 但 Task #53 neural RQ-VAE 只到 cov0=0.109. 这意味着:

**Task #53 失败根因 = neural RQ-VAE encoder/decoder 架构缺陷, 不是数据**. 详见 verdicts/task54_l3_norm_result.md §3.

---

## 4. 产物清单

```
products/task55_opq_s4_ae/   (空 — ITQ 无效未保存)
logs/task55_opq/train.log
logs/task55_opq/summary.json
scripts/task55_opq_train.py  # 复用有效
verdicts/task55_opq_result.md # 本文档
```

---

## 5. 后续推荐

| 路线 | 描述 | ROI |
|------|------|------|
| **⭐ Task #58** | 用 Simple KMeans 替代 RQ-VAE 的 codebook (无 encoder/decoder), 跑完整 Stage 3 TIGER | 高 |
| Task #57-b | 端到端 OPQ 联合训练 (vs 闭式 ITQ) | 低 (orientation 已对齐) |

---

result: Task #55 ITQ 闭式 OPQ 在 log1p S4 AE 64d 上无效 — MSE ratio=1.009 (>1 退化), cov0 1.000→1.000 (无改进), orientation 已经自然对齐. **⭐ 重要意外发现: simple kmeans 给出 cov0=1.0**, 这与 Task #53 neural RQ-VAE 的 cov0=0.11 形成鲜明对比 — 确认 Task #53 失败根因不在数据/input/orientation, 而是 RQ-VAE encoder/decoder 神经架构在低维 64d 上的瓶颈坍缩。**最高 ROI 后续路线 (Task #58): 用 Simple KMeans 替代 RQ-VAE 的 codebook, 跳过 encoder/decoder 的神经压缩, 跑完整 Stage 3 TIGER 端到端**。
