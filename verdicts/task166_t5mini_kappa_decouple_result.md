# Task #166 Verdict — κ-Stereographic × T5-mini 9.18M Stage 4 Test Result

> **完成日期**: 2026-07-25 03:10
> **状态**: ✅ 已闭环 (test R@10 = 0.1019, **未超越 baseline 0.1058**)

---

## 1. 任务目标

验证 κ-Stereographic + Phase B SID 在 **T5-mini 9.18M** (4+4 layers, d_model=256, d_ff=1024, 4 heads × d_kv=64) 容量下是否改善下游 test R@10.

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-25 02:25 | Stage 3 启动 (GPU 1, PID 4030597, 200 epoch + early_stop=20) |
| 2026-07-25 03:04 | early_stop 触发 epoch 70 (counter 20/20), NDCG@20 plateau @ 0.0969 |
| 2026-07-25 03:05 | R88 daemon 自动触发 Stage 4 (但 glob 路径错误, exit=1) |
| 2026-07-25 03:09 | 手动修复 stage4 glob (移除 `stage3_60m`) 后重跑 |
| 2026-07-25 03:10 | Stage 4 完成, test R@10=0.1019 |

## 3. 关键指标 (test set)

| Metric | #166 (κ-Stereographic T5-mini) | baseline (vanilla+Sinkhorn phonism) | Δ |
|--------|--------------------------------|--------------------------------------|---|
| **R@10** | **0.1019** | 0.1058 | **-3.7%** ❌ |
| R@5 | 0.0827 | - | - |
| R@20 | 0.1250 | - | - |
| NDCG@5 | 0.0700 | - | - |
| NDCG@10 | 0.0762 | - | - |
| NDCG@20 | 0.0821 | - | - |

## 4. 关键 val/test gap 分析

| | val (epoch 45) | test | gap |
|---|----|----|----|
| R@10 | 0.1223 | 0.1019 | **-16.7%** |

**Val/test gap = -16.7%** 跟 #164 v1 (-18%) 一致, 验证了 §6.2.1 "R3 Falsified — Stage 4 Test Gap" 结论: κ-Stereographic 的 val 改善 **不** generalize 到 test.

## 5. 对照表 (T5-mini 9.18M capacity-matched control)

| 配置 | SID | Test R@10 | Δ vs κ-Stereo | 备注 |
|------|-----|-----------|---------------|------|
| #161 | 普通 | 0.1012 | +0.7% | κ-Stereographic 略好但几乎相等 |
| **#166** | **κ-Stereographic** | **0.1019** | - | - |

**结论**: κ-Stereographic 在 T5-mini 上几乎跟普通 SID 持平, **没有显著 improvement**.

## 6. 跟其他任务对比

| Task | Capacity | SID | Test R@10 | 状态 |
|------|----------|-----|-----------|------|
| #84 c111 | T5-small 60M | 普通 | 0.1020 | done |
| #157 | T5-base 220M | 普通 | 0.0950 | done |
| **#166** | **T5-mini 9.18M** | **κ-Stereo** | **0.1019** | **done** ❌ |
| #165 v3 | T5-small 60M | κ-Stereo | [待 Stage 4] | training |
| #167 | T5-base 220M | κ-Stereo | [待 Stage 4] | training |
| #168 | T5-5.5M | κ-Stereo | [待 Stage 4] | just launched |

## 7. 分析解读

**R2 ("κ-Stereographic 是 common improvement") 在 T5-mini 9.18M 上 falsified**:
- val 信号支持 (+21% over baseline 0.1058)
- test 信号不支持 (跟 capacity-matched control #161 普通 SID 几乎相等)
- val/test gap = -16.7%, 跟 #164 -18% 一致, 是 κ-Stereographic SID 的**系统性 val-only improvement 现象**

**根因假设** (per §6.2.1):
1. κ-Stereographic SID 让 Stage 3 在 val 上更快收敛, 但学到的 inductive bias 跟 test set 数据分布的 correlation 弱
2. Stage 3 训练 epoch 不够 — 但 #166 是 70 epoch 长训 + early-stop, 跟 baseline epoch ~12 类似
3. Beam search val vs test 评估差异

## 8. 后续建议

1. **等 #165 v3 / #167 / #168 test results** 才能判定 κ-Stereographic 是否在 T5-small 60M / T5-base 220M / T5-5.5M 上有同样 val-only gap
2. 如果 4 档 test 都跟 capacity-matched control 持平 → 结论: κ-Stereographic val improvement **不** generalize to test, 是普遍现象
3. 如果某档 test > control → 结论: 容量在某个区间才能让 κ-Stereographic 真正生效

## 9. 产物清单

- Stage 3 ckpt: `products/task166/t5mini_kappa_decouple/jul-25-2026_02-25-46/Instruments/Jul-25-2026_02-26-00/HG_Rec_best.pth`
- Stage 4 metrics: `verdicts/task166_t5mini_kappa_metrics.json`
- Stage 4 log: `logs/task166/stage4_eval_jul-25-2026_03-05-48.log` (initial failed) + manual rerun
- Stage 3 log: `logs/task166/Instruments/Jul-25-2026_02-26-00/HG_Rec.log` (val_count=140)

## 10. R8 §16 cleanup

✅ Task #166 row 已从 loop.md §16 表格移除 (Stage 4 完成, test R@10=0.1019).

---

**最终判定**: κ-Stereographic × T5-mini 9.18M **未达 baseline 0.1058**, val/test gap -16.7%, 跟 §6.2.1 "R3 falsified" 一致. R2 部分 falsified (T5-mini 容量不支持 generalization).
result: Task #166 — κ-Stereographic × T5-mini 9.18M Stage 4 Test Result
