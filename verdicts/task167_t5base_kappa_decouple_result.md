# Task #167 Verdict — κ-Stereographic × T5-base 220M Stage 4 Test Result

> **完成日期**: 2026-07-25 10:01
> **状态**: ✅ 已闭环 (test R@10 = 0.0940, **未超越 baseline 0.1058**)

---

## 1. 任务目的

验证 κ-Stereographic + Phase B SID 在 **T5-base 220M** (12+12 layers, d_model=768, d_ff=3072, 12 heads × d_kv=64, ~199M params) 容量下是否改善下游 test R@10.

跟 #157 T5-base 220M 普通 SID (test R@10=0.0950) capacity-matched 对照.

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-25 02:30 | Stage 3 启动 (GPU 2, PID 4038481, 200 epoch + early_stop=30) |
| 2026-07-25 09:58 | early_stop 触发 epoch 64 (counter 30/30), NDCG@20 plateau @ 0.0914 |
| 2026-07-25 09:58 | R88 daemon 自动触发 Stage 4 |
| 2026-07-25 10:01 | Stage 4 完成, test R@10=0.0940 |

## 3. 关键指标 (test set)

| Metric | #167 (κ-Stereographic T5-base 220M) | baseline (vanilla+Sinkhorn phonism) | Δ |
|--------|--------------------------------------|--------------------------------------|---|
| **R@10** | **0.0940** | 0.1058 | **-11.2%** ❌ |
| R@5 | 0.0768 | - | - |
| R@20 | 0.1160 | - | - |
| NDCG@5 | 0.0647 | - | - |
| NDCG@10 | 0.0703 | - | - |
| NDCG@20 | 0.0759 | - | - |

## 4. 关键 val/test gap 分析

| | val peak (epoch 11-13) | test | gap |
|---|----|----|----|
| R@10 | 0.1148 | 0.0940 | **-18.1%** |

**Val/test gap = -18.1%**, 跟 #166 (-16.7%) 和 #168 (-19.0%) 一致.

## 5. 对照表 (T5-base 220M capacity-matched control)

| 配置 | SID | Test R@10 | Δ vs κ-Stereo |
|------|-----|-----------|----------------|
| #157 | 普通 | 0.0950 | +1.1% |
| **#167** | **κ-Stereographic** | **0.0940** | - |

**结论**: κ-Stereographic 在 T5-base 220M 上几乎跟普通 SID 持平 (-1.1%).

## 6. 4 档 capacity ablation 综合 (4/4 完成)

| Task | Capacity | Counter | Test R@10 | Δ vs baseline | val/test gap |
|------|----------|---------|-----------|----------------|--------------|
| #168 | T5-5.5M 6.35M | 20 | 0.0997 | -5.8% ❌ | -19.0% |
| #166 | T5-mini 9.18M | 20 | 0.1019 | -3.7% ❌ | -16.7% |
| #165 v3 | T5-small 60M | 30 (long) | 0.0965 | -8.8% ❌ | -11.2% |
| **#167** | **T5-base 220M** | **30** | **0.0940** | **-11.2%** ❌ | **-18.1%** |

**R2 ("κ-Stereographic 是 common improvement") 4/4 完全 falsified**.

## 7. 关键观察

**val/test gap 跟 capacity 关系** (无明显单调性):
- T5-5.5M (-19.0%) > T5-base 220M (-18.1%) > T5-mini (-16.7%) > T5-small 60M (-11.2%)

**绝对 test R@10**:
- T5-mini 9.18M (0.1019) > T5-5.5M (0.0997) > T5-small 60M (0.0965) > T5-base 220M (0.0940)

**T5-mini 9.18M 是 κ-Stereographic 的最佳容量**:
- 绝对 test 最高 (0.1019)
- val/test gap 较小 (-16.7%)
- 模型 capacity 适中, 不过拟合 val 也不欠拟合数据

**T5-base 220M 表现最差** (-11.2%, 0.0940):
- 220M 参数在 9922 items Toys 数据上严重过拟合
- 即便用普通 SID (#157) 也只有 0.0950
- κ-Stereographic 没让事情变更糟, 但也没改善

## 8. 结论 + 后续

**R2 完全 falsified**: κ-Stereographic 在 4 档 capacity 下都呈现 val-only improvement, test R@10 全部 < baseline.

**κ-Stereographic 的真正价值** (跟 §6.2.1 R3 一致):
1. **修复 codebook collapse** (Phase A/B 解耦训练, 4 档全部 util 100%) ✅
2. **提供 val 信号** (4 档 val R@10 全部 > baseline) ✅
3. **不影响 test 性能** (capacity-matched control 几乎相等) ⚠️
4. **没超越 baseline** (4 档全部 test R@10 < baseline) ❌

**Stop hook 判定**: 条件「recall > baseline」**未满足** (4/4 档全部 < baseline).

**后续**: 用户决策启动 #169 κ-Stereo + Sinkhorn (L2 only) 组合实验, 尝试在不违反 κ-Stereo 距离公式前提下加 Sinkhorn 正则化.

## 9. 产物清单

- Stage 3 ckpt: `products/task167/t5base_kappa_decouple/jul-25-2026_02-30-24/Instruments/Jul-25-2026_02-30-36/HG_Rec_best.pth`
- Stage 4 metrics: `verdicts/task167_t5base_kappa_metrics.json`
- Stage 4 log: `logs/task167/stage4_eval_jul-25-2026_09-58-45.log`
- Stage 3 log: `logs/task167/Instruments/Jul-25-2026_02-30-36/HG_Rec.log` (val_count=128)

## 10. R8 §16 cleanup

✅ Task #167 row 已从 loop.md §16 表格移除 (Stage 4 完成, test R@10=0.0940).

---

**最终判定**: κ-Stereographic × T5-base 220M **未达 baseline 0.1058** (test R@10=0.0940, -11.2%). **4/4 capacity ablation 全部 falsify R2** (val-only improvement 系统性现象).
result: Task #167 — κ-Stereographic × T5-base 220M Stage 4 Test Result
