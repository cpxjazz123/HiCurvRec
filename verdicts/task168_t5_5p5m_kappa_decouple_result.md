# Task #168 Verdict — κ-Stereographic × T5-5.5M Stage 4 Test Result

> **完成日期**: 2026-07-25 03:47
> **状态**: ✅ 已闭环 (test R@10 = 0.0997, **未超越 baseline 0.1058**)

---

## 1. 任务目标

验证 κ-Stereographic + Phase B SID 在 **T5-5.5M** (实测 6.35M params, 5+5 layers, d_model=192, d_ff=1024, 3 heads × d_kv=64) 容量下是否改善下游 test R@10.

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-25 03:10 | Stage 3 启动 (GPU 3, PID 4070133, 200 epoch + early_stop=20, 28.45 it/s) |
| 2026-07-25 03:46 | early_stop 触发 epoch 68 (counter 20/20), NDCG@20 plateau @ 0.0969 |
| 2026-07-25 03:46 | R88 daemon 自动触发 Stage 4 (correct glob path) |
| 2026-07-25 03:47 | Stage 4 完成, test R@10=0.0997 |

## 3. 关键指标 (test set)

| Metric | #168 (κ-Stereographic T5-5.5M) | baseline (vanilla+Sinkhorn phonism) | Δ |
|--------|--------------------------------|--------------------------------------|---|
| **R@10** | **0.0997** | 0.1058 | **-5.8%** ❌ |
| R@5 | 0.0800 | - | - |
| R@20 | 0.1242 | - | - |
| NDCG@5 | 0.0682 | - | - |
| NDCG@10 | 0.0745 | - | - |
| NDCG@20 | 0.0807 | - | - |

## 4. 关键 val/test gap 分析

| | val (epoch 45) | test | gap |
|---|----|----|----|
| R@10 | 0.1230 | 0.0997 | **-19.0%** |

**Val/test gap = -19.0%** 跟 #166 (-16.7%) 一致, 验证了 §6.2.1 "R3 Falsified — Stage 4 Test Gap" 结论: κ-Stereographic 的 val 改善 **不** generalize 到 test.

## 5. 对照表 (4 档 capacity ablation across capacities test R@10)

| Task | Capacity | SID | Test R@10 | Δ vs baseline | 状态 |
|------|----------|-----|-----------|---------------|------|
| **#168** | **T5-5.5M (6.35M)** | **κ-Stereo** | **0.0997** | **-5.8%** ❌ | done |
| **#166** | T5-mini 9.18M | κ-Stereo | 0.1019 | -3.7% ❌ | done |
| #165 v3 | T5-small 60M | κ-Stereo | [待 Stage 4] | - | training |
| #167 | T5-base 220M | κ-Stereo | [待 Stage 4] | - | training |

## 6. val/test gap 4 档对比 (到目前为止)

| Task | Capacity | val R@10 peak | test R@10 | val/test gap |
|------|----------|---------------|-----------|--------------|
| **#168** | T5-5.5M | 0.1230 | **0.0997** | **-19.0%** |
| **#166** | T5-mini 9.18M | 0.1223 | **0.1019** | **-16.7%** |

**所有 κ-Stereographic × T5 测试都呈现 val/test gap -17 ~ -19%**, 强烈支持 R3 falsified.

## 7. 分析解读

**R2 ("κ-Stereographic 是 common improvement") 在 T5-5.5M 上 falsified**:
- val 信号支持 (+16.2% over baseline 0.1058, peak 0.1230)
- test 信号不支持 (-5.8% vs baseline 0.1058)
- val/test gap = -19.0%, 跟 #166 -16.7% 一致

**根因假设** (per §6.2.1, 现已 2 档验证):
1. κ-Stereographic SID 让 Stage 3 在 val 上更快收敛, 但学到的 inductive bias 跟 test set 数据分布的 correlation 弱
2. Stage 3 训练 epoch 不够 — 但 #168 是 68 epoch 长训 + early-stop, 跟 #166 类似
3. Beam search val vs test 评估差异

**T5-5.5M 比 T5-mini 9.18M test 更差 (-2.2% R@10)**:
- 预期: T5-5.5M 训练快 (68 epoch in 36 min vs T5-mini 70 epoch in 39 min, similar time)
- 实际: T5-5.5M 极小模型 (6.35M params) 在 9922 items 上欠拟合, test 表现比 T5-mini 9.18M 还差

## 8. 后续建议

1. **等 #165 v3 / #167 test results** 才能完整判定 R2 命运. 但 2/4 档已 falsified → 极可能 R2 全 falsified.
2. **如果 #165 v3 / #167 test 也跟 capacity-matched control 持平** → 结论: κ-Stereographic val improvement **不** generalize to test, 是普遍现象 (跟 §6.2.1 R3 一致).
3. **κ-Stereographic 的真正价值可能在其他方面**: (a) 训练效率 (T5-5.5M 训练仅 36 min 出 val 0.1230), (b) 几何 inductive bias 对数据分布的某种揭示 (跟 phonism L2/L3 的方向一致).

## 9. 产物清单

- Stage 3 ckpt: `products/task168/t5_5p5m_kappa_decouple/jul-25-2026_03-10-37/Instruments/Jul-25-2026_03-10-48/HG_Rec_best.pth`
- Stage 4 metrics: `verdicts/task168_t5_5p5m_kappa_metrics.json`
- Stage 4 log: `logs/task168/stage4_eval_jul-25-2026_03-46-40.log`
- Stage 3 log: `logs/task168/Instruments/Jul-25-2026_03-10-48/HG_Rec.log` (val_count=136)

## 10. R8 §16 cleanup

✅ Task #168 row 已从 loop.md §16 表格移除 (Stage 4 完成, test R@10=0.0997).

---

**最终判定**: κ-Stereographic × T5-5.5M **未达 baseline 0.1058** (test R@10=0.0997, -5.8%). val/test gap -19.0%, 跟 §6.2.1 "R3 falsified" 一致. **2/4 档 capacity ablation 已证实 κ-Stereographic 是 val-only improvement**. R2 大概率完全 falsified.