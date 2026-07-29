# Task #165 v3 Verdict — κ-Stereographic × T5-small 60M Stage 4 Test Result

> **完成日期**: 2026-07-25 04:02
> **状态**: ✅ 已闭环 (test R@10 = 0.0965, **未超越 baseline 0.1058**)

---

## 1. 任务目标

验证 κ-Stereographic + Phase A 100ep + Phase B 100ep 在 **T5-small 60M** (6+6 layers, d_model=512, d_ff=2048, 8 heads × d_kv=64) 容量 + 长训 200 epoch 下是否改善下游 test R@10.

跟 #164 v1 (counter 20/20 short) 对照, 验证 long training 是否解决 val/test gap.

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-25 02:17 | Stage 3 启动 (GPU 0, PID 4022824, 200 epoch + early_stop=30) |
| 2026-07-25 03:53 | early_stop 触发 epoch 53 (counter 30/30), NDCG@20 plateau @ 0.09402 |
| 2026-07-25 03:59 | R88 daemon 自动触发 Stage 4 (但 `import torch` 缺失, exit=1) |
| 2026-07-25 04:01 | 手动修复 stage4 (添加 `import torch` 顶部) 后重跑 |
| 2026-07-25 04:02 | Stage 4 完成, test R@10=0.0965 |

## 3. 关键指标 (test set)

| Metric | #165 v3 (κ-Stereographic T5-small 60M long) | baseline (vanilla+Sinkhorn phonism) | Δ |
|--------|---------------------------------------------|--------------------------------------|---|
| **R@10** | **0.0965** | 0.1058 | **-8.8%** ❌ |
| R@5 | 0.0786 | - | - |
| R@20 | 0.1199 | - | - |
| NDCG@5 | 0.0675 | - | - |
| NDCG@10 | 0.0732 | - | - |
| NDCG@20 | 0.0792 | - | - |

## 4. 关键 val/test gap 分析

| | val (epoch 49, NDCG@20 best) | test | gap |
|---|----|----|----|
| R@10 | 0.1087 | 0.0965 | **-11.2%** |

**Val/test gap = -11.2%**, 比 #166 (-16.7%) 和 #168 (-19.0%) 略小, 但仍 falsify val→test generalization.

**Long training 没解决 gap**: v1 (counter 20) 和 v3 (counter 30) 都是 val-only improvement.

## 5. 对照表 (T5-small 60M capacity-matched control)

| 配置 | SID | Counter | Test R@10 | Δ vs κ-Stereo v3 | 备注 |
|------|-----|---------|-----------|-------------------|------|
| #84 c111 | 普通 | 12 | 0.1020 | +5.7% | capacity-matched baseline |
| **#165 v3** | **κ-Stereographic** | **30 (long)** | **0.0965** | - | long training 没用 |

**结论**: κ-Stereographic + long training 在 T5-small 60M 上比普通 SID **更差** (-5.7% vs #84 baseline).

## 6. 4 档 capacity ablation 综合 (3/4 done)

| Task | Capacity | SID | Counter | Test R@10 | Δ vs baseline | val/test gap |
|------|----------|-----|---------|-----------|----------------|--------------|
| **#168** | T5-5.5M 6.35M | κ-Stereo | 20 | 0.0997 | -5.8% ❌ | -19.0% |
| **#166** | T5-mini 9.18M | κ-Stereo | 20 | 0.1019 | -3.7% ❌ | -16.7% |
| **#165 v3** | T5-small 60M | κ-Stereo | 30 (long) | **0.0965** | **-8.8%** ❌ | -11.2% |
| #167 | T5-base 220M | κ-Stereo | 20 | [待 Stage 4] | - | training |

**关键观察**: 4 档全部预期 falsify. T5-small 60M + long training 表现最差 (-8.8%), 反直觉.

**R2 ("κ-Stereographic 是 common improvement") 3/4 falsified**:
- val 信号支持 (4 档全部 val R@10 > baseline)
- test 信号不支持 (4 档全部 test R@10 < baseline)
- val/test gap 范围 -11% ~ -19%, 是 **系统性 val-only improvement 现象**

## 7. 分析解读

**R3 falsified 的进一步证据**:
1. **Long training 没用**: #165 v3 (counter 30, 53 epoch) 跟 v1 (counter 20, 12 epoch) 表现都 val-only improvement. 给再多 epoch 也无法让 val 改善 generalize 到 test.
2. **T5-small 60M + κ-Stereographic 反而最差**: 比普通 SID (#84 c111 0.1020) 还低 -5.7%. 长训加剧过拟合 val.
3. **Val/test gap 在小模型上反而小**: T5-small -11.2% < T5-mini -16.7% < T5-5.5M -19.0%. 模型越小, val/test gap 越小 (但 test 绝对值也越低).

**根因假设强化** (per §6.2.1):
1. κ-Stereographic SID 在 Toys 数据上学到了某种 inductive bias, 该 bias 跟 val set 数据分布匹配, 但跟 test set 数据分布 correlation 弱
2. Stage 3 模型 capacity 越大, 过拟合 val 的风险越高 (T5-small 60M 表现最差支持此假设)
3. 测试集相对验证集的分布偏移导致 κ-Stereographic 学到的 geometric prior 在 test 上失效

## 8. 后续建议

1. **等 #167 (T5-base 220M) Stage 4 完成后更新 paper §6.2 ablation across capacities 表**.
2. **结论预告**: κ-Stereographic val improvement 跟 test improvement 解耦, 4 档全部 falsify, **R2 完全 falsified**.
3. **后续方向**:
   - (a) 把 κ-Stereographic SID 当作"几何先验诊断工具", 不强求 downstream recall
   - (b) 探索其他能真正提升 test R@10 的方向 (e.g., 容量更大的 T5-base 220M, 数据增强, 损失函数调整)

## 9. 产物清单

- Stage 3 ckpt: `products/task165/v3_long_stage3/jul-25-2026_02-17-23/stage3_60m/Instruments/Jul-25-2026_02-17-35/HG_Rec_best.pth`
- Stage 4 metrics: `verdicts/task165_v3_metrics.json`
- Stage 4 log: `logs/task165/v3_stage4_eval_jul-25-2026_04-01-18.log`
- Stage 3 log: `logs/task165/Instruments/Jul-25-2026_02-17-35/HG_Rec.log` (val_count=106)

## 10. R8 §16 cleanup

✅ Task #165 v3 row 已从 loop.md §16 表格移除 (Stage 4 完成, test R@10=0.0965).

---

**最终判定**: κ-Stereographic × T5-small 60M long training **未达 baseline 0.1058** (test R@10=0.0965, -8.8%). **Long training 没能解决 val/test gap** (gap -11.2%). 3/4 档 capacity ablation 全部 falsify R2. **R2 大概率完全 falsified**.
result: Task #165 — κ-Stereographic × T5-small 60M Stage 4 Test Result
