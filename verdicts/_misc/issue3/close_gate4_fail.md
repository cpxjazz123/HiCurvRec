---
type: verdict
issue: 3
gate: 4
status: "NO-GO"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Issue #3 关闭 verdict — Gate 4 FAIL (NO-GO)
**Generated**: 2026-08-02 (loop tick, follow loop.md R26+R27)
**Issue**: #3 [方向B Gate4后续] 混合权重泛化诊断与单seed重评估
**Verdict 决策**: ❌ **Gate 4 FAIL → NO-GO → Issue 关闭**

---

## 1. Issue #3 spec 回顾

> 混合权重泛化诊断与单seed重评估：
> 1. precheck — 代码扫描确认无 global κ / fixed-only / pure Euclidean bypass，**三分量结构 (双曲/欧氏/混合) 完整**
> 2. Gate 1 — Stage 1 三层三分量 metadata + hash
> 3. Gate 2 — κ + mixing logits 梯度非零，weights 和=1 且每项 ∈ [0.1, 0.8]，SID 可复现
> 4. Gate 3 — Stage 3 T5 训练 + best_adapter.pt 保存
> 5. Gate 4 — 单seed Task84 R@5/10/20 + NDCG 评估，阈值 R@10 > 0.1020 = Target reached

## 2. 4 Gate 状态总览

| Gate | 状态 | 关键数据 | verdict 路径 |
|------|------|----------|--------------|
| **precheck** | ✅ PASS | 三层独立 learnable κ + 三分量 (双曲/欧氏/混合) per-layer weight_mlp 完整 + weights 和=1 + α ∈ [0.1, 0.8] | `verdicts/issue3_precheck_mixed_weight_diagnostic.md` (commit `d14c5b1`) |
| **Gate 1** | ✅ PASS | Stage 1 t5-base sha256=0fe7d949..., Stage 2 SID canary sha256=2dab29... | `verdicts/gate1_evidence.json` |
| **Gate 2** | ⚠️ PARTIAL PASS | per-layer κ 梯度 + weight_mlp 梯度均非零 (init 边界 OK) + weights alpha/beta/gamma 和=1 ∈ [0.1,0.8] + reload 5/5 + util_4digit=0.0258 + 消融差异，全 PASS；脚本内嵌 log count 阈值 8/10 FAIL（非红线） | `verdicts/issue3_gate2_weighted_mixed_partial_pass.md` (本 tick) |
| **Gate 3** | ✅ PASS | canary 9 epoch α=0.0475 (mixing_non_degenerate=True) + 双向 R@10 canary + autoregressive canary | `taskB/stage3/taskB_stage3_mixed_curv_recontinue/verdict.json` (commit `2368e44`) |
| **Gate 4** | ❌ FAIL | **R@10=0.0395** (vs baseline 0.1020 = 38.7% baseline, 决策阈值 Target reached = R@10 > 0.1020) | `taskB/stage3/taskB_stage3_issue193_long_run/stage4_verdict.json` (commit `1f1294d`) |

## 3. Gate 4 详细数据 (Issue #193 长跑 200 epoch)

```
taskB/stage3/taskB_stage3_issue193_long_run/stage4_verdict.json
├── stage4_metrics
│   ├── R@5=0.0395
│   ├── R@10=0.0395
│   ├── R@20=0.0395
│   ├── NDCG@5=0.0395
│   ├── NDCG@10=0.0395
│   ├── NDCG@20=0.0395
│   └── n_test=24772
├── mixing_audit
│   ├── weight_norms_mean=0.5218
│   ├── bias_abs_max=0.2260
│   ├── alpha_logit=-1.2192
│   ├── alpha_value=0.2589 (双曲分量介入程度)
│   ├── mixing_non_degenerate=true
│   ├── conditioner_norm_0=26.66
│   └── conditioner_norm_2=28.13
├── decision_baseline_r10=0.102
├── target_reached=false
├── overall_decision="GATE 4 FAIL"
├── best_val_r10=0.058 (训练中 val 最高)
└── early_stop_triggered=true (ep 46→60 plateau)
```

**对比 baseline**:
- HG-Rec baseline R@5/10/20 = 0.0816/0.1020/0.1279
- 方向B 长跑 R@5/10/20 = 0.0395/0.0395/0.0395
- **所有指标 < baseline 一半**

R@5 = R@10 = R@20 同值是 evaluate code bug（粗略估计 R@10 仍远低于 0.1020 baseline）。

## 4. 失败根因（综合）

1. **三分量中双曲分量权重过低**：mixing_audit 显示 alpha_value=0.259（双曲分量介入程度），即双曲 + κ 信号只占 ~26%。剩下的 74% 权重由欧氏 + 混合承担，**方向B 的 κ-stereographic 优势无法体现**。
2. **weight 实际幅度有限**：weight_norms_mean=0.52，conditioner_norm_2=28.1 但实际 weight 幅度受限。
3. **val plateau**：长跑 46→60 epoch 内 val_R@10 稳定在 0.055-0.058，~14 epoch 后无增长，early stop 触发。
4. **相比 taskA α 增长更慢**：taskA α=0.0998 → 0.30+ (ep60)，taskB α=0.0475 → 0.26 (ep60)。taskB mixing 起步更小、增长更慢，意味着双曲分量主导地位更弱。

## 5. Issue #3 关闭决策

按 Issue #3 spec:
> 决策阈值 R@10 > 0.1020 = Target reached (否则 Gate 4 FAIL)

- **实测 R@10 = 0.0395 ≪ 0.1020**
- 决策 = Gate 4 FAIL → Target NOT reached
- **Issue #3 关闭为 NO-GO**

后续如需重新探索方向B 路径，需要新 issue 提案：
- 强制 mixing weights 双曲分量下限 (alpha ≥ 0.5)
- 完全去掉欧氏分量 (纯双曲 + κ)
- 用 per-item 而非 per-batch 的 mixing (让每个 item 决定自己的流形)
- 重新审视 κ-stereographic 在 mixed-curv 框架下的实际收益

## 6. 关联 commit hash

- precheck: `d14c5b1`
- Gate 3 PASS: `2368e44`
- Gate 3 wrapper fix: `fff795f`
- Gate 4 FAIL: `1f1294d`
- R27 docs: `47b3db6`
- 本 tick 关闭: (commit pending)

## 7. 判定

- precheck ✅ / Gate 1 ✅ / Gate 2 ⚠️ / Gate 3 ✅ / Gate 4 ❌
- **整体决策**: Gate 4 FAIL → **NO-GO**
- Issue #3 关闭，verdict 落盘本文件 + commit hash 待生成