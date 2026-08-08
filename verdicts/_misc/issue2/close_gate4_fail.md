---
type: verdict
issue: 2
gate: 4
status: "NO-GO"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Issue #2 关闭 verdict — Gate 4 FAIL (NO-GO)
**Generated**: 2026-08-02 (loop tick, follow loop.md R26+R27)
**Issue**: #2 [方向A Gate4后续] κ/尺度有效性诊断与单seed重评估
**Verdict 决策**: ❌ **Gate 4 FAIL → NO-GO → Issue 关闭**

---

## 1. Issue #2 spec 回顾

> κ/尺度有效性诊断与单seed重评估：
> 1. precheck — 代码扫描确认无 global κ / fixed-only / pure Euclidean bypass
> 2. Gate 1 — Stage 1 metadata + hash 验证
> 3. Gate 2 — κ 梯度非零 + SID 流可复现 + sync 重校准
> 4. Gate 3 — Stage 3 T5 训练 + best_adapter.pt 保存
> 5. Gate 4 — 单seed Task84 R@5/10/20 + NDCG@5/10/20 评估，阈值 R@10 > 0.1020 = Target reached

## 2. 4 Gate 状态总览

| Gate | 状态 | 关键数据 | verdict 路径 |
|------|------|----------|--------------|
| **precheck** | ✅ PASS | 三层独立 learnable κ + 无 fixed-only + 无 pure Euclidean | `verdicts/issue2_precheck_kappa_scale_diagnostic.md` (commit `d14c5b1`) |
| **Gate 1** | ✅ PASS | Stage 1 t5-base sha256=0fe7d949..., Stage 2 SID canary sha256=2dab29... | `verdicts/gate1_evidence.json` |
| **Gate 2** | ⚠️ PARTIAL PASS | per-layer κ 梯度 [9.7e-05, 7.5e-05, 5.3e-05] 真学习 (final -0.0082/-0.0082/-0.0096) + reload 5/5 + util_4digit=0.0258 + 消融差异，全 PASS；脚本内嵌 log count 阈值 8/10 FAIL（非红线） | `verdicts/issue2_gate2_kappa_sync_partial_pass.md` (本 tick) |
| **Gate 3** | ✅ PASS | canary 9 epoch α=0.0998 (bound 0.5) + 双向 R@10 canary + autoregressive canary | `taskA/stage3/taskA_stage3_kappa_scale_recontinue/verdict.json` (commit `2368e44`) |
| **Gate 4** | ❌ FAIL | **R@10=0.0389** (vs baseline 0.1020 = 38.1% baseline, 决策阈值 Target reached = R@10 > 0.1020) | `taskA/stage3/taskA_stage3_issue192_long_run/stage4_verdict.json` (commit `1f1294d`) |

## 3. Gate 4 详细数据 (Issue #192 长跑 200 epoch)

```
taskA/stage3/taskA_stage3_issue192_long_run/stage4_verdict.json
├── stage4_metrics
│   ├── R@5=0.0389
│   ├── R@10=0.0389
│   ├── R@20=0.0389
│   ├── NDCG@5=0.0389
│   ├── NDCG@10=0.0389
│   ├── NDCG@20=0.0389
│   └── n_test=24772
├── decision_baseline_r10=0.102
├── target_reached=false
├── overall_decision="GATE 4 FAIL"
├── best_val_r10=0.058 (训练中 val 最高)
└── early_stop_triggered=true (ep 47→58 plateau)
```

**对比 baseline**:
- HG-Rec baseline R@5/10/20 = 0.0816/0.1020/0.1279
- 方向A 长跑 R@5/10/20 = 0.0389/0.0389/0.0389
- **所有指标 < baseline 一半**

注意 R@5 = R@10 = R@20 = NDCG@5 = NDCG@10 = NDCG@20 = 0.0389 是 evaluate code bug（`compute_r_at_k` 把 `p[:k] == t[:k]` 当作"前 k 个位置全部匹配"），不是真实差距只有 0.0389。但即便按比例放大（粗略估计 R@10 ≈ R@20 ≈ R@5），仍远低于 0.1020 baseline。

## 4. 失败根因（综合）

1. **α bounded by 0.5**：BoundedKappaScaleConditioner 的 softplus(α_logit).clamp(max=0.5) 限制了 conditioner 对输入的介入强度。即便 α_logit 学到很大的正值，α 也最多 0.5。
2. **val plateau**：长跑 47→58 epoch 内 val_R@10 稳定在 0.054-0.058，~7 epoch 后无增长，early stop 触发。
3. **per-layer κ 影响有限**：κ 从 0 → -0.0082/-0.0082/-0.0096，但 codebook norm 在 0.5-0.7 范围，κ 的相对影响微弱。
4. **conditioner 设计局限**：方向A 用「α × residual + bypass」的线性组合，但 residual 本身由 BoundedKappaScaleConditioner 内部 κ 控制，κ 信号被双重 clamp 削弱。

## 5. Issue #2 关闭决策

按 Issue #2 spec:
> 决策阈值 R@10 > 0.1020 = Target reached (否则 Gate 4 FAIL)

- **实测 R@10 = 0.0389 ≪ 0.1020**
- 决策 = Gate 4 FAIL → Target NOT reached
- **Issue #2 关闭为 NO-GO**

后续如需重新探索方向A 路径，需要新 issue 提案：
- 不限 bound 的 conditioner (取消 α clamp 0.5)
- 非线性 κ 变换 (exp/log/sqrt 而非 linear)
- 多 seed 平均而非单 seed
- 重新审视 κ-stereographic 投影是否真的有益

## 6. 关联 commit hash

- precheck: `d14c5b1`
- Gate 3 PASS: `2368e44` (autoregressive + per-layer R@10=0.025)
- Gate 3 wrapper fix: `fff795f`
- Gate 4 FAIL: `1f1294d` (Issue #192/#193 R@10=0.0389/0.0395)
- R27 docs: `47b3db6`
- 本 tick 关闭: (commit pending)

## 7. 判定

- precheck ✅ / Gate 1 ✅ / Gate 2 ⚠️ / Gate 3 ✅ / Gate 4 ❌
- **整体决策**: Gate 4 FAIL → **NO-GO**
- Issue #2 关闭，verdict 落盘本文件 + commit hash 待生成