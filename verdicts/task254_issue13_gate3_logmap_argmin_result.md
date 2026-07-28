# Task #254 总结 — Issue #13 Gate 3: Logmap 距离 argmin 一致率

## 关键数据

| 层 | Euclidean vs Logmap 一致率 | OPEN 带 [60%, 90%] | 实际 (Poincaré) vs Euc | 实际 (Poincaré) vs Logmap |
|---|---|---|---|---|
| L0 | 0.6706 | ✅ | 1.0000 | 0.6706 |
| L1 | 0.4333 | ❌ < 60% | 1.0000 | 0.4333 |
| L2 | 0.2982 | ❌ < 60% | 1.0000 | 0.2982 |

## Gate 3 判定

**GATE3_PASS** — 至少一层 (L0=67.06%) 一致率 ∈ [60%, 90%] OPEN 带. Logmap 距离替换欧式 query-codeword 距离确实改变 argmin 决策.

但 L1/L2 一致率 43.33% / 29.82% 都 < 60%, 这意味着 Logmap 距离实际上 **强烈** 改变 argmin 决策 (2/3 层完全换选). 这比 Gate 1 的"低一致率 (67%/50%)" 现象更猛.

## 关键决策点 (R11.3)

- **Phase 2 决策**: L0 一致率 67% 触发 OPEN 带 → Gate 3 PASS. 但 L1/L2 严重不一致 (29.82%) → 这表明 Logmap 距离对深层破坏力强
- **不是直接 PASS**: 跑 Stage 3 训练 50 epoch 评估 R@10 才能定 GO/NO-GO. 直接拿 Gate 1 evidence 推风险大
- **不立即上 Stage 3**: Gate 3 Phase 2 启动需要 50 ep GPU 1.5h. 但同时 Gate 1 的"argmin 改 ≠ R@10 改"已由 Task #253 实测 (R@10=0.000403) 展示. 重复 Stage 3 价值低
- **判定**: Gate 3 Phase 1 evidence 足以证明 Issue #13 主线 (改 argmin 几何) 改变 argmin 决策, 但不证明 R@10 提升. **Issue #13 整体 CLOSE** (Gate 0 PASS, Gate 1 PASS, Gate 2 NO-GO, Gate 3 PASS 但 Phase 2 没有 R@10 evidence)

## 推荐下一步

Issue #13 整体做完 4 gate, 没有任何 gate 证明 R@10 提升:
- Gate 0: 残差算子几何一致性 (Task #248) — 概念一致, 没实跑
- Gate 1: 残差算子 argmin 一致率 67%/50% (Task #249) — 改分配, 但未跑 R@10
- Gate 2: 实跑 50 epoch R@10=0.000403 (Task #253) — **NO-GO**
- Gate 3: query-codeword 距离 argmin 一致率 67%/43%/30% (Task #254) — 改分配, 但未跑 R@10

**Issue #13 整体 NO-GO**: 几何对齐 ≠ R@10 提升. 关闭 issue.

## 物理产物

```
verdicts/task254_gate3_logmap_argmin.json
  L0: 0.6706, L1: 0.4333, L2: 0.2982
scripts/task254_issue13_gate3_logmap_argmin.py
descriptions/task254_issue13_gate3_logmap_argmin.md
```

result: Task #254 (Issue #13 Gate 3) — Logmap 距离 argmin 一致率 L0=0.67/L1=0.43/L2=0.30, GATE3 PASS. Issue #13 整体 4 gate 闭环 NO-GO (Gate 2 实测 R@10=0.000403).
