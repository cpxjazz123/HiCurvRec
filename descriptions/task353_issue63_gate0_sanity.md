# Task #353 / Issue #63 Gate 0 — zero-GPU sanity 数值预检

**日期**: 2026-07-31
**前置**: Gate -1 8/8 PASS (verdict task352_issue63_gate_minus1_result.md)
**任务**: Gate 0 zero-GPU 数值 sanity (distance scale / codebook score / assignment entropy / NaN/Inf)
**基线**: HG-Rec Task #84 Test R@10=0.1020
**决策**: PASS → 进 Gate 1 (Stage 1 训练); FAIL → NO-GO 收口

---

## 1. Gate 0 spec (per Issue #63 body)

Gate 0 = 跟 #47 同标准 sanity. 检查:
1. **distance scale**: κ-Stereographic distance 数值范围 (应 ~[0, +∞), 无 NaN/Inf)
2. **codebook score**: per-layer codebook utilization 健康 (无 collapse)
3. **assignment entropy**: per-layer 分配熵 (entropy ≈ log(K) 健康)
4. **NaN/Inf**: 所有中间 tensor 无 NaN/Inf
5. **fixed vs learnable κ ablation**: κ=fixed=0 (欧氏退化) vs κ=learnable 输出对比

## 2. 测试设计 (zero-GPU, ~5s)

| Test | 检查 |
|------|------|
| T1 | κ=0 退化: κ-Stereo distance ≡ Euclidean (L2²/2) |
| T2 | κ=2 max: distance 数值饱和 (~π/√c ≈ 2.22), 不爆炸 |
| T3 | Distance 矩阵对称性 (d(x,y) = d(y,x)) |
| T4 | Distance triangle inequality (抽样验证) |
| T5 | Assignment entropy: random init 后, per-layer entropy ≥ log(K)/2 |
| T6 | No NaN/Inf in codebook / forward / distances |
| T7 | Forward reconstruction loss 数值合理 (~10-50 range) |
| T8 | Codebook utilization ≥ 50% for random batch (sanity) |

任一 FAIL → NO-GO 收口.

## 3. 后续 Gate

- Gate 1: Stage 1 训练 L0/L1/L2 utilization ≥ 90%, collision 跟 #55/#56 对比
- Gate 2: Stage 2 SID unique/collision/utilization 健康
- Gate 3: Stage 3 T5 + Stage 4 R@10 eval, R@10 > 0.1020 GO

## 4. R11.5 决策

- Gate 0 沿用 #47 同标准 (用户已批过), 不需要 user-confirm
- 决策阈值: 8/8 PASS 才进 Gate 1

---

result: Gate 0 spec — 8 项 zero-GPU 数值 sanity 测试. PASS → Gate 1 Stage 1 训练, FAIL → NO-GO.