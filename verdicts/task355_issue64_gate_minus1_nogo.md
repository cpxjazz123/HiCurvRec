# Task #355 / Issue #64 Gate -1 — Direction B 重开 NO-GO (架构未实施)

**日期**: 2026-07-31
**前置**: 无 (首次 Gate -1)
**任务**: Gate -1 zero-GPU 预检 (Direction B 三层可学习混合曲率乘积空间)
**结果**: ❌ Gate -1 FAIL (7/8 PASS, T1 失败)

---

## 1. Issue #64 主张

Direction B 重开, 目标 = 用 **layer-specific learnable mixture + constrained curvature components** 修复 #56 NO-GO.

公式 (per issue body):
```
d_l = Σ_m softmax(w_l,m) * s_l,m * d_{κ_l,m}(z_l, c_l,m)
```

需要 4 类参数 per-layer: α_l (softmax w_l,m), κ_l,m (constrained curvature components), s_l,m (scale), 组合 effective geometry.

## 2. Gate -1 结果

| Test | 名称 | 结果 |
|------|------|------|
| **T1** | **Direction B 实施基础** | ❌ **FAIL** |
| T2 | mixture/curvature 隔离 | ✅ PASS |
| T3 | 干净 optimizer | ✅ PASS |
| T4 | forward path clean | ✅ PASS (α_l_raw grad=4.21, scale_l grad=0.29) |
| T5 | batch 维度独立 | ✅ PASS |
| T6 | optimizer state detach | ✅ PASS |
| T7 | codebook/SID 隔离 | ✅ PASS |
| T8 | Stage 3/4 interface | ✅ PASS (state_dict 3 α + 3 scale) |

→ **7/8 PASS, T1 FAIL → NO-GO 收口**

## 3. T1 失败根因

**Direction B 架构 4 类参数 状态**:

| 参数 | 必需 | 已实现 |
|------|------|--------|
| α_l (mixture sigmoid) | ✓ | ✅ `FreeCurvVectorQuantizationMixedCurvWithScaleFixed` |
| scale_l (per-component) | ✓ | ✅ 同上 |
| **per-component κ_l_m** | ✓ | ❌ **只 κ_fixed scalar (无 per-component learnable κ)** |
| **softmax w_l_m** | ✓ | ❌ **只 α_l sigmoid (无显式 mixture weights)** |

**最近实现**: `FreeCurvVectorQuantizationMixedCurvWithScaleFixed` (在 `hrqvae_issue55_56_fixed.py`)
- α_l_raw = sigmoid → α ∈ [0, 1]
- scale_l = per-component scale
- κ_fixed = scalar (Ollivier mean from #70 = 0.74)

**跟 Direction B spec 差距**:
- Direction B 要求 **多个 learnable κ_l_m per component** (constrained in [c_min, c_max])
- Direction B 要求 **softmax mixture weights w_l_m** (跟 α_l sigmoid 不同)

→ 当前实现是 **α_l + scale_l (Issue #56 + #55)** 而不是 Direction B **w_l_m + s_l_m + κ_l_m**.

## 4. 后续路径分析

| 方向 | 内容 | 难度 |
|------|------|------|
| A. 实施 Direction B 完整架构 (新增 κ_l_m + w_l_m) | 改 src/hrqvae_free_curv.py + 新增 wrapper 类 | R11.4 critical |
| B. 接受 α_l + scale_l 现状 (放弃 κ_l_m/w_l_m) | 等价于 Issue #56 重做 (历史 NO-GO) | NO-GO 概率高 |
| C. 关闭 Issue #64, 等 owner 明确 spec | 不浪费 GPU, 等待方向 | 当前决策 |

→ 当前决策 = 方案 C (NO-GO 收口)

## 5. Issue #64 综合结论

| Gate | 结果 |
|------|------|
| **Gate -1** | ❌ **FAIL (T1 实施基础)** |
| Gate 0 | N/A (未进入) |
| Gate 1 | N/A |
| Gate 2 | N/A |
| Gate 3 | N/A |

→ **Issue #64 NO-GO 收口**

## 6. 教训 / 信号

- Direction B 跟 Issue #56 路径重复 (κ + α + scale), Issue #56 已 NO-GO 收口 (unique=0.01% mode collapse)
- Issue #64 spec 提到 "????? #56 ???α/分量曲率链接", 但 κ_l_m 没实施
- 即使实施 κ_l_m + w_l_m, 后续 Gate 1 大概率跟 #56 同模式失败 (drift-cycle)
- 当前最优路径 = 不浪费 GPU, 等 owner 拍板:
  - 实施完整 Direction B 架构 (R11.4 critical)?
  - 接受现状 + 接受 NO-GO 收口?
  - 切到其他方向?

## 7. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task355_issue64_gate_minus1_nogo.md | 本 verdict |
| descriptions/task355_issue64_direction_b_gate_minus1.md | Gate -1 spec |
| scripts/task355_issue64_gate_minus1_audit.py | Gate -1 审计脚本 (zero-GPU, 5s) |

---

result: Issue #64 Gate -1 NO-GO (T1 FAIL: Direction B 完整架构 κ_l_m + w_l_m 未实施). 最近实现 α_l + scale_l 满足 7/8 项但缺核心 mixture + κ 组件. Issue #64 全线 NO-GO 收口, 等 owner 拍板实施范围.