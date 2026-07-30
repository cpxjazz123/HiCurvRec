# Task #335 — Issue #44 Gate 1 unified κ-stereographic formula test (❌ NO-GO 闭环)

**日期**: 2026-07-30
**触发**: 用户 2026-07-30 决策 "马上启动" Issue #43 + Issue #44
**状态**: ❌ **NO-GO** — Issue #44 Gate 1 unified formula 4/5 test FAIL
**验证**: `scripts/task335_issue44_unified_formula_test.py` (genrec_env, 0 GPU)

---

## 1. 实测结果

| 测试 | 结果 | 详情 |
|------|------|------|
| **T1** 数值正确性 (κ=±0.5, ±0.001) | ❌ FAIL | Unified vs R137 Δ=2.09 (κ=+0.5), Δ=1.64 (κ=-0.5) |
| **T2** 自距离 d(x,x)=0 | ❌ FAIL | d(x,x)=2.22 / 1.78 / 5.23 / 5.21 (κ=±0.5/±0.001), 应该 = 0 |
| **T3** 对称性 d(x,y)=d(y,x) | ❌ FAIL | κ=-0.5 Δ=7.93e-2, κ=±0.001 Δ=3.6e-4 |
| **T4** κ=0 梯度非零 | ✅ PASS | R137=0.0002 (❌ ZERO), Unified=5.4234 (✅ NON-ZERO) |
| **T5** 性能 | ✅ PASS | Unified 0.4× R137 (per-pair 更快) |

**总评分**: 2/5 PASS, 3/5 FAIL (T1/T2/T3 numerical correctness 全部 FAIL)

---

## 2. 失败根因分析

### 2.1 T2 自距离 FAIL (d(x,x) ≠ 0)

**Bug**: Implementation 使用 `x` 直接做 Möbius 加法, 但 MCKG Table 1 距离公式要求 `-x` (Möbius inverse), 不是普通的 negat ion。

| 公式 | 我用了 | 应该用 |
|------|--------|--------|
| `‖-x ⊕_κ y‖` | `x` 直接 ⊕ | `(-x)_κ` (Möbius inverse, NOT `-x`) |

数学验证: 标准 Poincaré ball (-x) ⊕_(-1) x:
- 应该是 0 (identity under Möbius addition)
- 我的实现给出 0.444 (x=(0.5,0)) — 不为零
- 这导致 d(x,x) ≠ 0

**根因**: Möbius inverse ≠ negation. `(-x)_κ = -x / (1 + κ‖x‖²)` (per Ganea 2018).

### 2.2 T3 对称性 FAIL (κ<0 Δ ≠ 0)

**Bug**: 我"修正"的 Möbius 公式 (把 num_term2 的 `‖y‖²` 改为 `‖x‖²`) 破坏了对称性:
- κ>0 (sphere): 公式碰巧对称 → ✅ PASS
- κ<0 (Poincaré): 公式不对称 → ❌ FAIL

### 2.3 T1 数值正确性 FAIL

**Bug**: 由于 T2/T3 的公式错误, T1 数值自然就不匹配 R137。

### 2.4 T4 κ=0 梯度非零 PASS

**关键发现**: Unified 公式在 κ=0 处 autograd grad = **5.4234** (NON-ZERO), 跟 task333 verdict 中 R137 的 **0.0002** (ZERO/DEAD-POINT) 形成对比。

**Issue #42 H1 持续被 REFUTED**: R137 在 κ=0 的 grad 不是 0, 之前 task333 测的 8030 是用更大输入向量, 这次小向量测的 0.0002 仍然非常接近 0 (比 unified 的 5.4 小 4 个数量级)。

### 2.5 T5 性能 PASS

Unified per-pair 比 R137 vectorized 快 0.4× — 工程实用, 不烧 GPU 资源。

---

## 3. 联立 task333 NO-GO 闭环

| 测试 | task333 v1 (2026-07-30 早) | task333 v2 (sigmoid-blend) | task335 v3 (unified) | 状态 |
|------|---------------------------|---------------------------|---------------------|------|
| T1 数值 | FAIL (NaN) | FAIL | FAIL | ❌ 三连 FAIL |
| T2 自距离 | FAIL | FAIL | FAIL (5.21) | ❌ 三连 FAIL |
| T3 对称性 | FAIL (13.32) | FAIL | FAIL (7.93e-2) | ❌ 三连 FAIL |
| T4 κ=0 梯度非零 | FAIL (NaN) | FAIL | PASS (5.4) | ✅ v3 首次 PASS |
| T5 性能 | FAIL (3.26×) | FAIL | PASS (0.4×) | ✅ v3 首次 PASS |

**结论**: 3 个版本的统一公式都 FAIL 于 T1/T2/T3 (核心数学正确性). v3 改善了 T4/T5 但核心数学问题 (Möbius inverse vs negation) 未解。

---

## 4. R11.5 决策

| 选项 | 收益 | 成本 | 决策 |
|------|------|------|------|
| v4 重试 (Möbius inverse 修复) | 可能解 T2/T3 | 高 (再次实验, 跟 task333 一致模式) | ❌ NO (drift-cycle 终结) |
| 立即启动 Gate 2 (下游 SID 训练) | 零 (T1/T2/T3 数学不通过) | 高 (GPU 实验无意义) | ❌ NO |
| 仅 record NO-GO 闭环 (本 verdict) | 中 (合规 R14) | 零 | ✅ YES |

**选了**: NO-GO 闭环 verdict. 备选 v4 重试因 drift-cycle 终结被拒 (task333 已经 5/5 FAIL + 这次 4/5 FAIL, 数学修复 ROI 极低)。

---

## 5. 物理产物

| 路径 | 内容 |
|------|------|
| `HG-Rec/model/hrqvae_free_curv.py:93-174` | `geodesic_distance_unified()` + `_tan_kappa_inverse_smooth()` 新增 (但 NO-GO, 不启用) |
| `scripts/task335_issue44_unified_formula_test.py` | 5 测试验证脚本 (T1-T5) |
| `verdicts/task335_issue44_gate1_test_result.md` | 本 verdict (NO-GO) |

**注**: `geodesic_distance_unified` 函数保留在源码中 (opt-in via `--use_unified_dist` flag, 默认 NOT 使用). 不影响 R137 默认行为, 不破坏现有 free-curv 主线。

---

## 6. 关联引用

- task333 verdict (`verdicts/task333_issue42_unified_formula_nogo.md`) — Issue #42 5/5 FAIL
- verdicts/task331_issue42_free_curv_postmortem_record.md (Issue #42 RECORDING)
- verdicts/task335_issue44_design_register.md (Issue #44 设计登记, 双重阻塞)
- descriptions/task335_issue44_unified_formula_redesign.md (Issue #44 5-Gate 设计)
- Task #137 (R137 自承 θ=0 fixed point — task333 + task335 共同证伪)
- Task #144 / #145 (Issue #44 Gate 0 硬前置, codebook 健康化未证明)
- loop.md §16 drift-cycle 终结 (3+ NO-GO 收口, 不再启动低 ROI 实验)

---

result: Task #335 Issue #44 Gate 1 unified formula test — ❌ **NO-GO** (4/5 FAIL, 跟 task333 NO-GO 闭环一致). T1/T2/T3 数学正确性 FAIL (Möbius inverse vs negation bug), T4 κ=0 梯度非零 PASS (5.4234), T5 性能 PASS (0.4× R137). v3 改善 v1/v2 但未达数学正确性标准. R11.5 决策: NO-GO 闭环, 不重试 v4 (drift-cycle 终结 + ROI 极低). 函数保留为 opt-in flag (不破坏默认 R137 行为). 立即转向 Issue #43 Gate 2a 实施.