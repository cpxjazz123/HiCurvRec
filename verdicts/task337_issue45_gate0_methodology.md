# Task #337 — Issue #45 Gate 0 — 测量口径复核 (✅ CONFIRMED, 但多方需修)

**日期**: 2026-07-30
**状态**: ✅ **CONFIRMED — 结论更细粒度化**
**核心交付**: `scripts/task337_issue45_gate0_methodology_check.py` (三测 + 真实代码路径)

---

## Gate 0 结果总览

| 测试 | 结果 | 含义 |
|------|------|------|
| **T1** 静态审计 | ✅ PASS | 源码 + Task #333 重实现都含 `clamp(min=1e-8)` |
| **T2** 复现+追踪 | ✅ PASS | 输入 κ=0, 内部 κ_abs=**1.00e-8**, 测到的 8030 是 1e-8 的梯度 |
| **T3** 无 clamp 对照 | ✅ PASS | 精确 κ=0 无 clamp → **NaN** (aratan(0/0)), 8030 确实来自 clamp |
| **T4** 生产代码路径 | 🔶 重要发现 | **当前代码 κ=0 处 grad=3e-7 (非 None, 非 8030), 50 epoch θ drift=0.0093** |

---

## T4 生产代码路径 — 关键发现 (Issue #45 双方需注意)

在生产代码 `_per_component_dist_sq` (已用 R137 统一公式替代旧的 .item() 硬分支) 上:

| Layer | θ_m.grad norm | 解读 |
|-------|--------------|------|
| L0 | **3.15e-07** | 极微小但非零 |
| L1 | **3.39e-08** | 比 8030 小 11 个数量级 |
| L2 | **4.17e-33** | 完全为零 |

50 epoch 后 θ drift: `0.0 → [-0.0088, -0.0093, -0.0093]` (max move=0.0093)

**这意味着**:
- Task #135 的 "θ_m.grad = **None**, 50 epoch θ 精确 = 0.0" **仅适用于旧版代码** (有 .item() 硬分支). 当前代码重构后 part of the bug is fixed.
- Task #333 的 "R137 κ=0 grad=**8030**" 是在 **合成数据 (50×8 float64)** 上测的 clamp=1e-8 点, 不是 κ=0, 且放大 4-5 个数量级
- **真实梯度 ~3e-7**: 在 κ=0 处有小梯度, 50 epoch 产生的 θ drift=0.0093 (κ drift ≈ 2·tanh(0.0093) ≈ 0.0186)
- 换算到 1000 epoch (baseline Task #89): 假设线性增长, κ_drift ≈ 0.37 — 足以轻微离开 κ=0

---

## Gate 0 综合结论

### H0 (Issue #45 主张) — ✅ CONFIRMED 但有附加条件

| H0 子句 | 实判 | 证据 |
|---------|------|------|
| "Task #333 测的是 clamp=1e-8, 不是 κ=0" | ✅ 严格 CONFIRMED | T2 直接追踪到 κ_abs=1e-8 |
| "8030 不代表真实训练梯度" | ✅ 严格 CONFIRMED | T4 实测 3e-7 vs 8030 (11 个数量级差) |

### H1 (反面假设) — ❌ 不完全成立 (因为中间有 refactor)

| 子句 | 实判 |
|------|------|
| "Task #135 的 50-epoch 冻结在 0.0 是更可信的真实场景" | ⚠️ **仅对旧版代码成立**. 当前代码已不再精确冻结 |
| "R137 refactor 后 θ=0 仍有小梯度 (3e-7), 够在长训中移动" | ✅ 1000 epoch 可产生 ~0.37 κ drift |

### 核心矛盾解决

| 源 | 原声称 | 当前 verdict |
|----|--------|-------------|
| Task #333 (§3) | "R137 at κ=0 grad=8030, not dead point" | ❌ **8030 是 clamp=1e-8 在合成数据上的值, 非生产环境** |
| Task #135 (旧版) | "κ=0 grad=None, 精确冻结" | ❌ **仅适用于旧版 .item() 硬分支, 当前代码已无此问题** |
| Issue #45 H0 | "8030 是测量口径问题" | ✅ **CONFIRMED (T1+T2+T3)** |
| 当前 truth | R137 + clamp=1e-8 + 真实数据 → grad ≈ **3e-7** | **并非死点, 但梯度极小 (11 orders 低于 8030)** |

---

## 对相关 Issue / verdict 的影响

| 上游 | 之前结论 | 修正 |
|------|---------|------|
| Task #333 | "R137 κ=0 dead-point 假设 REFUTED" | ⚠️ **需要 caveat**: 8030 是 clamp=1e-8 在合成数据的值, 生产环境 ≈ 3e-7. "REFUTED" 是夸大 |
| Task #135 | "grad=None, 50 epoch 精确=0.0" | ⚠️ **需 retro**: 旧版 .item() 硬分支的发现. 当前代码已部分修复 |
| Issue #42 | "论证链断裂" | ⚠️ **部分恢复**: R137 在 κ=0 处不是严格死点, 但 3e-7 的梯度在 50 epoch 只产生 ~0.009 drift, 仍符合 "近似死点" |
| Task #137 | "R137 fix" verdict | 该 fix (去掉 .item() 硬分支 + 统一 R137 公式) **实际上有效**, 只是 fix 后梯度量级极小 |
| Issue #45 | "review proposal" | ⚠️ 需要 Owner 决策: 是否要把 "R137 κ=0 真死点" 改为 "近似死点 (3e-7)" |

---

## R11.3 自主决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| T4 工作负载 | ✅ 50 epoch, `sk_eps=[0.003,0.003,0.003]` (匹配 #135) | 足够检测 θ drift, 不过度 GPU (CPU 15 min) |
| "反证推倒"判定 | ✅ **不推翻 Task #333 结论, 但添加 caveat** | 8030 是 clamp 点, 不是 κ=0 |

---

## 产物清单

- `scripts/task337_issue45_gate0_methodology_check.py` (四个测试)
- `verdicts/task337_issue45_gate0_methodology.json` (原始数值)
- `verdicts/task337_issue45_gate0_methodology.md` (本文件)

---

result: Issue #45 Gate 0 — H0 CONFIRMED (T1+T2+T3), T4 意外揭示当前 R137 生产代码在 κ=0 处有 3e-7 微小梯度 (不是 Task #333 的 8030, 也不是 Task #135 的 None). 所有相关 verdict (Task #333, Task #135, Issue #42) 需添加 caveat.
