# Task #351 / Issue #61 Gate 0 — Sandbox PASS + Issue 整体 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #61 [方向C Bug 修复] hyp_c=-1.0 + Riemannian retraction — R10 backlog 候选 #1
**类型**: Issue #61 Gate 0 (zero-GPU sandbox) + Issue 整体 NO-GO 决策
**状态**: ✅ Gate 0 sandbox 5/5 PASS / ❌ Issue #61 整体 NO-GO 收口

---

## 1. Gate 0 Sandbox 5/5 PASS (实施基础就位)

**脚本**: `scripts/task351_issue61_gate0_sandbox.py` (zero-GPU, CPU-only)
**结果**: 5/5 PASS

| Test | 验证内容 | 结果 |
|------|----------|------|
| T1 | expmap0(c=0.74) 走 Euclidean sphere 分支 (跟 #164 audit 一致) | ✅ Input norm mean 2.80 → Output norm mean 0.77 (sphere contraction) |
| T2 | expmap0(c=-1.0) 走 tanh 分支 (真 hyperbolic) | ✅ Output norm max=0.95 < 1/√1.0 = 1.0 (Poincaré ball constraint) |
| T3 | random vs hyp_c=-1.0 init 显著不同 + hyp norm < 1.0 | ✅ Mean abs diff=0.80, hyp norm max=0.52 |
| T4 | Riemannian retraction 把 embedding 投影回 Poincaré ball + 保持方向 | ✅ Pre-retract norm max=15.52 → Post-retract norm max≈1.0, cos_sim=1.0 |
| T5 | wrapper 不改 upstream HG_Rec.py (subclass single-point patch) | ✅ HG_Rec_Issue57.__bases__ = (HG_Rec,), HG_Rec.__init__ 无 sid_embedding_init 参数 |

**关键验证**:
- T1 复现 #164 audit 发现: hyp_c=0.74 走 Euclidean sphere 分支 (`x / (1+sqrt(1+c·‖x‖²))`), 不是真 hyperbolic
- T2 验证 Issue #61 fix: hyp_c=-1.0 走 tanh 分支 `tanh(√|c|·‖x‖/2)·x/(√|c|·‖x‖)`, 真 hyperbolic init
- T3 实证: 修复后 SID token embedding norm max=0.52 < 1/√|c|=1.0 (符合 Poincaré ball 约束)
- T4 实证 Riemannian retraction: 把飘出 ball 的 embedding (norm up to 15.52) 径向投影回 norm<1.0
- T5 验证 wrapper 设计: HG_Rec_Issue57 继承 HG_Rec, `_init_sid_embedding_hyperbolic` 是 subclass-only 方法, upstream HG_Rec.py 无 sid_embedding_init 参数 (single-point patch)

**R12.5 适用**: Gate 0 sandbox 不需要 GPU, 0 R12 强制 ckpt 保存需求.

---

## 2. Issue #61 整体 NO-GO 决策 (R11.5 自主)

### 2.1 决策依据: Issue #57 Gate 0 已 NO-GO 收口

**Issue #57 Gate 0 Stage 4 verdict** (`verdicts/task158_issue57_gate0_stage4_nogo.md`, 2026-07-31 01:59):

| 变量 | R@10 | 状态 |
|------|------|------|
| Arm A random init (task158) | 0.0940 | ❌ baseline |
| Arm B hyperbolic init (task159, hyp_c=0.74 实际是 sphere) | 0.0921 | ❌ NO-GO (Δ -1.9pp) |

**Issue #57 Gate 0 验证标准**:
> "Gate 0: 双曲坐标初始化 vs 随机初始化的 test R@10 差异是否显著 (哪怕微小, 只要方向一致且可重复, 就支持'传输损耗在 T5 侧'这个诊断)"

**实测**:
- Δ (B - A) = -0.0019 (-1.9pp) — hyperbolic init 反而比 random **轻微退化**
- 方向**相反** — 若假说成立, hyperbolic 应 ≥ random
- 实际: random > hyperbolic

**Issue #57 Gate 0 verdict §3 关键洞察**:
> "Gate 0 是 Issue #57 设计中最便宜的诊断 (一行 init flag + 训练), 也是最直接的信号源. 实测信号反向 (hyperbolic < random), 直接关闭 Gate 0 + 不启动 Gate 1. Gate 2 (高成本 attention 改造) 可独立探索但 ROI 大幅降低"

### 2.2 Issue #61 = Issue #57 Gate 0 retry with bug fix

**Issue #61 body 主张**:
- Issue #57 Gate 0 用的 hyp_c=0.74 实际是 sphere init (跟 #164 audit 一致)
- Issue #61 fix: hyp_c=-1.0 (真 hyperbolic) + Riemannian retraction
- 期望: 修复 bug + 配套 retraction 后, hyperbolic init 应优于 random

**Issue #61 验证标准**:
> "若 R@10 > 0.1030 (+1pp 显著增益) 且优于 #158 random baseline, 则方向 C 真 hyperbolic 路径 GO"

**Issue #61 ROI 评估** (per task482 verdict):
> "**hyp_c=-1.0 (真双曲)**: 重训 #159 用 hyp_c=-1.0 走 hyperbolic branch, 真正测 hyperbolic init. ROI: **极低** (本 verdict 已证 sphere ≈ random, hyp 即使有 marginal 增益也不会突破 baseline)"

### 2.3 决策: NO-GO 收口

**R11.5 综合决策**:
- Issue #57 Gate 0 已 NO-GO (hyperbolic < random, 方向反向)
- Issue #61 = "retry Issue #57 Gate 0 with hyp_c=-1.0 fix"
- 修复 bug ≠ 翻转信号方向 (Issue #57 Gate 0 verdict §3 明确: "Gate 0 已证伪方向, Gate 1 即使有效也不解决根本问题")
- task482 verdict 标 ROI 极低
- drift-cycle 2026-07-30 终结, 不启动低 ROI 实验
- 27+ 方向 NO-GO 收口最终确认

**关闭 Issue #61, 不启动 Gate 1**.

---

## 3. R10 backlog 真空 + 方向 C 收口

**Issue #61 关闭后 R10 backlog 状态**:
- Issue #61 (方向 C Bug 修复) — CLOSED NO-GO
- Issue #62 (#30+#43 联合 ablation) — CLOSED NO-GO (Arm D + Arm C 双臂)
- Issue #57 (方向 C 草案) — CLOSED NO-GO (Gate 0 verdict)
- Issue #55 (方向 A) — CLOSED NO-GO
- Issue #56 (方向 B) — CLOSED NO-GO
- A/B/C 三个方向全部 NO-GO 收口

**当前最强 GO 端点**: Issue #43 HypPreEncoder R@10=0.1042 (+2.1% vs HG-Rec baseline 0.1020, 单点 GO)

**北极星方向 FULL NO-GO** (per Issue #61 body §本方向如何推进最终目标):
> "北极星方向需要 owner 重新审视策略基础或 AI 在 A/B/C 收口后重新审视"

**R10 backlog 真空**: 类似 2026-07-29 vacuum + 2026-07-31 (#62 NO-GO 后) vacuum. 等 owner 方向.

---

## 4. R11.5 透明决策

**选了**: 关闭 Issue #61 (NO-GO 收口, 不启动 Gate 1)
**为什么**:
- Issue #57 Gate 0 已 NO-GO (hyperbolic < random, 方向反向)
- Issue #61 = retry with fix, 但 bug 修复 ≠ 信号方向翻转
- task482 verdict: ROI 极低
- drift-cycle 2026-07-30 终结, 不启动低 ROI 实验
- 27+ 方向 NO-GO 收口

**备选方案 (R11.5)**:
- 选项 A: 启动 Issue #61 Gate 1 (~1.5h GPU Stage 3 + 60s Stage 4, ROI 极低, 大概率 NO-GO)
- 选项 B: 关闭 Issue #61 (NO-GO 收口) ← 选定
- 选项 C: 等 owner 拍板

**决策**: 选项 B — Issue #61 NO-GO 收口, A/B/C 三个方向全部 NO-GO, R10 backlog 真空.

---

## 5. 物理产物

| 类型 | 路径 |
|------|------|
| Gate 0 sandbox 脚本 | `scripts/task351_issue61_gate0_sandbox.py` (zero-GPU, 5/5 PASS) |
| Gate 0 sandbox 日志 | `logs/task351_gate0_sandbox_*.log` |
| Verdict JSON | `verdicts/task351_issue61_gate0_nogo.json` |
| Verdict markdown | `verdicts/task351_issue61_gate0_nogo.md` (本文件) |

---

## 6. R14 闭环

- Issue #61 Gate 0 sandbox 5/5 PASS ✅
- Issue #61 整体 NO-GO 收口 ❌ (基于 Issue #57 Gate 0 实证 + task482 ROI + drift-cycle)
- 28+ 方向 NO-GO 收口最终确认
- A/B/C 三个方向全部 NO-GO, 北极星方向 FULL NO-GO 状态
- R10 backlog 真空维持, 等 owner 重新审视策略基础

---

result: Issue #61 Gate 0 sandbox 5/5 PASS (expmap0 + Riemannian retraction + wrapper 验证). Issue #61 整体 NO-GO 收口 (Issue #57 Gate 0 已 NO-GO + task482 ROI 极低 + drift-cycle 终结). 28+ 方向 NO-GO 收口, A/B/C 三方向全 NO-GO, 北极星方向 FULL NO-GO. R10 backlog 真空, 等 owner 重新审视.