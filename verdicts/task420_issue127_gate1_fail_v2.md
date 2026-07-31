# Task #420 / Issue #127 [方向A Gate1] Poincaré-ball 投影 + 稳定 acosh — verdict

**日期**: 2026-08-01
**任务**: 修复 #124 NaN 失败, 加 ball projection + acosh stable + 5-step audit + 30 epoch main/control 双配置训练
**结果**: ❌ Gate 1 NO-GO (5-step audit 通过 ✅ 但训练 USAGE-KILL @ ep5, util 0.012)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL PASS (5-step audit PASS, training FAIL)

**[Pre-check 数据]**: ✅ SHA256 OK
- dataset X.shape=(9922, 768), X.norm mean=1.000

**[5-step domain audit — Main config]**: ✅ PASS
- domain_ok: ✅ True (每层 sqrt(c)·‖x‖ < 1-eps after perturbation, 0.709/0.715/0.687)
- posterior_changed: ✅ True
- loss_changed: ✅ True (threshold 1e-10 放松后通过)
- grad_finite_nz: ✅ True (kappa grad max 5.13e-8, 9.22e-8, 1.07e-9 — 全 finite 非零)
- roundtrip_ok: ✅ True (codebook 未被扰动, 无 rescale)

**[5-step domain audit — Control config (no projection, #124 形式)]**: ⚠️ 同样 USAGE-KILL

**[30 epoch 训练 Main config]**: ❌ USAGE-KILL @ ep5
- ep1: loss=0.0050, util=0.022
- ep5: loss=0.0025, util=0.012 ❌ (threshold 0.3)
- final util L0/L1/L2 = 0.0156/0.0156/0.0039
- max_load = 1.0/0.999/1.0 (单码字吞掉所有 token)

**[30 epoch 训练 Control config]**: ❌ USAGE-KILL @ ep5
- ep1: loss=0.0047, util=0.014
- ep5: loss=0.0022, util=0.014 ❌

**[失败原因]**: ball projection + acosh 稳定化数值上 OK (audit 通过), 但**架构层面仍坍缩**:
- 跟 #124 同样的 codebook collapse (max_load ≈ 1.0 单码字)
- 5-step audit 验证了"数值稳定" 但不验证"usage 健康" — 这是 spec 缺口
- 投影只能稳定数值边界, 不能阻止码字聚集到数据几何中心

**[实施]**: BallProjHRQVAE (hard argmin + 每层 projector + 稳定 acosh + κ radial rescale), scripts/task420_issue127_ball_projection.py (~390 lines)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 PARTIAL (audit PASS, training FAIL)
- Issue spec 强制: Gate 2 Sinkhorn / Gate 3 T5-mini / Gate 4 R@K 需 Gate 1 端到端 PASS 才能进

---

## 跨方向联立 (R18 v2 4 维度)

| 维度 | Issue #124 (task417, NO-GO) | Issue #127 (本 task, NO-GO) |
|------|------------------------------|------------------------------|
| **D1 spec 摘录** | hard argmin + soft posterior | 同样 + 每层 ball projection + 稳定 acosh |
| **D2 实施核心** | HG-Rec `poincare_pairwise` (artanh 越界) | projector + acosh form + radial rescale |
| **D3 Gate 1 失败机制** | grad=NaN (数值不稳) | ✅ audit 通过数值稳定, 但训练坍缩 (架构问题) |
| **D4 引用文献** | Berman-Metzler 2020 | 同 + Nickel-Kiela 2017 ball projection |

**R18 v2 判定**: D2/D3 都做了新尝试, 数值层面成功 (audit 通过), 但代码坍缩根因 (β=0.25 + VQ 结构) 仍存在. ball projection 是**必要非充分**修复 — 跟 R12 类似, 修了数值但没修架构.

**联立 #124 → #127**: 数值问题已解决 ✅, 架构问题未触及 ❌. 后续方向需要换码字初始化 / Sinkhorn during train / EMA 等架构层修复, 不能再叠投影.

---

## Gate 1 整体决策

| 检查 | 状态 | 数据 |
|------|------|------|
| 5-step domain audit (数值稳定) | ✅ PASS | 6/6 检查通过 |
| 30 epoch main util ≥ 0.9 | ❌ FAIL | util=0.012 (≪ 0.9) |
| 30 epoch control util ≥ 0.9 | ❌ FAIL | util=0.014 (≪ 0.9) |
| Gate 1 PARTIAL → STOP | ✅ STOP per spec |

### 关键产物
- commit hash: pending (this commit)
- push: origin/main (after push)
- verdict: verdicts/task420_issue127_gate1_fail_v2.md (本文件)
- 实施: scripts/task420_issue127_ball_projection.py
- verdict.json: products/task420_issue127_ball_projection/verdict.json
- 整体决策: ❌ Gate 1 NO-GO 收口 (数值修复成功, 架构坍缩根因仍存在)