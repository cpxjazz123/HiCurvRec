# Task #457 / Issue #164 Gate 0-3 深层码字靠近双曲边界判别间隔 + SID 翻转率零训练实证 — NO-GO 收口

## 任务摘要

Issue #164 owner 2026-08-01 派发: 验证深层码字靠近双曲边界是否通过放大最近-次近码字判别间隔 Δ = d_2 - d_1 提高量化稳定性。本任务为 **零训练实证** (frozen Task #84 HRQ-VAE ckpt, CPU 计算), Gate 0/1/2/3 完整跑完, 结论 NO-GO.

## 锚定

- **ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth` (SHA256 24d25501ac40b9757e52c22ee1bd1579f145d769c7ec02e2eb0bf0b68314c61e)
- **数据集**: Musical_Instruments (Amazon, 9922 items, 768→32 e_dim)
- **baseline**: Task #84 R@10=0.1020, c=1, num_emb=[64,128,256], beta=1.0

## R18 4 维度对比 (vs 历史 codebook 坍缩 / 径向位置 NO-GO 任务)

| 维度 | Task #457 (#164 Gate 0-3) | 历史 (Issue #30 #32 #28 #153 #154) |
|------|---------------------------|--------------------------------------|
| **D1 spec** | **零训练径向干预** (frozen ckpt) + 只改径向半径 | 训练端到端 Stage 1 RQ-VAE 重训 |
| **D2 实施** | **径向缩放 T_s(x) = exp_0^c(s·log_0^c(x))** 同时改 residual + codeword 半径 | codebook transform r_l/s_l / κ-decouple / bilateral quota |
| **D3 Gate 1 失败机制** | 量化判别间隔 Δ 跟 R@K 解耦 (Issue #84/#178 历史已证) | mode collapse / collision / codebook utilization 坍缩 |
| **D4 引用** | 边界距离放大效应 (Poincaré ball 边界几何) | per-layer Codebook Transforms / Hungarian / Truncated Simplex |

→ **4 维度全部不一致**, R18 强制零训练实验 (零 GPU 训练成本).

## Gate 0 (径向对象 profile)

逐层报告 ρ_l^code / ρ_l^path / ρ_l^res 三对象 × 7 统计量 (mean / std / p5 / p50 / p95 / min / max).

**关键结论**:
- ρ_l^code↓ (深层码字靠近原点) 跟 ρ_l^path↑ (累计路径节点远离原点) **同时成立** — 验证 Issue #164 §1 区分两类 code 的合理性
- ρ_l^res↓ 跟 ρ_l^code↓ 同趋势 (residual 跟 codeword 都向原点收缩, 这是 Phase 0 mode collapse 共享机制)

产物: `products/task457_issue164_gate0_3_radial/gate0_radial_profile.json`

## Gate 1 (零训练径向干预)

s ∈ {0.6, 0.8, 1.0, 1.2, 1.4} 同时缩放 residual + codeword, 测 Δ_l = d_2 - d_1 margin.

**关键数据** (margin_mean):
- s=1.0 → s=1.4 双曲 margin 单调上升 (Layer 0/1/2 全部 PASS H1)

产物: `products/task457_issue164_gate0_3_radial/gate1_radial_margin.json`

## Gate 2 (扰动 FlipRate)

σ ∈ {0.001, 0.005, 0.01} 测 FlipRate_l^(s).

**关键数据** (FlipRate 跨 s=1.0 → s=1.4):
- Layer 0: σ=0.001 1.22% → 2.22% (FAIL), σ=0.005 3.86% → 3.51% (PASS), σ=0.01 7.80% → 5.63% (PASS)
- Layer 1: σ=0.001 3.11% → 2.22% (PASS), σ=0.005 16.38% → 12.01% (PASS), σ=0.01 31.25% → 23.09% (PASS)
- Layer 2: σ=0.001 7.40% → 5.23% (PASS), σ=0.005 32.81% → 24.98% (PASS), σ=0.01 57.02% → 43.95% (PASS)

**Layer 0 σ=0.001 反例** (s=1.4 FlipRate 2.22% > s=1.0 1.22%) — 弱扰动 + 大 K (256) 时深层 margin 单调性不能让 FlipRate 单调下降。

**H2 整体判据**: 大部分 layer × σ 组合 PASS, 但 Layer 0 σ=0.001 失败, **H2 = FAIL** (不一致)。

产物: `products/task457_issue164_gate0_3_radial/gate2_fliprate.json`

## Gate 3 (欧氏距离对照)

欧氏 margin 在 s=1.0 → s=1.4 同样上升:
- Layer 0: 0.0460 → 0.0644 (+40%)
- Layer 1: 0.0112 → 0.0157 (+40%)
- Layer 2: 0.0049 → 0.0068 (+39%)

**H3 = FAIL**: 欧氏 margin 也改善 ~40%, 跟双曲 ~同样幅度。改善是普通向量缩放效应, **不能归因双曲边界几何特有效应**。

产物: `products/task457_issue164_gate0_3_radial/gate3_euclidean_comparison.json`

## Gate 0-3 整体决策

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **H1 margin 单调** | ✅ PASS | s=1.4 双曲 margin > s=1.0 (Layer 0/1/2 全 PASS) |
| **H2 FlipRate 下降** | ❌ FAIL | Layer 0 σ=0.001 反例 (1.22% → 2.22%), 不一致 |
| **H3 欧氏对照无改善** | ❌ FAIL | 欧氏 margin 同样改善 ~40% (普通向量缩放效应) |

**Issue #164 spec Gate 0-3 收口 = NO-GO** (需 H1 + H2 + H3 同时满足)。

## Gate 4 不启动

按 Issue #164 spec: "只有机制得到支持后，才进入 Gate 4 训练级验证" — Gate 0-3 不支持, **Gate 4 (Arm A/B/C/D 训练级) 不启动**。

## 关键决策点 (R11.5 自主决策)

1. **零训练实证 (Gate 0-3) vs 训练级 (Gate 4)**: Issue #164 spec 强制 Gate 0-3 先零训练, 避免无谓 GPU 训练。决策 = 跑 Gate 0-3, Gate 4 条件启动。
2. **径向缩放函数**: T_s(x) = exp_0^c(s · log_0^c(x)) (Poincaré ball 标准径向缩放, R11.2 兜底 #2 上游 framework 默认)
3. **τ 阈值**: 0.01 (Issue #164 spec 提到 "τ 运行前固定不允许事后选择", 选 0.01 是合理经验值)
4. **H2 判据严格度**: 任一 (layer, σ) 组合 FAIL → H2 整体 FAIL (per Issue #164 "需 H1/H2/H3 同时满足")。
5. **H3 反证严格度**: 欧氏 margin 上升 > 5% 阈值 → H3 FAIL (避免噪声误判, R11.5 兜底)。

## 实证产物 (R12 强制 ckpt + R15 push + R16 close)

1. `gate0_radial_profile.json` (2470 bytes) — 三对象 × 三层 × 七统计量
2. `gate1_radial_margin.json` (4265 bytes) — s × 三层 × 五统计量
3. `gate2_fliprate.json` (6429 bytes) — (s, σ) × 三层 FlipRate
4. `gate3_euclidean_comparison.json` (7728 bytes) — 双曲 vs 欧氏 margin + FlipRate
5. `summary.json` (639 bytes) — Gate 0-3 整体决策 NO-GO
6. `ckpt_sha256.txt` (Task #84 ckpt SHA256 锚定)

## 后续 (R10 v2 idle 允许)

Issue #164 Gate 0-3 NO-GO 收口, 跟历史 Issue #30 #32 #28 #153 #154 跨方向 NO-GO 累积共同锁死 baseline recipe 内部 R@10 杠杆已穷尽结论。后续架构层方向 (R@10 > 0.1020 真杠杆) 需 owner 派工新 issue。