# Task #457 / Issue #164 [方向D Gate0-3] 深层码字靠近双曲边界判别间隔 + SID 翻转率零训练实证

## 任务背景

Issue #164 2026-08-01 owner 派发. Issue spec 强制:
- **Gate 0**: 区分三种径向对象 (单层 codeword 半径 / 累计路径节点半径 / residual 半径) 逐层 profile
- **Gate 1**: 零训练径向干预 s ∈ {0.6, 0.8, 1.0, 1.2, 1.4} 同时缩放 residual + codeword, 测 Δ_l = d_2 - d_1 margin
- **Gate 2**: 扰动 σ ∈ {0.001, 0.005, 0.01} 测 SID FlipRate
- **Gate 3**: 欧氏距离对照, 排除普通向量缩放效应
- **Gate 4 (条件启动)**: 仅 Gate 1-3 支持假设后才启动 Arm A/B/C/D 训练级验证
- 锚定 Task #84 健康 HRQ-VAE ckpt (c=1 baseline), 不重训 Gate 0-3
- 锚定 Musical_Instruments (9922 items)

## R18 4 维度对比 (vs 历史 codebook / κ-Stereo / radial 任务)

| 维度 | 历史任务 (Issue #30 #32 #28 #153 #154 #155 #156) | Task #457 (#164 Gate 0-3) |
|------|-----------------------------------------------|---------------------------|
| **D1 spec** | 训练端到端 (Stage 1 RQ-VAE) + Sinkhorn + T5 完整 R@K | **零训练径向干预** (frozen ckpt) + 只改径向半径 + 不重训 Stage 1 |
| **D2 实施** | codebook transform r_l/s_l / κ-decouple / bilateral quota / clamp saturation | **径向缩放 T_s(x) = exp_0^c(s·log_0^c(x))** 同时改 residual + codeword 半径 |
| **D3 Gate 1 失败机制** | mode collapse / collision / codebook utilization 坍缩 | **量化判别间隔 Δ 跟 R@K 解耦**: 仅 Δ 改善 ≠ R@10 改善 (Issue #84 / #178 历史已证) |
| **D4 引用** | per-layer Codebook Transforms / Hungarian / Truncated Simplex | 边界距离放大效应 (Poincaré ball 边界几何, 跟 κ-Stereographic 不同) |

→ **4 维度全部不一致**, R18 强制零训练实验 (零 GPU 训练成本, 仅 ckpt 加载 + 径向变换 + margin/FripRate 测量).

## 实施目标 (Issue #164 spec Gate 0-3)

### Gate 0 (零训练, ckpt 加载即可)
- 加载 Task #84 baseline ckpt (`products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`)
- 加载 Musical_Instruments train+test 数据集
- 前向跑 Stage 1 RQ-VAE (freeze weights), 收集三层 residual + codeword + 累计路径节点
- 逐层报告 ρ_l^code / ρ_l^path / ρ_l^res 的 mean / std / p5 / p50 / p95 / min / max
- Gate 0 产出: `gate0_radial_profile.json` (三层 × 三对象 × 7 统计量 = 63 个数)

### Gate 1 (零训练径向干预)
- 固定 ckpt + 数据 + 码字编号 + 角度结构 + 曲率 (c=1 baseline)
- 对每层同时变换: r_l^(s) = T_s(r_l), e_{l,j}^(s) = T_s(e_{l,j})
- s ∈ {0.6, 0.8, 1.0, 1.2, 1.4} 5 档
- 测 Δ_l^(s) = d_2^(s) - d_1^(s), 逐层报告 margin mean / p5 / p50 / p95 / 小 margin 样本比例 (τ = 0.01 固定)
- Gate 1 产出: `gate1_radial_margin.json` (3 层 × 5 s × 5 统计量)

### Gate 2 (扰动 FlipRate)
- σ ∈ {0.001, 0.005, 0.01} 三档扰动
- 对每个 (s, σ) 组合测 FlipRate_l^(s)
- Gate 2 产出: `gate2_fliprate.json` (3 层 × 5 s × 3 σ = 45 个 FlipRate)

### Gate 3 (欧氏距离对照)
- 对相同坐标计算 d_E margin + FlipRate
- 比较双曲 vs 欧氏: Gate 3 产出: `gate3_euclidean_comparison.json`

### Gate 4 (条件启动)
- **仅在 Gate 1-3 同时支持 H1 + H2 时启动**
- Arm A 自由学习 (baseline 复现)
- Arm B 三层相同目标半径 τ_0=τ_1=τ_2=0.5
- Arm C 逐层向边界 τ_0=0.3 < τ_1=0.5 < τ_2=0.7 (目标方案)
- Arm D 逐层向原点 τ_0=0.7 > τ_1=0.5 > τ_2=0.3 (反向)
- 200 epoch Stage 1 full + Sinkhorn + Stage 3 T5 + Stage 4 R@K 六项

## Gate 0-3 决策阈值

- **GO**: Gate 1 Δ^(s=1.4) > Δ^(s=1.0) (margin 单调上升) AND Gate 2 FlipRate^(s=1.4) < FlipRate^(s=1.0) AND Gate 3 欧氏未现同等改善
- **NO-GO**: 任一不满足
- **Gate 4 启动**: 仅 GO 时启动

## 8 件套

1. `gate0_radial_profile.json` — 三对象 × 三层 × 七统计量
2. `gate1_radial_margin.json` — s × 三层 × 五统计量
3. `gate2_fliprate.json` — (s, σ) × 三层 FlipRate
4. `gate3_euclidean_comparison.json` — 双曲 vs 欧氏 margin + FlipRate
5. `verdict_gate0_3.md` — Gate 0-3 整体 GO/NO-GO 决策 + Gate 4 启动建议
6. `radial_intervention_log.txt` — 完整执行 log
7. `ckpt_sha256.txt` — 加载的 Task #84 ckpt SHA256 校验
8. `gate4_recommendation.md` (条件) — Arm A/B/C/D 详细启动 spec

## 防错机制

- **R7 GPU**: Gate 0-3 零训练, GPU 不占用 (CPU 即可, 仅 ckpt 加载 + 前向 + 矩阵运算)
- **R12 ckpt**: Gate 0-3 不存 ckpt (frozen ckpt), 仅 Gate 4 才需 R12
- **R15 push**: Gate 0-3 verdict 写完后立即 commit + push
- **R20 4 Gate 详细**: commit message + Issue #164 comment 每个 Gate ≥3-5 行
- **R21 v2**: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
- **R16 close**: Gate 0-3 verdict 写完后立即 R20+R21 v2+R16 闭环
- **R137**: Gate 0-3 零 GPU 训练无需 TRITON_CACHE_DIR (Gate 4 才需 per-task 隔离)

## R11.5 自主决策

- **Gate 0-3 立即启动**: 零训练实证, 几乎零成本 (CPU 几小时)
- **径向缩放函数**: T_s(x) = exp_0^c(s·log_0^c(x)) (Poincaré ball 标准径向缩放, R11.2 兜底 #2 上游 framework 默认)
- **τ 阈值**: 0.01 (Issue #164 spec 提到 "τ 运行前固定不允许事后选择", 选 0.01 是合理经验值, R11.5 兜底)
- **Gate 4 启动条件**: Gate 1-3 同时支持 H1 + H2 才启动 (R11.5 不允许跳过 Gate 4 启动门槛)
