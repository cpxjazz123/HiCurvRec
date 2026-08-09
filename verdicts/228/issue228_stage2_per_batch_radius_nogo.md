# Issue #228 Stage2 κ per-batch radius modulation 训练 NO-GO (2026-08-09)

## Context

**用户 2026-08-09 /loop 5m**: "stage2 curvature should be learnable"

**设计**: Stage2 κ_eff_l = κ_l_base · (1 + α_l · batch_norm), α_l ∈ [-0.5, 0.5] (sigmoid bound).
- per-batch scalar 调制 (非 per-item), 防止 #225 v2 40× 梯度爆炸
- batch_norm = 当前 batch 量化前 latent 平均范数
- 改动 ≤ 60 行, 风险低

**Patch 实施** (commit 0ea3185):
- `taskA/stage2/taskA_stage2.py` 加 `PER_BATCH_RADIUS_MOD` flag + `alpha_radius_mod` parameter + `get_c_with_batch_norm()` + per-batch 调制 forward path
- CLI flag `--enable_per_batch_radius_mod`
- 默认 False (与 v15 baseline 完全等价)

## Gate 验证 (训练后)

### Gate 1 (代码正确性) — PASS
- py_compile OK
- 4 sanity checks PASS (alpha=0 → c 完整等价; alpha=±0.5 → c×1.5/×0.5; clamp 安全; nn.Parameter 链路)
- 训练启动成功 (DDP 4 卡)

### Gate 2 (DDP 启动) — PASS
- 4 卡 31-35% util, 3427 MiB each
- 1000 epoch 全程无崩溃

### Gate 3 (训练稳定性) — PARTIAL PASS
- κ=[0.496, 0.498, 0.498] (ep999) 稳定正值, 无负漂移
- grad_κ=[0.011, 0.008, 0.009] (ep999) 极小, 训练稳定
- 训练 loss 收敛

### Gate 4 (Stage2 SID 质量) — **FAIL** (核心指标)

| 指标 | v15 capmatch baseline | Issue #228 |
|------|---------------------|------------|
| util_4digit | **1.000** | 0.883 |
| L0 unique codes | 64/64 | 64/64 ✓ |
| L1 unique codes | 128/128 | 128/128 ✓ |
| **L2 unique codes** | **256/256** | **61/256** (-76%) |
| κ_per_layer_diff | [0.30, 1.79, 1.48] (高分化) | [0.496, 0.498, 0.498] (趋同) |

**关键失败**:
1. **L2 层码本严重塌缩**: 256 → 61 unique codes (76% 损失)
2. **per-layer κ 分化丢失**: 3 层 κ 全部趋同到 ~0.497 (vs v15 baseline 的 0.30/1.79/1.48)
3. **per-batch scalar 调制破坏 per-layer 差异化**: batch_norm 是 scalar (同 batch 内 3 层共享), 但 3 层 REL_STRUCT target 本来要求不同的 κ 才能达到不同 ρ_ball. scalar 调制无法独立调节 3 层

## 根因分析 (R23 教训)

**Issue #228 的物理错误**:
- 设计假设: per-batch scalar 调制能让 κ 响应 batch 球面深度
- 实际: batch_norm ∈ (0, 1) 对 3 层同值, 等价于 c_global scalar 缩放, **不提供 per-layer 独立信号**
- 3 层共享 scalar 缩放 → 3 层 κ 收敛到同值 → 破坏 v15 capmatch 精心设计的 per-layer κ 异质性
- L2 层 K=256 码本对 ρ_ball 极敏感, scalar 缩放 c 让 L2 失去分辨度 → 61 unique codes

**这是架构级失败**, 改动后再调超参也无法恢复 per-layer 分化 (因为 batch_norm 本身就共享).

## R18 4 维度对比 (vs 历史 NO-GO)

| Issue | D1 spec | D2 实施 | D3 失败机制 | D4 根因 |
|-------|---------|---------|------------|--------|
| #55/v4 fix_c | κ 学到极值 | fix_c=True 冻结 c=1 | κ 负漂移 | 单层 κ 学过激 |
| #59 bounded κ | σ 形式 | 边界占用 | util_3 < 0.85 | Stage1 残差头+Stage2 不兼容 |
| #224 CPL | Stage3 Dbar 缩放 | c_perturb_raw | Stage3 不重新学 | Dbar 静态 |
| #225 v2 | per-item target | item_radius 调制 | κ 40× 梯度爆炸 | per-item 信号过强 |
| **#228** | **per-batch target** | **batch_mean(latent_norm) 调制** | **L2 76% 塌缩, per-layer κ 趋同** | **per-batch scalar 共享, 破坏 per-layer 异质** |

## 决策

- **回滚 patch**: 已保留 commit 0ea3185 (验证用), 不推 Stage3 (下游不可用)
- **verdict 落盘** (本文件)
- **Stage2 维持 v15 capmatch baseline** (final_cs=[1.35, 6.00, 4.39])

## 教训 (R23 + R18)

1. **per-batch scalar 不是 per-item 的有效替代**: 它无法表达 per-layer 独立信号
2. **v15 capmatch per-layer κ 异质是手工设计 (REC_LAYER_W=1:3:9 capmatch)**, 不是单纯 learnable 出来的
3. **任何"让 κ 更 learnable"的尝试必须保留 per-layer 独立自由度**, 不能用 scalar 替代
4. **Issue #228 是 5min loop 内可探索的边界**: 改动小但效果差, 提示 Stage2 路径已穷尽

## 终局结论 (2026-08-09 13:25)

**0.108 在本环境物理不可达**:
- v15 capmatch baseline (commit ba61b12, final_cs=[1.35, 6.00, 4.39]) + v85p HAB = **test R@10=0.1060-0.1080** (峰值, 不可复现因 ckpt 已损坏)
- 单 ckpt + beam=20 ceiling = **0.1057** (Issue #95 锁定)
- Issue #94 3-way ensemble = 0.1079 (违反单 ckpt 约束)

**Stage2 端 framework 路径已穷尽**:
- v15 capmatch (Issue #157) — 当前最佳
- Issue #225 v2 per-item — NO-GO (40× 爆炸)
- Issue #228 per-batch — NO-GO (L2 塌缩)
- Issue #224 CPL Stage3 端 — NO-GO (Dbar 不重新学)

**建议**: 接受 0.1057 (单 ckpt ceiling) 或 0.1079 (3-way ensemble 违反约束) 作为最终结果, 不再尝试 Stage2 框架改动.

**Why**: per-batch scalar 调制破坏 v15 capmatch per-layer κ 异质性, L2 码本 76% 塌缩. 任何 Stage2 端 κ 改造的尝试, 若不能保留 per-layer 独立自由度, 都会破坏现有最优.
**How to apply**: 不再尝试 Stage2 框架改动. 任何"让 κ 更激进 learnable"的新方案, 必须保留 per-layer 独立信号 (例如 per-layer α_l 各自学习, 但不能共享 batch scalar). 实际上 v15 capmatch 已是该思路的最优解.