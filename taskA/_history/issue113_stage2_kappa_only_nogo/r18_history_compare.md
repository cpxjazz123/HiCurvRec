# Issue #113 R18 4 维度对比详细分析 (2026-08-10)

## R18 规则 (用户强约束)

> "新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任一不同 → 必须实验, 禁'路径同构' NO-GO."

## 历史 κ 学习路径全清单

| Issue | commit | 路径 | 实施机制 | Gate 1 结果 |
|-------|--------|------|---------|------------|
| #55 / v4 fix_c | - | 冻结 c=1 + κ 学到极值 | `fix_c=True` 单层 κ 学过激 | NO-GO (κ 负漂移) |
| #59 bounded κ | - | σ 形式 + 边界占用 | `MLR_ENTROPY_TARGET_RATIO` Stage1 残差头+Stage2 不兼容 | NO-GO (util_3 < 0.85) |
| #96 / #157 | `6ae239f` | v15 capmatch baseline | KAPPA_ANCHORS + REL_STRUCT + κ EMA + RAD_SAFE + rescale | **PASS** (基线, R@10=0.1057) |
| #224 CPL | `912a5f5` | Stage3 端 c_perturb | Dbar 静态扰动, Stage3 不重新学 | Gate 1/3 PASS, Gate 4 N/A |
| #225 v2 | - | per-item-conditioned κ | `kappa_peritem_mlp` per-layer | NO-GO (40× 梯度爆炸) |
| #228 | `cd4815e` | per-batch scalar mod | `get_c_with_batch_norm()` + α_l sigmoid | NO-GO (L2 256→61, -76%) |
| #235 redux | `e7b1fb1` | 同 #228 flag | 同 #228 | NO-GO (κ uniform, 50ep) |
| **#113 (阉割)** | - | **删所有抗塌缩机制, 仅 κ 学习** | **KAPPA_ANCHORS + REL_STRUCT + CURV_PRIOR + κ EMA** | **NO-GO (util_4digit=0.0258, -97.4%)** |

## R18 4 维度对比 (vs 历史)

### D1 spec 摘录

| Issue | D1 spec |
|-------|---------|
| #55 fix_c | "Stage2 κ 学到极值, 用 fix_c=True 冻结 c=1 阻止" |
| #59 bounded κ | "σ 形式让 κ 有界, 但 Stage1 残差 Lorentz 头不兼容" |
| #96 v15 capmatch | "per-layer κ anchor + drift + REL_STRUCT target 反解" |
| #224 CPL | "Stage3 Dbar 静态 + c_perturb_raw 训练时扰动" |
| #225 v2 | "per-item-conditioned κ target (item_radius 调制)" |
| #228 | "per-batch scalar mod (batch_norm × κ_base)" |
| #235 redux | "同 #228 flag 路径 (误用 --enable_per_batch_radius_mod)" |
| **#113 (阉割)** | **"仅保留 κ 学习路径, 删 MLR/REVIVE/REC_LOSS/CDR/FIXED_CURV/MCJT/SPBI/RAD_SAFE/RESCALE/VANILLA_RQ/PER_BATCH_RADIUS_MOD/UTIL_REVIVE"** |

### D2 实施核心

| Issue | D2 实施 |
|-------|---------|
| #55 | `fix_c=True` 冻结 c=1, κ 学到极值 |
| #59 | Stage1 残差 Lorentz 头 + Stage2 MLR τ 退火 |
| #96 | `KAPPA_ANCHORS=[0.30, 1.79, 1.48]`, `REL_STRUCT` 反解 RHO_BALL_TARGET, RAD_SAFE, codebook rescale |
| #224 | Stage3 `c_perturb_raw` scalar, `Dbar_l` precompute 时混入 |
| #225 v2 | `kappa_peritem_mlp = nn.Sequential(Linear(1,1), Tanh())` per-layer, 接收 `item_radii` |
| #228 | `get_c_with_batch_norm()` + `α_l ∈ [-0.5, 0.5]` sigmoid bound, per-batch scalar |
| #235 redux | 同 #228 flag (`--enable_per_batch_radius_mod`) |
| **#113 (阉割)** | **删 12 个机制 (`HyperbolicHyperplaneMLR` 304 行 + `compute_rec_loss` 120 行 + `codebook_diversity_loss` 30 行 + main() MLR 块 120 行 + epoch loop MLR/REVIVE/τ-recalib 150 行 + precheck MLR 400 行 + train_step 4 分支 50 行 + 顶部常量 150 行), 保留 `KappaAwareVectorQuantization` 父类 (κ EMA + drift) + `KappaAwareHRQVAE` (vq_layers 改父类实例, 走硬 argmin) + CURV_PRIOR + REL_STRUCT** |

### D3 Gate 1 失败机制

| Issue | D3 Gate 1 失败机制 |
|-------|-------------------|
| #55 | κ 负漂移 (drift 不收敛) |
| #59 | Stage1 Stage2 不兼容, util_3 < 0.85 |
| #96 | **PASS** (基线, util_4digit=1.0) |
| #224 | Gate 4 N/A (训练中) |
| #225 v2 | κ 40× 梯度爆炸 (R23 wrapper broken) |
| #228 | L2 256→61 unique codes (-76%), per-layer κ 趋同 [0.496, 0.498, 0.498] |
| #235 redux | κ 50ep uniform (Δ<0.02), 与 #228 同 |
| **#113 (阉割)** | **util_4digit=0.0258 (-97.4%), κ ceiling saturation + 几何过度扭曲 + positive feedback loop** |

### D4 引用文献

| Issue | D4 引用 |
|-------|--------|
| #96 v15 capmatch | arXiv:2405.13979 (学习曲率与双曲尺度同步) |
| #224 CPL | 同 + arXiv:2006.02510 (Lorentz 曲率扰动) |
| #225 v2 | arXiv:2405.13979 + arXiv:2106.05381 (per-item radius) |
| #228 | arXiv:2405.13979 |
| **#113 (阉割)** | **arXiv:2405.13979** (沿用) |

## 维度差异总结

**#113 与历史 NO-GO 的关键差异**:

1. **D1 spec (反向删除 vs 加法)**:
   - #225/#228/#235 = "加新机制让 κ 更激进 learnable"
   - #113 = "删 12 个机制, 仅 κ 学习" — **完全反向**

2. **D2 实施 (无 anchor 引导)**:
   - #96 v15 capmatch = KAPPA_ANCHORS 手工锚点 + RAD_SAFE + rescale (多重软引导)
   - #113 = 保留 KAPPA_ANCHORS 但删除 RAD_SAFE + rescale (失去"软引导")

3. **D3 失败机制 (新机制 vs 老机制)**:
   - #228 = per-batch scalar 破坏 per-layer 异质
   - #225 v2 = per-item 信号过强 → 梯度爆炸
   - **#113 = κ ceiling saturation (drift→上限) + 几何扭曲 (√c→∞ 压扁测地距离) + positive feedback loop**

4. **D4 引用 (相同)**:
   - 全部沿用 arXiv:2405.13979, 文献层同构

**R18 判定**: D3 失败机制不同 → 必须实验验证 (已实验, NO-GO 确认).

## R37 决策线 (强制)

**#113 比 v15 capmatch baseline (Issue #96) 差 -97.4% util_4digit**, 实质 test_R@10 << 0.1057 ceiling:
- **决策**: 回退至 v15 capmatch baseline 重新创新
- **回退目标**: Issue #96 v15 capmatch (final_cs=[1.35, 6.00, 4.39], κ=[0.30, 1.79, 1.48])
- **阉割版仅留作记录** (verdict + run1 产物归档), 不作为下一版本起点

## 下一版本起点 (R37+R36 合规)

**R36 强制**: "禁止通过调参形式提升指标, 必须通过改善曲率框架"
**R37 强制**: "新版本必须以当前最优版本为唯一基础"

| 方向 | 机制 | R36 合规 | 风险 |
|------|------|---------|------|
| **A: CPL Stage3 端** | 不动 Stage2, Stage3 加 c_perturb + Dbar 重算 | ✓ 完全合规 | 历史 Issue #224 Gate 4 N/A |
| **B: inverse REL_STRUCT** | ρ_ball 偏离 TARGET 时调 κ (几何反馈) | ✓ 机制变更, 非 sweep | sign_aware_factor 调度需论证 |
| **C: κ 范围衰减** | epoch-dependent KAPPA_ANCHOR_RANGE | 边界 (可能被视为调参) | 需 coswarmup 而非线性 |
| **D: anchor 重设计** | 改 KAPPA_ANCHORS 数值 | ✗ 强 R36 违反 | 禁止 |

**推荐**: 方向 A (CPL Stage3 端) — 风险最小, 路径最清晰, 不破坏 v15 capmatch 健康 Stage2 ckpt.