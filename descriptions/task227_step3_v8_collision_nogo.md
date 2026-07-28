# Task #227 — Step 3 v8 collision chase: 架构级 NO-GO

> **任务目的**: 在 Task #226 完成 5 条件 (agreement / utilization / cos_std / radius-only / no-NaN) 全部 PASS 后, 追加第 6 条件: **tuple collision rate ≤ 12% (对齐 HG-Rec Task #84 baseline 9.07% 的复现范围)**. 测了 v7 (Sinkhorn sk_eps=0.03 平滑) 和 v8 (去掉 w_div + batch=1024 + β=1.0 + 200 epoch).

> **完成日期**: 2026-07-27
> **状态**: 🟡 进行中 → 🔴 **架构级 NO-GO**

## 1. 背景

M-arm 走的是 product_manifold 架构 (2D hyp + 32D euc 拆分). Task #226 验证 5 条件时没覆盖 collision — 仅看几何性. 现在要在已 PASS 的几何约束上, **追加** 一条标准下游 SID collision rate ≤ 12%.

Task #84 baseline 在 vanilla 32D Poincaré 跑 50 epoch 自然收敛到 9.07% (epoch 49), 8.31% (epoch 54). K=[64,128,256] 完全同款.

## 2. 实验轨迹

### v6 (Task #226 已 PASS 5 条件配置)
- 配置: β=0.5, batch=256, --w_div 100, --w_angular 10, --product_manifold, sk_eps=0
- 5 条件 PASS: agreement < 0.90, util ≥ 0.97, cos_std = 0.74, radius-only < 2%, max(c·‖e_k‖²) < 0.22
- **tuple collision: 95.68%** ❌ (整 (c1,c2,c3) unique tuples = 411/9922)

### v7 (在 v6 上加 Sinkhorn 平滑 sk_eps=0.03)
- 配置: 同 v6 + --sk_epsilons 0.03 0.03 0.03 + --sk_iters 50
- 5 条件仍 PASS
- **tuple collision: 97.28%** ❌ (比 v6 更差, smooth distribution 没让 encoder 解锁)
- 整 unique tuples = 270/9922 (只有 270 个不同 SID 组合)

### v8 (去掉 --w_div, batch 256→1024, β 0.5→1.0, epochs 50→200)
- 配置: 5 条件 knobs 全保 (LR_LOG_R, anti_collapse, w_angular, normcap, r_target, r_spread, γ_norm)
- 假设: baseline β=1.0 + batch=1024 是收敛关键
- 实测: trainer-side collision 单调爬升 78.75% (epoch 4) → 88.99% (9) → 96.59% (29) → 99.89% (54) → 99.84% (89)
- 跟 baseline 9% 完全逆向. 200 epoch 内必不会 < 12%
- **tuple collision (推断): 100%** ❌

## 3. 架构层面根因

Task #84 baseline (vanilla 32D Poincaré) 跟 M-arm (2D hyp + 32D euc) 的关键差异:

| 配置 | Task #84 baseline | M-arm v6/v7/v8 |
|------|-----|-----|
| 嵌入维度 | 单 32D | 2D hyp + 32D euc |
| distance 公式 | 全 32D `d_poincare(z, c_k)` | `α·d_poincare(z_h, c_h_k) + β_radial·‖z_e - c_e_k‖²` |
| β | 1.0 | 1.0 (v8) |
| batch | 1024 | 256 (v6/v7) / 1024 (v8) |
| 收敛 | epoch 54 → 8.31% | stuck 95-100% |
| train recon_loss | ~8.6 (epoch 49) | ~13-15 (epoch 49, 卡死) |

**结构性差异**: K=64 in 2D 角度 + 半径空间很难区分. 即便 cos_std=0.74, K=64 (angle, radius) 对在 2D ball 边际 (r_tgt=2.0, 面积 ∝ ρ²·sinh²(√c·ρ) ≈ 14.5 sq) → 间距 √(14.5/64) ≈ 0.48, 跟 baseline 32D 的间距 √(32/K_avg) 完全不可比. 32D 给 K=64 的角分辨率是 16× 富余, 2D 的角带宽根本不够.

**两份数据正面顶撞**:
- baseline vanilla: 在 32D 单空间里 Sinkhorn 自然把 9922 items 分给 64 个聚类
- M-arm product: 在 2D hyp 子空间里, Sinkhorn 推不动 K=64 items 分到 64 个不同码字. encoder 找不到方向, encoder 输出被锁在小角度扇区.

加上 5-cond knobs (normcap 把 euc 钉 ‖e‖=0.3, γ_norm=5 把 euc norm 钉 [1.0,1.35,1.7], cos_std pen 把 hyp 方向推散) 进一步缩小可动空间, encoder 没有自由度去找碰撞解.

## 4. 决定性差异点

**product_manifold 是 NO-GO 在 collision, 不是在几何**. 5 几何条件全 PASS 表明 M-arm 双曲 + 欧氏拆分**确实激活了几何效应** (radius spread, cos_std, agreement < 0.90), 但**它不能解 collision**. 这是设计空间的取舍:

- product_manifold: 激活几何 ✅, 解 collision ❌
- vanilla 32D: 不激活可拆分的几何, 但 9% collision ✅

如果目标是 R@10 下游指标, vanilla 32D 赢 (低 collision = T5 可学出更多 SID 组合). 如果目标是"几何可解释性 + 拆开编码", product_manifold 赢, 但代价是 collision 失控.

## 5. 工程产物

- v6 launcher: `scripts/m_arm_step3_50ep_wdiv100_wangular.sh`
- v7 launcher: `scripts/m_arm_step3_50ep_sinkhorn_wangular.sh`
- v8 launcher: `scripts/m_arm_step3_200ep_bs1024_b10_nowdiv.sh` (PID 2704028, killed at epoch 89)
- collision 测量: `scripts/m_arm_collision_rate.py` (standard 下游 SID)
- v6 训练产物: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular/`
- v7 训练产物: `products/m_arm/m_radius_spread_step3_50ep_sinkhorn_wangular/`
- v8 训练产物: `products/m_arm/m_radius_spread_step3_200ep_bs1024_b10_nowdiv/`

## 6. 后续可能方向 (跟本任务正交)

如需让 product_manifold collision 进 HG-Rec 复现范围, **必须先攻架构假设**而不是 recipe:

| 方向 | 思路 | 风险 |
|------|------|------|
| A. **aug angular_dim** | 2D → 4D 或 8D hyp 子空间, 角带宽↑, K=64/128/256 更稀 | 3-4× 更多 hyp 量, 跟 euc 拆分的语义减弱 |
| B. **改径向编码** | euc branch 32D → 16D + 16D extra hyp 总 4D hyp | euc 分支信息丢失, 可能伤下游 R@10 |
| C. **接受 NO-GO** | 结论定 model 不走 product_manifold collision 优化路径 | 失去 5 条件 PASS 的几何可解释性 |
| D. **加宽码字量** | K layer 0 从 64 → 128, 实际有效 cluster 涨 | token 4 变 5, T5 stage 3 pipeline reshape |

本任务暂不做方向 A/B/D, 等用户对碰撞优先级决策.

## 7. 收口结果 (2026-07-27 v9 + v10 增量实验)

**v9 + v10 实际验证后**: 即便加宽 angular_dim=4, wall 仍在. v10 (4D + v6 recipe) 起步 epoch 4 collision 27.95% 是 5 variants 最低, 但 50 epoch 后撞 97.06% — 跟 v6/v7/v8/v9 终态一致. 进一步推论:
- **angular_dim 不是 collision 瓶颈** (2D/4D 都撞墙)
- **recipe 维度穷尽** (β/batch/sk_eps/w_div/w_angular/epochs 7 维都试过)
- **bottleneck = product_manifold × 5-cond knobs 乘法效应**

**唯一能让 collision ≤ 12% 的路径 = 方向 D (回退 vanilla 32D) 或方向 E (κ-Stereographic)**. 推荐方向 D, 理由: 已验证 collision 9% (Task #84 baseline), 实施风险最小.

完整 verdict: `verdicts/task227_step3_v8_collision_nogo_result.md`.
