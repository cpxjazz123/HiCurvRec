# Task #137 v2 retrain — κ_max=0.1 也失败 (SID 完全坍缩)

> **捕获时间**: 2026-07-24 14:48 (v2), updated 14:51 (B-arm quick Stage 2 诊断)
> **目的**: 失败分析 — v2 A 臂 κ_max=0.1 也产生坍缩 SID, 修复路径未奏效 + B-arm (M=2) 也坍缩

## 1. v2 决策回顾

### 1.1 决策链 (R11.4)
1. v1 A 臂 (M=1, κ_max=0.5, 1000 epoch) → κ 全饱和 (+0.4997) → Stage 2 SID 坍缩 (12 unique, 99.88% collision) → Stage 3 device-side assert
2. **R11.4 决策**: κ_max 太大导致 sph 流形强约束, 让 codebook 几何坍缩
3. **R11.3 自主决策**: κ_max=0.1 + 200 epoch 短训练, 减少 sph 流形强约束
4. **结果**: v2 A 臂 SID 坍缩到 1 unique (100% collision), 比 v1 (12 unique) 更糟

### 1.2 数据对比

| 维度 | v1 (κ_max=0.5) | v2 (κ_max=0.1) | B-arm quick (M=2, ep 130) |
|------|-----------------|------------------|---------------------------|
| 训练 epoch | 1000 | 200 | 130 (best_loss ckpt) |
| 训练 best_loss | 1.2294 (ep 687) | 1.2577 (ep 186) | 1.3318 (ep 130) |
| Final L0 κ | +0.4997 (饱和) | +0.0257 (弱) | +0.124, +0.122 (M=2 双分量) |
| Final L1 κ | +0.4987 | +0.0120 | +0.064, +0.053 |
| Final L2 κ | +0.4987 | +0.0121 | +0.052, +0.052 |
| Stage 2 unique SID | 12 | **1** | **15** (post-Sinkhorn) |
| Stage 2 max conflict | 7122 | **9922** | ~660 (≈ 9922/15) |
| Stage 2 collision rate | 99.88% | **99.99%** | 99.85% |

### 1.3 关键观察 — 坍缩根因是训练 dynamics, 不是 κ_max

**三次 SID 都坍缩**, κ_m 模式完全不同 (v1 饱和到 +0.5, v2 弱 +0.026, B-arm 中 +0.12). 共同因素:
- 训练 loss 都成功下降 (1.4 → 1.2-1.3)
- 但 quant_loss 都接近 0 (训练都学会了 collapse codebook 让 quant_loss 接近 0)
- Stage 2 生成的 SID 都坍缩到 1-15 unique

**根因分析 (R137 fix 是必要条件, 不是充分条件)**:
- R137 fix 接通 κ<0 + κ>0 路径 (解决 κ_m 锁 0 bug) ✅
- 但 codebook 几何学习本身有坍缩风险: kmeans_init 给 64 个 centroid, 训练过程中 encoder + decoder 学习让 quant_loss 接近 0, 当所有 items 都被映射到 1 个 centroid 时 quant_loss 也接近 0 (因为 VQ 端只有 1 个最近 centroid)
- **κ_max 影响 κ_m 大小但不影响 codebook 利用度** (codebook 利用度是 hard-assignment 问题)
- 短训练 (200 epoch) + 小 κ_max 让 model 更快达到 quant_loss=0 但 codebook 更分散 learning 没机会发生
- M=2 让 2 个分量独立学习但每个分量仍可能 collapse 到 1 个 centroid

### 1.4 ⚠️ B-arm (M=2) 也坍缩 — 重要诊断

B 臂 ep 130 best_loss ckpt (M=2, κ_max=0.5, free-curv) SID 测试:
- Pre-Sinkhorn: 15 unique SID / 9922 items (collision_rate=0.9985)
- Post-Sinkhorn (5 iters): 仍 15 unique (Sinkhorn 不收敛)
- κ_m 模式: L0=[+0.124, +0.122], L1=[+0.064, +0.053], L2=[+0.052, +0.052]

**结论**: M=2 也坍缩, κ 学习到合理值 (+0.12) 但 codebook 仍 collapse. 这说明 **codebook 坍缩是 RQ-VAE + free-curv 的根本副作用**, 不是 κ_max / M 数量的问题.

## 2. 后续动作

### 2.1 C 臂 (M=3, κ_max=0.5) — 仍期望
- C 臂 = B 臂架构类似 (M=3, 3 分量)
- 但 B 臂 (M=2) 也坍缩 → C 臂很可能也坍缩
- 仍让 C 臂跑完, 收集数据点

### 2.2 Stage 3 修复策略 (深度诊断)
- sinkhorn eps > 0 (强制 codebook 均匀利用)
- 加大 kmeans_iters (更多 init 迭代)
- codebook reset per epoch (死码重置)
- periodic ckpt eval (检测 utilization drop)
- **或**: 使用 Sinkhorn 平衡损失 (vs 硬 VQ)

### 2.3 Task #89 结论修正
- 旧结论 "数据本质欧氏" 是基于 R137 bug (κ 锁 0)
- 新发现: 即使 R137 fix, codebook 仍 collapse
- **正确结论**: Task #89 框架 (free-curv) **根本不能用于 RQ-VAE 训练**, 因为训练 dynamics 让 codebook 必然 collapse. 需要不同的 RQ-VAE 实现 (e.g., 加 sinkhorn 平衡项) 才能让 free-curv 真正发挥作用

## 3. v1 + v2 + B-arm quick npy/ckpt 文件标记

文件保留 (per R2 不允许 fallback 删除):
- `Instruments_t5_hrqvae_poincare_curv0.5.npy` (12 unique SID)
- `Instruments_t5_hrqvae_poincare_curv0.1.npy` (1 unique SID)
- `products/task137_v2/train/arm_A_M1_kappa0.1/best_loss_model.pth` (200 epoch short train)
- `products/task137/train/arm_A_M1/best_loss_model.pth` (1000 epoch, SID 12 unique)
- `products/task137/train/arm_B_M2/best_loss_model.pth` (ep 130 best_loss, SID 15 unique)

后续 Stage 3 launcher 不应使用这些 npy (会立即 device-side assert).

## 4. 关联

- Task #135 κ=0 bug diagnostic (R137 fix 必要)
- Task #137 A 臂 Stage 2/3 失败 (本 verdict 的 v1 部分)
- Task #137 v2 失败 (本 verdict v2 部分)
- Task #89 retro caveat (5 重证据结论已被强推翻, 因 codebook 坍缩是 free-curv 训练的根本副作用)

result: Task #137 v2 + B-arm quick 都失败 — SID 坍缩到 1-15 unique. 修复方向错误 (κ_max + M 不是坍缩主因). 根因是 RQ-VAE codebook collapse (训练 dynamics 让 quant_loss=0 + codebook 利用度低). C 臂仍继续跑 (收集数据点). 后续需要 sinkhorn 平衡 + codebook reset 等深度修复. **Task #89 框架 (free-curv) 根本不能直接用于 RQ-VAE**, 必须重写 RQ-VAE 训练加 sinkhorn 平衡项.