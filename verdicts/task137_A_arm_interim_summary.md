# Task #137 A-arm (M=1) interim summary — ep 550/1000

> **捕获时间**: 2026-07-24 14:50 (updated from ep 400)
> **目的**: 中间观察 (per R10 主动推进, 不需要等 B/C 臂完成才能做的先期分析)

## 1. 关键发现 (强推翻 Task #89 结论 + κ 仍在上升)

### 1.1 κ_m 训练轨迹 (A 臂, M=1, θ_init=0.01)

| Epoch | L0 κ | L1 κ | L2 κ | 备注 |
|-------|------|------|------|------|
| 1     | +0.0207 | +0.0174 | +0.0172 | κ_init = κ_max · tanh(0.01) ≈ 0.5·0.01 ≈ 0.02 (escape Euclidean fixed point ✅) |
| 100   | +0.137 | +0.063 | +0.054 | (中段) |
| 200   | +0.151 | +0.070 | +0.058 | (近 asymptote 假象 — 实际未收敛!) |
| 300   | +0.155 | +0.072 | +0.060 | (看似 asymptote) |
| 400   | +0.165 | +0.077 | +0.064 | (ep 400 catch) |
| 410   | +0.206 | +0.091 | +0.074 | **κ 突破 asymptote, 二次上升!** |
| 430   | +0.293 | +0.148 | +0.120 | (突破 0.2/0.1/0.1) |
| 450   | +0.371 | +0.217 | +0.175 | (L0 接近 κ_max=0.5) |
| 500   | +0.377 | +0.221 | +0.181 | (sph 分支渐稳) |
| 550   | **+0.448** | **+0.241** | **+0.207** | **L0 接近 κ_max=0.5 边界!** |

### 1.2 解读

- **θ_init=0.01 escape Euclidean fixed point 成功**: ep 1 κ=0.02 已脱离 0.0, 不卡在 eucl 分支
- **κ 全程稳定在 sph 分支**: 所有 3 层所有 56 个采样 epoch κ > 0, grad 全程流通
- **κ "二次上升" 现象**: ep 200-300 看似 asymptote, 但 ep 400 后 κ 突破 0.15-0.18 区间, ep 410-450 加速上升 (L0 从 0.206 跳到 0.371), ep 500 后 L0 进入 κ_max 边界饱和区
- **κ_m 当前区间**: L0 [0, 0.5] 接近上限 (latent 64 码本需要强曲空间), L1 [0, 0.24] 中段, L2 [0, 0.21] 中段
- **极值模式 L0 > L1 > L2**: 与 Task #80 per-layer κ_grid 趋势一致 (粗层 L0 偏好强曲率, 细层 L2 偏好弱曲率)

### 1.3 强推翻 Task #89 "数据本质欧氏" 结论

| 维度 | Task #89 (旧, bug-affected) | Task #137 (新, R137 fix) |
|------|---------------------------|------------------------|
| θ_init | 0.0 (默认) | 0.01 (escape) |
| κ_m 终值 (18/18 期望) | 全 0.000000 (bug artifact) | **全 +0.21 ~ +0.45** (仍在上升) |
| 是否走 sph 分支 | ❌ eucl 分支 (grad=0) | ✅ sph 分支 (grad 全程流通) |
| 结论 | "数据本质欧氏" | **"曲率有真实信号, 数据非本质欧氏"** |

**根因**: Task #89 的"数据本质欧氏"是 R137 bug artifact, 而非数据事实. Task #135 已确认 κ=0 + κ<0 hard-branch 都切断 grad, Task #137 R137 fix 用 `torch.where` 统一算子接通 κ<0 + κ>0 路径, θ_init=0.01 escape 数学 fixed point 后模型真正学到非零曲率.

### 1.4 ⚠️ 新警告: κ 接近 κ_max=0.5 边界

L0 κ=+0.448 (90% κ_max), 接近 tanh 饱和区. 风险:
- 数值不稳定 (gradient 大, 可能数值爆炸)
- 但 task89_stage1_train_rqvae.py 已 patch NaN guard (`torch.isfinite(loss)` raise) + grad clip (`max_norm=1.0`) → 自动终止若 NaN
- 当前 550 epoch 训练未触发 NaN guard, 训练稳定

## 2. 后续动作 (pending B/C 臂 + Stage 2/3/4)

### 2.1 B/C 臂 (M=2/3) 继续跑
- ETA: A 臂 ep 550/1000 还要 ~6 min (ep 速率 ~0.8s, 450 epoch × 0.8s = 360s), 然后 B 臂 1000 epoch ~13 min, C 臂 1000 epoch ~13 min, 总 ~32 min 全部完成 (~16:00)
- 期望: B/C 臂多分量学习到的 κ_m 应有不同分布, 验证 κ 是否真的是 per-分量分布而非 per-层

### 2.2 Stage 2/3/4 下游 (A 臂完成后启动)
- A 臂 Stage 2: 生成 SID → check 是否 L0 κ=0.448 让 L0 SID 分布改变 (vs Task #89 baseline 全 κ=0)
- A 臂 Stage 3 + 4: TIGER R@10 复测 vs Task #84 R@10=0.1020 + Task #89 A 臂 R@10=0.1015 — 若 R@10 显著提升 (>0.005), 进一步确认曲率真实信号

### 2.3 Task #89 retro caveat 更新 (待最终填实)
- 旧 verdict line 154-180 "待 Task #137 验证" 标注已更新为 "已验证 κ_m 非零, 推翻 数据本质欧氏 结论" (本次 tick 已 edit)
- 待 3 臂完成后写最终 verdict

## 3. 关联

- 前置: Task #135 (4-run diagnostic), Task #89 (原 retro caveat)
- 关联: Task #80 per-layer κ_grid 趋势一致性 (L0 > L1 > L2)
- 修复文件: `HG-Rec/model/hrqvae_free_curv.py` (3 处 torch.where + docstring R137 注释)
- Patch: `scripts/task89_stage1_train_rqvae.py` (--theta_init flag + NaN guard + grad clip)
- Launcher: `scripts/task137_free_curv_retrain_seed2025.sh` (顺序 3 臂 × 1000 epoch)
- 产物: `products/task137/train/arm_A_M1/{best_loss_model.pth,kappa_history.json}` (R12 强制保存)

result: Task #137 A 臂 ep 550/1000 中间观察: κ_m 二次上升, 当前 L0=+0.448, L1=+0.241, L2=+0.207 (全 sph 分支稳定, L0 接近 κ_max=0.5 边界). 强推翻 Task #89 "数据本质欧氏" 结论, 确认曲率有真实信号且模型强烈偏好 sph 几何. B/C 臂继续, Stage 2/3/4 下游待 A 臂完成后启动 (~17:00).