# Issue #225 Stage2 learnable κ 框架设计 (用户 2026-08-09 新方向)

## Context

**用户原话**: "try to think the new curvation framework to improve stage2 curvation shoul learnable"

**现状分析** (基于代码考古):

Stage2 v15 capmatch 实际上**已经是 learnable κ**:
- `kappa_drift` 是 `nn.Parameter` (taskA_stage2.py:565)
- CURV_PRIOR=True (line 134)
- REL_STRUCT 推动 per-layer ρ_ball → κ (line 138)
- κ EMA trust region 防 κ 漂移 (line 1372)
- 训练 1000 ep 后学到 κ = [0.30, 1.79, 1.48] (c = [1.35, 6.00, 4.39], 已非平凡)

所以 "learnable" 这层已实现。用户隐含的意思可能是:

### 解读 A: κ 不应该是 per-layer 标量,而是 per-item-conditioned
- 当前: κ_l 是一个标量,所有 item 共享 (3 个标量)
- 改进: κ_{l,i} 应该取决于 item i 在球面的位置 (i.e., per-item radius)
- 物理动机: 球面上不同位置的 item,双曲距离的意义不同. 离球心近的 item 用更平的 κ, 远的用更曲的
- 这与 Stage1 per-item radius (Issue #141 v77) 是协同的: Stage1 给每 item 一个 radius, Stage2 用这个 radius 调制 κ_l

### 解读 B: Stage2 应该比现在更激进地学习 κ
- 当前 CURV_PRIOR_LAMBDA=0.1 + REL_STRUCT_LAMBDA_BALL=200
- 改进: 加大这两个正则项, 让 κ 在训练中更大程度偏离 baseline
- 风险: κ 学到极值 → 码字撞边界 → Stage3 端 R@10 塌缩 (Issue #59 历史教训)

### 解读 C: κ 不应该只是 frozen scalar 传给 Stage3
- 当前: final_cs (3 个标量) 序列化进 ckpt, Stage3 HAB 预计算 Dbar_l 时使用
- 改进: κ 应该真正是 per-batch learnable, 即 Stage3 也能微调它
- 这就是 Issue #224 CPL (Stage3 端加扰动). 用户可能想要的是 (A)+(B)+(C) 三层协同.

## 设计: 三个实施方案

### 方案 1: Stage2 κ per-item-conditioned (推荐 ★★★★★)

**思想**: Stage1 v77 给每 item 一个 per-item radius r_i (∈ [0, 1)), Stage2 用 r_i 调制 κ_l:
```
κ_l_i = κ_l_base + α_l · r_i · tanh(MLP_κ([r_i]))   # per-layer per-item κ
```
- κ_l_base 来自 Stage2 训练 (3 个标量, 保持 v15 capmatch 学习)
- α_l 是 per-layer 调制系数 (新学, init 0.05)
- MLP_κ 是 1→1 的简单网络 (r → bounded offset)
- 物理意义: 远球面的 item 用更大 κ (更曲), 近球心用更小 κ (更平)

**实施步骤**:
1. `taskA/stage2/taskA_stage2.py`: 修改 HRQVAEVectorQuantization
   - `__init__` 加 `self.kappa_peritem_mlp = nn.Sequential(nn.Linear(1, 1), nn.Tanh())` (per-layer)
   - `forward` 接收 `item_radii` (B,) 参数, 替换 scalar κ_eff 计算
2. `train.py`: 从 Stage1 加载 item_radius, 在 dataloader 里传 item_radius
3. 训练 ~1000 ep, 验证 Stage2 Gate1 (码本 utilization > 0.85)

**风险**: 改动 Stage2 训练数据流 + ckpt 兼容性. 需保留 backward compat (Stage1 没传 item_radius 时用 baseline κ_l).

### 方案 2: Stage2 加强 κ 学习信号 (备选 ★★★)

**思想**: 直接加大 CURV_PRIOR_LAMBDA 和 REL_STRUCT_LAMBDA_BALL, 让 κ 学得更激进.

**实施**:
```python
CURV_PRIOR_LAMBDA = 0.5   # 0.1 → 0.5 (5×)
REL_STRUCT_LAMBDA_BALL = 400.0  # 200 → 400 (2×)
```

**风险**: κ 学到极值 → 码字撞边界 → Stage3 R@10 塌缩 (Issue #59 重演). 需要加强 κ EMA trust region.

### 方案 3: Stage2 + Stage3 联合 κ 微调 (与 Issue #224 CPL 协同 ★★★★)

**思想**: Stage2 训练好 κ 传给 Stage3, Stage3 用 CPL 进一步微调 (Issue #224 已实施).

**实施**: Issue #224 CPL 已经在跑 (PID 66272), 训练中. 等结果.

## 推荐优先级

1. **P0 (立即)**: Issue #224 CPL 训练 (PID 66272) 已在跑, 等结果 (5-50 min)
2. **P1 (后续)**: Issue #225 方案 1 (per-item-conditioned κ) — 需要 1-2 小时实施 + 训练
3. **P2 (后续)**: Issue #225 方案 2 (加强 κ 信号) — 需要 30 min 实施 + 训练

## 当前状态 (2026-08-09 12:33)

- Issue #224 CPL Stage3 训练 **正在跑** (PID 66272, 4 卡 70% util, **ep30 valid_R@10=0.1127**, 已 4 min)
- c_perturb warmup T0=50 还未到 (ep50 时启动曲率扰动)
- Issue #225 实施: **待用户授权启动** (GPU 已被 CPL 占满,无法并行 Stage2 训练)
- 0.108 复现: **待训练结果**

## 新思考角度 (2026-08-09 12:33)

**用户原话深度解读** "stage2 curvature should be learnable" 的真正含义可能是:

1. **不是"是否 learnable" (已经是), 而是"更细粒度 learnable"**
   - 当前: per-layer scalar κ (3 个标量, 三层共享一个池化)
   - 用户想要: per-item κ (9922 个 item 各自独立 κ) 或 per-(layer, item-pair) κ

2. **具体路径: 让 κ 的学习信号来自 per-item 多样性**
   - 当前: REL_STRUCT 用 per-layer scalar target
   - 改进: target 变成 `target_base_l + β · item_radius_offset_i` (per-layer base + per-item offset)
   - 物理意义: 不同位置的 item 对 ρ_ball 有不同需求, κ 应该响应这个多样性

3. **方案 3-Minimal (P1 ★★★★★)**: 修改 REL_STRUCT 让 target per-item-conditioned
   ```python
   # 现有:
   target = self._struct_target()  # per-layer scalar
   
   # 改进:
   item_r = latent.detach().norm(dim=-1).mean()  # 当前 batch mean radius
   target = self._struct_target() + β_peritem * mlp(item_r.unsqueeze(0))  # per-item offset
   ```
   - 改动量: ~5 行代码
   - 训练时间: 仍 1000 ep (8 min DDP)
   - ckpt 兼容: 仍存 final_cs (3 个标量), Stage3 HAB 路径不变
   - 真正"让 Stage2 curvature learnable" — κ 信号来自 per-item, 但仍是 per-layer 标量 (Stage3 兼容)

4. **风险评估**:
   - 当前 v15 capmatch κ=[0.30, 1.79, 1.48] 已是非平凡值
   - 改进预期: κ 应该更激进学习 (如 [0.50, 2.50, 2.00])
   - Issue #59 历史教训: κ 学到极值 → 码字撞边界 → util < 0.85
   - 缓解: 仍保留 RAD_SAFE + κ EMA trust region, 加强 REL_STRUCT_LAMBDA_BALL

## 决策 (2026-08-09)

- 当前 GPU 被 CPL 训练占满 (4 卡 70% util), Stage2 训练无法并行
- 等 CPL 训练完成 (ep200 ~17 min 总时间) 后, 再决定是否启动 v2 Stage2 训练
- 如果 CPL ep50+ valid_R10 超过 0.130 (= v77 baseline 0.129),说明 CPL 已突破, 不需要 Stage2 重训
- 如果 CPL ep50+ valid_R10 持平 0.1127, 说明 CPL warmup 后无增益, 立即启动 v2 Stage2 训练 (per-item-conditioned target patch)

**Why**: 用户 2026-08-09 新方向, 暗示 Stage2 曲率本身应该更激进可学. 当前 κ 已是 learnable 但偏保守 (per-layer scalar), 用户可能希望更细粒度.
**How to apply**: 等待 CPL 训练结果 (Issue #224). 如果 R@10 > 0.108, CPL 路径成功. 如果 < 0.108, 启动 Issue #225 方案 3-Minimal.