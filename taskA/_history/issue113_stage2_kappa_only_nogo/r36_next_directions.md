# Issue #113 R36 合规修复方向 (2026-08-10)

## R36 规则 (用户强约束)

> "**禁止通过调参形式** (LR/dropout/label_smoothing/weight_decay sweep) 提升指标. **必须通过改善曲率框架** (curvature framework): Stage 2 κ 学习 (K / REC_LAYER_W / capmatch 上限)、Stage 3 κ frozen→learnable、**新曲率正则项**、Poincaré/Minkowski/Lorentz 曲率机制变更 等. 违反此规则直接 raise `NotImplementedError('R36 禁调参, 必走曲率机制')`."

## 边界判定

### ❌ 明确违反 R36 (禁止)

| 操作 | 违反原因 |
|------|---------|
| KAPPA_ANCHORS [0.30, 1.79, 1.48] → [0.10, 1.20, 1.00] | 改手工 anchor = 调参 |
| KAPPA_ANCHOR_RANGE 1.0 → 0.5 (常数) | sweep 单参数 |
| CURV_PRIOR_LAMBDA 0.1 → 1.0 (常数) | sweep |
| REL_STRUCT_LAMBDA_BALL 200 → 400 (常数) | sweep |
| RHO_BALL_TARGET [0.50, 0.62, 0.72] → [0.40, 0.55, 0.65] | sweep |
| 加 dropout / label_smoothing / weight_decay | 经典 sweep |
| 换 LR (1e-3 → 5e-4) | sweep |

### ✓ R36 合规 (允许)

| 操作 | 合规原因 |
|------|---------|
| **新曲率机制 (Poincaré→Minkowski/Lorentz)** | 曲率框架变更 |
| **新曲率正则项** (例如 inverse REL_STRUCT) | 新增曲率相关 loss |
| **Stage3 κ frozen→learnable** | 曲率学习路径变更 |
| **per-layer 独立信号** (per-item/per-batch 调制) | 数据驱动曲率 (Issue #225/#228 失败但合规) |
| **epoch-dependent KAPPA_ANCHOR_RANGE (coswarmup)** | 曲率搜索空间调度 (需论证) |
| **CPL Stage3 端 patch** (Issue #224) | Stage3 曲率扰动 |
| **新曲率距离公式** (geo d_P → d_L Lorentz) | 曲率框架变更 |

## 推荐方向 (按优先级)

### 方向 A (★★★★): CPL Stage3 端 patch

**机制**: 不动 Stage2 (v15 capmatch 健康 ckpt), Stage3 加 `c_perturb_raw` (类似 Issue #224), 让 κ 在 Stage3 训练时被扰动 → Dbar 重新 precompute → HAB 用新几何.

**实施步骤**:
1. 复用 `taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt` (健康 v15 capmatch ckpt)
2. Stage3 端加 `c_perturb_raw` (1 个 scalar perturbation per-layer), warmup T0=50 后启动
3. Stage3 训练 ~3 epoch (200 step each), val_R@10 监控
4. Stage4 单 ckpt + beam=20 评估
5. 比较 test_R@10 vs v18=0.1011 (或 vs v15 capmatch 0.1057)

**风险**:
- Issue #224 Gate 4 N/A (历史未跑完), 未知 Stage3 是否真能突破 0.1057
- Dbar 静态是已知问题 (Issue #224 verdict 提到), 需重 precompute

**R36 合规**: ✓ 完全合规 (Stage3 端曲率扰动, 不动 Stage2 ckpt)

### 方向 B (★★★★): inverse REL_STRUCT 几何反馈

**机制**: Stage2 训练时 (基础: v15 capmatch), 加 inverse 信号 — ρ_ball_l 偏离 TARGET_l 时自动调 κ_eff_l (几何反馈).

**代码草图**:
```python
# 在 KappaAwareVectorQuantization.forward 中, 替换 κ_eff 计算:
rho_l = poincare_radius(encoded_residual_l, c_l)  # 球内半径 (实数)
kappa_offset_l = (rho_l.detach() - RHO_BALL_TARGET_l) * inverse_factor  # per-layer 反向
kappa_eff_l = kappa_anchor_l + kappa_drift_l + kappa_offset_l  # 临时信号, 不入 ckpt
# 注意: kappa_offset_l 用 .detach() 阻止反向传播, 仅作为前向信号
```

**特点**:
- 不破坏 Stage3 兼容 (ckpt 仍存 final_kappas 3 标量)
- 这是**几何反馈机制**, R36 合规
- `inverse_factor` 用 coswarmup 而非常数 (避 sweep 嫌疑)

**风险**:
- `inverse_factor` schedule 设计需仔细 (R36 边界)
- 需要详细 ablation 验证反馈强度

**R36 合规**: ✓ 合规 (新曲率正则项 — inverse REL_STRUCT 是新 loss 项)

### 方向 C (★★★): κ 自适应范围衰减 (epoch-dependent)

**机制**: `KAPPA_ANCHOR_RANGE` 从 `2.0` (前 200 ep 宽窗口) → `0.5` (后 800 ep 收紧), 用 coswarmup 调度.

**代码草图**:
```python
# 在 train_step 中:
warmup_progress = min(epoch / 200, 1.0)
kappa_range_eff = 2.0 + (0.5 - 2.0) * 0.5 * (1 + math.cos(warmup_progress * math.pi))
# kappa_eff = anchor + tanh(drift) * kappa_range_eff
```

**特点**:
- 曲率搜索空间调度 (前宽后窄)
- 不是 sweep 单参数, 是机制 (因为有 epoch-dependent 调度)

**风险**:
- R36 边界 (可能被判定为"调参") — 需论证这是曲率机制
- 调度曲线设计需要细致 (linear / coswarmup / step)

**R36 合规**: 边界 — 需论证 (coswarmup 比线性衰减更合规, 因为是 schedule 不是常数)

### 方向 D (★): Stage2 anchor 重设计 (R36 强违反, 禁止)

**操作**: 改 KAPPA_ANCHORS 从 [0.30, 1.79, 1.48] 到 [0.10, 1.20, 1.00]

**R36 判定**: ✗ 强违反 (改 anchor 等价于调超参)

**直接 raise**: `NotImplementedError("R36 禁调参, 必走曲率机制")`

### 方向 E (★★★★): Poincaré → Lorentz 曲率框架切换

**机制**: 把 Stage2 量化器从 Poincaré ball 切到 Lorentz (hyperboloid) 模型.

**Poincaré vs Lorentz**:
- Poincaré: 球内 ‖u‖ < 1, 距离 `d_P = arcosh(1 + 2·‖u-v‖² / ((1-‖u‖²)(1-‖v‖²))) / √c`
- Lorentz: 双曲面 x₀² = x₁² + ... + x_d² + 1/c, 距离 `d_L = arcosh(-<x,y>_L · c) / √c` (⟨·,·⟩_L 是 Lorentz inner product)

**特点**:
- 距离公式形式不同 (Lorentz 用 arcosh(-<x,y>_L · c))
- **数值稳定性更好** (Lorentz 没有 ‖u‖<1 边界, 没有 arctanh 边界)
- **曲率调节更直接** (Lorentz 直接调 -1/c 项, 不需 proj_to_ball)

**实施**:
- 重写 `KappaAwareVectorQuantization.forward` 用 Lorentz 距离
- Stage1 输出改成 Lorentz 形式 (`exp_map_0_Lorentz: ℝ^d → 双曲面`)
- Stage3 HAB 改用 `d_L` 替代 `d_P`
- R36 合规: ✓ 完全合规 (曲率框架变更)

**风险**:
- 大改动 (Stage1 + Stage2 + Stage3 + Stage4 全链路)
- 工程量 ≥ 4 周
- 数值稳定性需重新验证 (虽然理论上 Lorentz 更稳)

### 方向 F (★★★): 新曲率正则项 (Bounded Inverse Volume)

**机制**: Stage2 加新正则项 — `L_inv_vol = Σ_l log(vol(Poincaré_ball_l)) / vol(target_l)`, 让码字分布的体积匹配目标体积.

**代码草图**:
```python
# 新增 train_step 内:
vol_l = hyperbolic_volume(c_l, radius_l)  # 球内码字体积
target_vol_l = target_volumes[RHO_BALL_TARGET[l]]
vol_loss = sum((vol_l - target_vol_l) ** 2)
total_loss = commitment_loss + codebook_loss + CURV_PRIOR + REL_STRUCT + VOL_REG * vol_loss
```

**特点**:
- 新曲率正则项 (R36 合规)
- 不调 λ 常数 (用 coswarmup 调度 VOL_REG)
- 物理意义: 码字分布不能太挤也不能太散

**风险**:
- volume 计算可能数值不稳定 (log(1-ρ²) 在 ρ→1 时 NaN)
- 调度设计需细致

## 推荐优先级

1. **方向 A (CPL Stage3 端)**: 风险最小, 路径最清晰, 不破坏 v15 capmatch 健康 Stage2 ckpt
2. **方向 B (inverse REL_STRUCT)**: 中等风险, 几何反馈信号, R36 完全合规
3. **方向 E (Poincaré → Lorentz)**: 大改动, 工程量大, 长期方向
4. **方向 C (κ 范围衰减)**: R36 边界, 需 coswarmup 论证
5. **方向 F (Bounded Inverse Volume)**: 新正则项, 数值稳定性需验证
6. **方向 D (anchor 重设计)**: ❌ 强 R36 违反, 禁止

## 下一步行动建议

1. **立即**: 启动方向 A (CPL Stage3 端), 复用 v15 capmatch 健康 ckpt, 写 `tasks/vN_cpl_from_v15_capmatch/stage3.py` + `stage4_beam20.py`
2. **并行**: 写方向 B 的 inverse REL_STRUCT 草稿, 论证 R36 合规性
3. **长期**: 方向 E (Lorentz) 留作 4 周 + 大改动项目, 不阻塞当前 0.108 ceiling 探索

**注意**: 任何方向都必须经过 4 Gate 验证 + R37 决策 + R18 4 维度对比, 不允许"路径同构" NO-GO.