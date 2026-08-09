# Issue #228 Stage2 κ per-batch radius modulation 提案 (用户 2026-08-09 新方向 ★★★★★)

## Context

**用户原话** (2026-08-09 /loop 5m): "try to think the new curvation framework to improve stage2 curvation shoul learnable"

**历史路径汇总**:
- ✅ v15 capmatch (Issue #157): Stage2 κ=[0.30, 1.79, 1.48], 已 learnable 但 per-layer scalar
- ❌ Issue #224 CPL (2026-08-09): Stage3 端 Dbar 缩放, valid ceiling 0.1151
- ❌ Issue #225 v2 per-item-conditioned target (2026-08-09): κ 负漂移到 -0.18, util_4digit=0.006 塌缩
- ❌ Issue #226 HPE (2026-08-09): T5 架构不兼容 (无 absolute position embedding)

**Gap**: 用户希望"Stage2 κ 更 learnable", 但 per-item 维度会引发 κ 漂移塌缩, per-layer scalar 已收敛饱和. 需要**新维度**让 κ 学习信号多样化。

## 设计: per-batch radius modulation (3-Minimal ★★★★★)

### 核心方程

```
κ_eff_l = κ_l_base * (1 + α_l * batch_mean(item_radius))

其中:
- κ_l_base: v15 capmatch 训练好的 per-layer κ (3 个标量)
- item_radius: Stage1 v77 输出的 per-item radius ∈ (0, 0.99]
- batch_mean(item_radius): 当前 batch 所有 item radius 的平均, scalar ∈ (0, 0.99]
- α_l: per-layer learnable 调制系数, init 0, range bounded to ±0.5 (sigmoid * 1.0 - 0.5)
```

### 与 #225 v2 (per-item-cond) 的关键差异

| 维度 | #225 v2 (NO-GO) | #228 (本方案) |
|------|----------------|---------------|
| 调制粒度 | **per-item** (9922 个) | **per-batch** (1 个 scalar) |
| 信号强度 | 9922× (相对 baseline) | ~1× (与 baseline 同量级) |
| κ 漂移风险 | **40× 梯度爆炸** (实测) | 几乎无 (scalar 调制, 与 κ base 同阶) |
| Stage3 兼容 | 需要 per-item κ 重计算 Dbar | final_cs 仍是 3 标量 (Stage3 HAB 不变) |
| Stage2 ckpt | 路径破坏 (per-item κ 需新字段) | 路径保持 (3 个 α_l 加 final_cs 末尾) |

### 物理意义

- batch_mean(item_radius) 是 batch 内 item 分布的"球面深度"
- 当 batch 大多是近球心 item (radius ~ 0.1): κ_eff ≈ κ_base * (1 - 0.05) ≈ κ_base × 0.95 (略平)
- 当 batch 大多是远球面 item (radius ~ 0.9): κ_eff ≈ κ_base * (1 + 0.45) ≈ κ_base × 1.45 (更曲)
- α_l 训练后捕捉 "batch 球面深度 vs κ 的最优耦合", 是 v15 baseline 不能表达的**二阶信号**

### 实施步骤

**`taskA/stage2/taskA_stage2.py`** (改动 ≤ 10 行):

1. `HRQVAEVectorQuantization.__init__` 加:
```python
self.alpha_radius_mod = nn.Parameter(torch.zeros(num_layers), requires_grad=True)  # init 0
self.alpha_radius_mod_max = 0.5  # sigmoid*1-0.5 后 bound 到 ±0.5
```

2. `forward` 在 `kappa_drift` 计算后插入 (REL_STRUCT 之前):
```python
# 新: per-batch radius modulation
if self.enable_per_batch_radius_mod:
    batch_item_r = item_radius[batch_indices]  # (B,) 来自 dataloader
    batch_r_mean = batch_item_r.mean().clamp(min=1e-3, max=0.99)  # scalar, 防退化
    # sigmoid bound α_l 到 [-0.5, 0.5]
    alpha_l_bounded = torch.sigmoid(self.alpha_radius_mod) - 0.5  # (num_layers,)
    # κ_eff = κ_base * (1 + α * r_batch_mean)
    kappa_drift = kappa_drift * (1.0 + alpha_l_bounded.unsqueeze(-1) * batch_r_mean)  # (num_layers, 1)
```

3. `train.py` 加 dataloader 输出 `item_radius` 字段 (从 Stage1 v77 加载)

### CLI flag

```python
_argparser.add_argument("--enable_per_batch_radius_mod", action="store_true",
                        help="Issue #228: Stage2 κ per-batch radius modulation (per-batch scalar, not per-item)")
```

### 训练配置 (复刻 v15)

- EPOCHS = 1000
- BATCH_SIZE = 1024 (DDP 4 卡 per-rank 256)
- LR = 1e-3
- α_radius_mod LR = 1e-3 (与其他 stage2 参数一致)
- κ EMA trust region (已存在) 仍然生效
- REL_STRUCT, CURV_PRIOR 损失不变 (只多 α 信号源)

### 验证

- **Stage2 Gate1**: util_4digit > 0.85 (硬阈值)
- **Stage2 Gate2**: κ ∈ [0.1, 2.0] (避免 #225 v2 负漂移)
- **Stage2 Gate3**: κ 末值变化 < 30% (vs v15 capmatch baseline)
- **端到端 Gate4**: Stage3 + Stage4 eval, test_R@10 > 0.1080 (v77 baseline)

## R18 4 维度对比 (vs 历史)

| Issue | D1 spec | D2 实施核心 | D4 失败风险 |
|-------|---------|------------|-----------|
| #55/v4 fix_c | κ 学到极值 | fix_c=True 冻结 c=1 | κ 负漂移撞边界 |
| #59 bounded κ | σ 形式 κ | 边界占用 | util_3digit < 0.85 |
| #224 CPL | Stage3 Dbar 缩放 | c_perturb_raw | Dbar 不重新计算 → Stage3 不重新学 |
| #225 v2 | per-item target | item_radius 调制 per-item | **40× 梯度爆炸** |
| **#228 (本)** | **per-batch target** | **batch_mean(item_radius) scalar 调制** | **per-batch scalar ≈ baseline 梯度量级** |

**关键差异**: #225 v2 失败因 per-item 让 κ 信号强度暴增, #228 用 per-batch scalar 让 κ 信号强度与 baseline 同阶, 不触发 40× 梯度爆炸。

## 决策 (2026-08-09)

- **推荐立即实施** ★★★★★
- GPU 空闲 (0/4), DDP 4 卡 ready
- 训练 ~8 min (v15 capmatch 1000ep 已验证时间)
- 改动 ≤ 10 行, 风险极低 (per-batch scalar, EMA trust region 防漂移)
- 如 Gate1-3 PASS, 立即接 Stage3 + Stage4 端到端验证 (~25 min 总时间)

## R10 + R15 闭环

- [x] Issue #228 提案落盘 (本文件)
- [ ] 实施 patch (≤ 10 行)
- [ ] Stage2 训练 1000 ep (DDP 4 卡, ~8 min)
- [ ] Gate 1-4 验证
- [ ] commit + push + close

**Why**: per-batch scalar 调制让 κ 学习新维度信号 (batch 球面深度), 不破坏 per-layer scalar 已收敛 base. 物理意义明确, 实施风险极低.
**How to apply**: 启用 `--enable_per_batch_radius_mod` 即激活. 默认 alpha_radius_mod=0 (与 v15 baseline 完全等价). Stage3 HAB 路径不变 (final_cs 仍 3 标量).