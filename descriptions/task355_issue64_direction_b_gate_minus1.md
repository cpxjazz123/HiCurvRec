# Task #355 / Issue #64 Gate -1 — Direction B 重开 三层可学习混合曲率乘积空间 预检

**日期**: 2026-07-31
**触发**: GitHub Issue #64 (owner 创建) `[方向B 重开] 三层可学习混合曲率乘积空间——修复α/分量曲率到SID的同源链路`
**基线**: HG-Rec Task #84 Test R@10=0.1020
**决策阈值**: Gate -1 任一 FAIL → NO-GO 收口

---

## 1. Issue #64 主张

Direction B 重开, 目标 = 用 **layer-specific learnable mixture + constrained curvature components** 修复 #56 NO-GO (SID unique=0.01% mode collapse).

公式 (per issue body):
```
d_l = Σ_m softmax(w_l,m) * s_l,m * d_{κ_l,m}(z_l, c_l,m)
```

需要 4 类参数 per-layer:
- **α_l (mixture state)**: softmax component weights w_l,m
- **κ_l,m (constrained curvature components)**: per-component curvature (constrained, e.g. ∈ [c_min, c_max])
- **s_l,m (scale)**: per-component scale factor
- **effective geometry**: 上述三者组合

## 2. 现有实现状态

### 已实现 (可重用)
| 类 | 文件 | 包含 |
|---|------|------|
| `FreeCurvVectorQuantization` | `hrqvae_free_curv.py` | per-layer θ_m (learnable κ_m, M=1) |
| `FreeCurvVectorQuantizationMixedCurv` | `hrqvae_issue55_56.py` | α_l_raw (sigmoid), κ_fixed scalar |
| `FreeCurvVectorQuantizationMixedCurvWithScale` | `hrqvae_issue55_56.py` | α_l_raw + scale_l, κ_fixed |
| `FreeCurvVectorQuantizationMixedCurvFixed` | `hrqvae_issue55_56_fixed.py` | Issue #59 修复: α_l_raw 不再 dead |
| `FreeCurvVectorQuantizationMixedCurvWithScaleFixed` | `hrqvae_issue55_56_fixed.py` | α_l_raw + scale_l 不再 dead, κ_fixed |

### 未实现 (Issue #64 新增)
- **per-component κ_l,m**: 多个 learnable κ per layer (现有 = κ_fixed scalar)
- **softmax weights w_l,m**: 显式 softmax mixture weights (现有 = α_l sigmoid 单 weight)
- **constrained curvature components**: 约束 κ ∈ [c_min, c_max] (现有 = 无约束)

## 3. Gate -1 spec (Issue #64 强制)

8 项检查, 全部 PASS 才算 Gate -1 PASS. 任一 FAIL → STOP, NO-GO 收口.

| Test | 名称 | 检查点 |
|------|------|--------|
| T1 | 实施基础就位 | Direction B 架构 (κ_l,m + w_l,m + s_l,m) 完整实现 + commit |
| T2 | L0/L1/L2 mixture/curvature 隔离 | per-layer α_l, κ_l,m, s_l,m 独立 + 互不影响 |
| T3 | 干净 optimizer (param groups) | 4 类参数都 requires_grad, 0 dead params |
| T4 | Forward path clean | encoder/assignment/loss 无 .item() detach |
| T5 | Batch 维度独立 | 无 global state pollution |
| T6 | Optimizer state detach | effective geometry autograd 不被 optimizer state 干扰 |
| T7 | codebook/SID update path 隔离 | L0/L1/L2 不互相污染 |
| T8 | Stage 3/4 接口对齐 | R12 ckpt save/load + state_dict round-trip |

## 4. R11.5 决策

按用户 loop 指令, 先做 Gate -1 (zero-GPU), FAIL 即 NO-GO 收口.

审计策略: 优先 audit 最近实现 `FreeCurvVectorQuantizationMixedCurvWithScaleFixed` (α_l + scale_l + κ_fixed, 跟 Direction B 最接近). T1 若确认 per-component κ_l,m 缺失 → 直接 FAIL → 写 NO-GO verdict, 不浪费 GPU.

---

result: Issue #64 Gate -1 — 8 项 zero-GPU 审计 (Direction B 实施基础 + mixture/curvature/scale 隔离 + optimizer + forward + interface). T1 失败 (κ_l,m 未实现) → 立即 NO-GO 收口, 不进 Gate 0.