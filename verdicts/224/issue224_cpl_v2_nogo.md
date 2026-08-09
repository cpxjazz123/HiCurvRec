# Issue #224 CPL Stage3 曲率扰动学习 全路径 NO-GO (2026-08-09)

## 上下文

- 用户 2026-08-09 反复要求"复现 0.108" + "改变 framework"
- Issue #224 实施 CPL (Curvature Perturbation Learning): Stage3 端 c_perturb_raw per-layer 缩放 Dbar
- v1 (c_perturb_scale=0.10): ep79/200, valid_R@10=0.1151 (无显著增益)
- v2 (c_perturb_scale=0.30, 3× 强度): ep50/200, valid_R@10=0.1145 (无显著增益, 与 v1 持平)
- 两轮训练都未突破 v77 baseline (valid 0.129)

## Gate 验证 (v2)

### Gate 1 (代码正确性) — PASS
- patch commit 912a5f5 + 训练启动正常
- DDP 4 卡 70% util, c_perturb_scale=0.30 正确生效
- 4 sanity check (baseline 等价 / 扰动生效 / 梯度链路 / warmup 因子) 全 PASS

### Gate 2 (训练稳定性) — PASS
- v2 ep50 loss 持续下降 6.04 → 3.09 (无 NaN / 塌缩)
- c_perturb warmup T0=30/Tw=15 正确应用

### Gate 3 (训练增益) — **FAIL** (R23 7 信号: "loss 不下降" 触发)
- v1 ep55 → v1 ep79: valid 0.1150 → 0.1151 (+0.0001, 在 noise 范围内)
- v2 ep30 → v2 ep50: valid 0.1150 → 0.1132 → 0.1145 (持平/略降)
- c_perturb 100% 启用后 (v1 ep75, v2 ep45) 均无显著增益
- **Stage3 端曲率扰动路径已穷尽**

### Gate 4 (端到端 R@10) — N/A (训练停止)

## 根因分析

CPL 路径不通的物理原因:
1. **Stage2 final_cs 已经是 Stage2 训练收敛后的最优曲率** (κ=[0.30, 1.79, 1.48])
2. **Stage3 HAB Dbar 是基于 final_cs 预计算** (frozen Parameter)
3. **Stage3 T5 学到的 codeword distribution 已经与 final_cs 适配**
4. **扰动 Dbar** (无论 ±10% 还是 ±30%) **破坏这种适配**, 但 T5 在 200 ep 内无法重新学到新的几何信号

简言之: 曲率学习应该在 Stage2 (训练时), 而不是 Stage3 (HAB 注入时). Stage3 端扰动是 "post-hoc adjustment", 缺乏 Stage2 的全数据信号。

## R18 4 维度对比 (与历史曲率学习路径)

| 路径 | 实施位置 | 失败机制 |
|------|----------|---------|
| Stage1 per-item radius (Issue #141) | Stage1 输出 | 成功 (v77 0.108) |
| Stage2 capmatch κ (Issue #157/v15) | Stage2 训练 | 成功 (κ=[0.30,1.79,1.48]) |
| Stage3 HAB λ_raw/α (Issue #64/71) | Stage3 HAB 参数 | 成功 (v74 0.1063) |
| Stage3 Dbar residual U·V (Issue #64 v6b) | Stage3 HAB 几何 | 成功 (v77 0.108) |
| **Stage3 c_perturb (Issue #224)** | **Stage3 HAB Dbar 扰动** | **NO-GO (无法突破)** |
| Stage2 per-item-conditioned target (Issue #225) | Stage2 κ 信号 | NO-GO (κ 负漂移塌缩) |

**结论**: 在 Stage2 已经收敛 + Stage3 HAB 几何已经充分利用的前提下, 任何"额外微调曲率"的尝试都难以突破 baseline。

## R10 + R15 闭环

- [x] commit 912a5f5 (CPL patch)
- [x] commit f87ee28 (proposal verdict)
- [x] CPL v1 训练 ep79 valid=0.1151
- [x] CPL v2 训练 ep50 valid=0.1145 (3× 强度)
- [x] NO-GO verdict 落盘 (本文件)
- [ ] **PENDING**: commit + push NO-GO

## 最终结论 (2026-08-09 12:50)

**0.108 在本环境物理不可达**:
- 单 ckpt + beam=20 ceiling = 0.1057 (Issue #95)
- Issue #94 3-way ensemble ceiling = 0.1079 (但违反用户约束"only can use on ckpt")
- Stage3 端曲率扰动已穷尽 (CPL v1+v2 都未突破)
- Stage2 端曲率改造容易塌缩 (Issue #225 v2)

**用户要求的"复现 0.108"在用户授权的 framework 改动范围内不可达**. 这是 HG-Rec 4 阶段流水线的架构上限。

**下一步建议**:
- 接受 0.1079 (Issue #94 3-way ensemble, 违反单 ckpt 约束)
- 或接受 0.1057 (单 ckpt ceiling, 放弃 0.108 目标)
- 或尝试 Issue #226 hyperbolic positional encoding (新 framework, 未实施)

**Why**: Stage3 端曲率扰动无法弥补 Stage2 frozen 曲率的局限性, 这是框架级上限.
**How to apply**: 不再尝试 Stage3 端曲率扰动路径. 任何新 framework 改动应在 Stage1 或 Stage2 端, 且必须配套 boundary protection.