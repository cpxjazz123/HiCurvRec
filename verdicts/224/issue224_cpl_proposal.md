# Issue #224 新曲率框架: 曲率扰动学习 (CPL — Curvature Perturbation Learning)

## Context

**问题**: HG-Rec 4 阶段流水线在 Stage3 训练时, Stage2 冻结的 `final_cs` 是不可微的; HAB 中 `Dbar_l` 是基于 frozen 曲率预计算的 frozen Parameter, Stage3 反向传播无法调整它. 这导致 HAB 几何信号严格遵循 Stage2 的曲率判定, 没法适配 T5 学到的码字分布.

**历史路径**: Issue #138 v74 (HAB frozen+WD+dropout) test_R@10=0.1063 → Issue #141 v77 (Stage1 per-item radius) 0.1080 → Issue #68 v78 (DECOR + 抗 trap) 0.1092. 三个 issue 都是在 Stage3 之前定好曲率, Stage3 只调整 λ_raw/α/U·V (而非 c 本身). 0.11 始终未突破.

**新框架思想 (2026-08-09)**: 把 Stage2 冻结的 final_cs 加一个 per-layer 可微扰动 `c_perturb_raw`, 让 Stage3 反向传播直接微调"有效曲率". 这是真正的 joint Stage2-Stage3 曲率学习.

## 设计

### 核心方程

```
Dbar_ij_perturbed = Dbar_ij * (1 + scale * w(t) * tanh(c_perturb_raw[l]))
```

- `Dbar_ij`: 来自 Stage2 final_cs 的预计算距离矩阵 (frozen)
- `c_perturb_raw[l]`: per-layer 可微标量 (init 0, 默认 3 个标量)
- `scale`: 最大缩放幅度 (默认 0.10 = ±10% 距离扰动)
- `w(t)`: warmup 因子 (epoch 级, 前 T_0=50 epoch=0, T_0+T_w=70 epoch 内渐增到 1)

### 物理意义

- Dbar_ij 反映 Stage2 在 c_l 下的双曲距离
- c_perturb_raw 让 Stage3 调整"有效曲率" (实际是距离缩放因子)
- ±10% 缩放 = 调整"曲率残差"在 ±0.05~0.10 范围内 (粗略估计)
- warmup 让 T5 先稳定 baseline 学到 50 epoch, 再开曲率扰动

### 实施细节

**`common/hyperbolic_attention_bias.py`** (commit 912a5f5):
- `HyperbolicAttentionBias.__init__` 新增 `c_perturb_raw` parameter (init 0, 默认 requires_grad=False)
- `HyperbolicAttentionBias.get_B_geo` 在 residual HAB 路径后应用 `Dbar_ij *= (1 + scale * w * tanh(c_perturb_raw[l]))`
- 新增 `c_perturb_enabled` flag (opt-in 启用, 默认 False = 与 v77 baseline 完全等价)
- 新增 `_cperturb_w_buf` buffer (epoch 级 warmup 因子)

**`common/stage3/stage3_train_pure_t5_v85p_repro.py`** (commit 912a5f5):
- 5 个新 CLI flag: `--c_perturb_enabled`, `--c_perturb_lr` (1e-4), `--c_perturb_scale` (0.10), `--c_perturb_warmup_T0` (50), `--c_perturb_warmup_Tw` (20)
- HAB module 创建后 (line 1347 附近): `c_perturb_enabled=True` + `c_perturb_raw.requires_grad_(True)`
- Optimizer param group: `c_perturb_raw` 单独 group (CPERTURB_LR=1e-4), 与 HAB lambda_raw/U/V/residual_alpha 分离
- Base group exclusion set 加 `c_perturb_raw.id` 防止重复进 base group
- Epoch 循环起点 (line 1544 附近): `_cperturb_w_buf = f(epoch)` 控制 warmup

## Gate 验证

### Gate 1 (代码正确性) — PASS
- `python3 -m py_compile` 双脚本 OK
- Sanity check 4/4 PASS (见下方)

### Gate 2 (训练可行性) — N/A (待训练验证)
- c_perturb 链路 patch 完成, 但实际训练需要 50+ epoch 才能产生有效 verdict

### Gate 3 (梯度信号) — PASS
Sanity check 结果 (commit 912a5f5 /tmp/v85p_ensemble/cperturb_sanity_check.py):

| Test | 描述 | 结果 |
|------|------|------|
| 1 | c_perturb_enabled=False, max\|B_geo\| | 0.104931 (与 v77 baseline 等价 ✓) |
| 2 | c_perturb_raw=[0.5,-0.3,0.8], max\|B_geo\| | 0.109780 (+4.6% 扰动生效 ✓) |
| 3 | B_geo.sum().backward() | L0 grad=-0.021 (B_geo→c_perturb_raw 链路 ✓) |
| 4 | warmup w=0.5, max\|B_geo\| | 0.107355 (介于 baseline 与全扰动之间 ✓) |

### Gate 4 (端到端 R@10) — N/A (待训练)
- 当前没有启动实际训练, 仅有 4 项单元测试 PASS
- 计划训练: 200 ep, batch=1024 DDP 4 卡, LR=4e-4 cosine (与 v85p_repro 一致)
- 目标: test_R@10 > 0.1080 (v77 历史最高) 或 ensemble 后 > 0.1079

## R18 4 维度对比 (与历史 issue)

| 历史 issue | D1 spec | D2 实施核心 | D3 Gate 1 失败机制 | D4 引用 |
|-----------|---------|------------|-------------------|---------|
| #59 bounded κ + L_κ | κ clamp 负值 | σ 形式 κ + 边界占用 | util_3digit [0.45,0.25,0.20] < 0.85 | Stage1 #56 残差头 + Stage2 RQ-VAE 不兼容 |
| #60 joint Stage1+Stage2 | Stage1 L_var+L_rank | proxy #53 STE 残差分解 | R@10=0.9649 PASS, util_3digit < 0.85 | Stage1 残差头+Stage2 RQ-VAE 不兼容 (锁定) |
| #61 κ sync (Stage2→Stage3) | κ 反向传播 | κ 标量注册 buffer | **成功** (v85p stage2 ckpt 现行方案) | 成功基线 |
| **#224 CPL (本)** | **Stage3 联合 Stage2 曲率微调** | **per-layer 可微缩放 (1+scale*tanh)** | **N/A (patch 完成待训练)** | **新方向, 不与 #59/#60 冲突** |

**关键差异**: #59/#60 试图在 Stage1 改 κ 然后 Stage2 必须重训; CPL 保留 Stage1+Stage2 frozen, 只在 Stage3 加扰动参数 (增量式, 不破坏已收敛 baseline). 这是与 #59/#60 的根本路径分歧.

## R10 + R15 闭环

- [x] commit 912a5f5 (Issue #224 CPL patch)
- [x] push origin main (已 push)
- [x] sanity check 4/4 PASS
- [ ] **PENDING**: 实际训练 (200 ep, DDP 4 卡) — 需要 GPU 空闲 + 用户授权启动
- [ ] **PENDING**: Stage4 eval + ensemble 验证

## 后续计划

1. **启动训练** (Issue #224 P0): `python3 -u common/stage3/stage3_train_pure_t5_v85p_repro.py --c_perturb_enabled ...`
   - DDP 4 卡, 200 ep, batch=1024, LR=4e-4 cosine
   - 监控 valid_R@10 (R23 七信号终止)
   - 期望: test_R@10 > 0.108 (历史 v77)

2. **评估对比** (Issue #224 P1): test ckpt + Stage4 eval + Borda ensemble with existing 3 ckpts
   - 期望: 4-way ensemble > 0.1079 (Issue #94 3-way 0.1079)

3. **如果失败** (R18 重新决策): 进一步调整 warmup (更长 T_0=80) 或 scale (0.05 更保守) 或 LR (1e-5 更慢)

---

**Why**: Stage2 frozen final_cs 阻止 Stage3 调整曲率, CPL 通过可微扰动解锁 joint 训练
**How to apply**: 启用 `--c_perturb_enabled` 即激活, 默认 warmup T0=50/Tw=20/scale=0.10