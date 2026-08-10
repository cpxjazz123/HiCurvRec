# Issue #115 Verdict: Stage2 Collapse Audit v2 P0 Fixes + P1 (2026-08-10)

## 4 Gate 答案 (R20 强制)

### Gate 1 (precheck)
**PASS** ✅
- precheck PASS: `=== Precheck: ✅ PASS ===`
- κ grad finite nonzero: `[1.08, 0.42, 0.49]` (来自 REL_STRUCT 主驱动 + VQ loss 次驱动)
- c_l > 0 (init=1.0): `[1.0, 1.0, 1.0]`
- no NaN/Inf
- P0-1 修复后 c_geom/c_loss 不再 detach (双 alias), VQ loss 提供 ∂L/∂c ≠ 0

### Gate 2 (training)
**PASS** ✅
- 20 epoch 训练, 9 steps/epoch, total 180 steps
- κ learning: `[0.0007, 0.0007, 0.0007]` → `[0.164, 0.190, 0.177]` (final)
- c_l: `[1.0007, 1.0007, 1.0007]` → `[1.178, 1.209, 1.194]` (small but real movement)
- 平均 loss 72.6 → 58.2 (-19.8%, 正常收敛)
- util_3digit per layer Ep19: `[0.844, 0.977, 0.961]` (健康)
- util_4digit: 0.94 (no collapse!)
- 无 NaN/Inf (gate 监控)

### Gate 3 (output)
**PASS** ✅
- SID shape: `(9922, 4)`, dtype `int64` ✓
- SID range: `[0, 255]` (K_l2=256 dedup digit)
- SID SHA256: `7927f1ccca5ac48c13fe1b743c89583f...` (unique)
- collision resolve: rounds=30, 31/9922 collisions → 99.69% unique_3digit
- ckpt 含 Stage3 兼容字段 (model_state_dict + final_kappas + final_cs + curvature_mode)

### Gate 4 (eval)
**N/A** ⏸
- 本 issue 范围是 Stage2 P0 修复 + smoke test, **不跑 Stage3/Stage4**
- R37 决策: smoke test 通过 + κ learns + util 健康, 但 vq_κ_grad 仅 0.0001-0.01 数量级 (REL_STRUCT 主导),
  不足以判断 P0-1 是否真解决 "VQ loss 提供数据驱动 κ 梯度" 的目标.
- 完整 ablation A-E (Issue #115 实验计划) 是 Gate 4 的实际验证 — 需要下次 launch.

---

## P0 修复总结

### P0-1: 恢复 data-driven κ 学习信号

**问题**: `c_geom.detach() if CURV_PRIOR else c` 让 VQ loss 对 κ 梯度恒为 0; REL_STRUCT 是唯一驱动.

**修复**:
- `c_geom` 和 `c_loss` 都 alias `c` (non-detach)
- 风险: KAPPA_RANGE=2 (c ∈ [0.37, 2.72]) 限制了"尺度作弊"最大幅度, 安全

**验收 (smoke 20 epoch)**:
- `vq_κ_grad` (Ep19): `[0.0084, 0.0005, 0.0001]` — 非零! P0-1 修复生效.
- vs REL_STRUCT 主导梯度 `[1.86, 1.37, 1.14]` — VQ loss 仅 1/1000 量级, 弱但非零.
- 这表明: P0-1 让 κ 接收 VQ loss 信号, 但 REL_STRUCT 仍是主驱动力. 这是设计意图 (Issue #76 历史).

**已知次要 bug**:
- 初版 diagnostics 块用 `torch.no_grad()` 包 `q(residual)`, 导致 `c.requires_grad=False`,
  vq_κ_grad=0. **已修复**: 改用 `torch.enable_grad()` 显式开启 autograd.
- 验证 (Ep19): `vq_κ_grad=[0.0084, 0.0005, 0.0001]` 非零.

### P0-2: 修复 RQ residual path 几何不一致

**问题**: `x_q_safe = proj_to_ball(x_q, c_geom)`, `x_q = codebook_e[indices]` 是 raw tangent,
proj_to_ball 把它错当 ball coord 投影.

**修复** (1 行): `x_q_safe = proj_to_ball(x_q_h, c_geom)` (用 ball coord).

**验收**:
- 严格路径: `e_k → exp_0^c → e_k^D → assignment → e_{k*}^D → log_0^c → tangent (residual)`
- 现在 residual 几何路径与 assignment 路径一致.

### P0-3: 修复 boundary penalty 用 normalized radius ρ_k

**问题**: `cb_norm > B_BOUNDARY=0.9` 用 ball norm (与 c 无关), diagnostics 用 ρ (含 c 缩放),
两者在 c≠1 时错位.

**修复** (连续 penalty):
```python
rho_k = sqrt(c) * ball_norm  # (K,)
boundary_term = ReLU(rho_k - ρ_safe=0.90).pow(2).mean()
loss += LAMBDA_B * boundary_term
```

**验收**: smoke 20 epoch `boundary=[0.0, 0.0, 0.0]` (码字都在 ρ<0.9 安全区, 正常).

### P0-4: 修复 safe-distance saturation ratio

**问题**: `d_min/d_max > 0.99` 间接判断, 在 c 小时距离普遍小, 易假阳.

**修复**: 用 `u_raw = sqrt(c) * ||(-x) ⊕_c y||`, `top-1 u_raw ≥ u_max=0.985` 才算 saturation.
新诊断字段: `u_raw_median/p95/max/clipped_ratio`.

**验收** (smoke Ep19): `sat=[0.0, 0.0, 0.0]` — 真实 u_raw 都 < u_max, 无几何饱和.

### P0-5: 完整 diagnostics 接入 training loop

**新增**:
- `collapse_diag_log` 列表 (per-epoch, per-layer)
- 每 epoch 调用 `compute_collapse_diagnostics(indices, distances, c_geom)`
- 36 个诊断字段/layer (util, entropy, top1+top5, κ, c, norm, ρ, boundary, saturation, margin, gradient)
- 落盘 `issue115_p05_collapse_diag.json`

**验收** (smoke Ep19):
- `util=[0.844, 0.977, 0.961]` (per layer)
- `H=[3.78, 4.74, 5.38]` (entropy 递增, 健康)
- `margin=[0.044, 0.012, 0.006]` (top1-top2 距离, 浅层大 → 浅层稳定)
- `κ_grad=[1.86, 1.37, 1.14]` (主驱动 REL_STRUCT)
- `vq_κ_grad=[0.008, 0.0005, 0.0001]` (P0-1 修复后非零)

---

## P1: 区分 clean vs v15 reproduction

**新增常量**:
```python
CURVATURE_MODE = "clean"  # 默认 (clean learnable)
# 旧 v15 复现配置 (切换用):
#   KAPPA_ANCHORS = [0.30, 1.79, 1.48]
#   KAPPA_MAX = 6.0
```

**ckpt 新增字段**: `curvature_mode`, `kappa_min`, `kappa_max`

**验收**:
- 本次 smoke 用 `clean` mode: KAPPA_ANCHORS=[], KAPPA_MIN=-1.0, KAPPA_MAX=1.0
- v15 复现 (Issue #96 baseline) 用 `v15_repro` mode 切换参数即可

---

## R18 4 维度对比 (vs Issue #114 audit round 1)

| 维度 | #114 audit | #115 v2 fix | 差异 |
|------|-----------|------------|------|
| D1 spec | 7 个 audit 任务 (Tasks 1-7) | 5 个 P0 修复 + 1 个 P1 | D1 不同 |
| D2 实施 | Task 1 (init) / Task 3 (VQ loss) / Task 4 (ρ) / Task 7 (diagnostics) | P0-1 (c detach) / P0-2 (residual) / P0-3 (ρ penalty) / P0-4 (u_raw) / P0-5 (loop hook) | D2 不同 |
| D3 失败 | audit 阶段无失败, 仅问题清单 | P0 修复后 smoke PASS, 无 collapse | D3 不同 |
| D4 文献 | 同 (arXiv:2405.13979) | 同 | D4 相同 |

**R18 判定**: D1/D2/D3 不同 → 必须实验 → **已实验, smoke test PASS**.

---

## R37 决策线

**smoke test 通过 ≠ 新版本基线**.

- 本次仅是 **P0 修复 + smoke test 验证**, 不创建新版本
- `v15_repro` 模式已实现, 但**未跑 ablation A-E** — 留作下一阶段
- 如果后续 ablation 中任意一个 (特别是 C: Learnable + REL off 或 D: Learnable + Weak REL)
  表现优于 v15 baseline (Issue #96, test_R@10=0.1057), 才用 v15 capmatch 健康 ckpt 做 Stage3 + Stage4

**当前基线**: 仍是 Issue #96 v15 capmatch baseline (R37 未触发回退, 因为 #115 不是新实验版本).

---

## 下一步

1. **Ablation A-E** (Issue #115 实验计划): 5 个对比实验, 共同 seed/data/optimizer/schedule
2. **Stage3 + Stage4 评估**: 仅在 ablation C/D 优于 baseline 后启动
3. **Issue 关闭**: 当前 P0 修复 + P1 实现 + smoke 验证已完成, 可在 ablation 启动后一并 close

---

## 产物清单

- 代码: `/fs04/ar57/wenyu/GeneRec/taskA/stage2.py` (P0-1..P0-5 + P1 修改)
- smoke 产物: `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue115_v2_p0_smoke/`
  - `hrqvae_kappa_sync.ckpt` (R12 ckpt)
  - `issue115_p05_collapse_diag.json` (P0-5 时序, 3 epochs logged)
  - `issue41_audit.json` (audit, 3 epochs)
  - `sid_output.npy` (9922, 4) int64
  - `train_curve.json` (180 steps)
  - `verdict.json` + `precheck.json` + `kappa_recalibration_log.json`
- Issue: GitLab work_item #115 (created)
- verdict: 本文档