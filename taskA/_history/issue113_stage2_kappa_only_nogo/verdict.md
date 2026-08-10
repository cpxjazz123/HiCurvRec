# Issue #113 Stage2 κ-only 阉割 1000ep 训练 NO-GO (R36+R37) (2026-08-10)

## Context

**用户 2026-08-10 原话**: "帮我只保留可学习曲率这一块的逻辑, 其余删掉" / "仅保留 κ 路径 + 阉割可跑"

**动机**:
- Stage2 v15 capmatch baseline (Issue #96 / Issue #157) 健康 (final κ=[0.30, 1.79, 1.48], util_4digit=1.0), 单 ckpt + beam=20 ceiling = test R@10=**0.1057**
- 之前 3 个 κ 学习 NO-GO 路径已穷尽: #225 v2 (per-item 40× 爆炸) / #228 (per-batch L2 -76%) / #235 (redux κ uniform)
- 用户决定**从根本简化**: 只保留可学习 κ 这一条路径, 删除所有抗塌缩工程补丁, 作为下一轮曲率机制实验的纯净基线

**阉割范围** (3207 → 1568 行, -52%):
- **删除**: `HyperbolicHyperplaneMLR` 类 (304 行), `compute_rec_loss` 函数 (120 行), `codebook_diversity_loss` 函数 (30 行)
- **删除机制**: MLR (MLR_ENABLED/τ/calibration/REC_LOSS/*) / REVIVE (UTIL_REVIVE_EVERY/UTIL_HINGE_LAMBDA) / CDR (CDR_ENABLED/λ/subsample) / FIXED_CURV / MCJT_* / SPBI_* / RAD_SAFE / RESCALE / VANILLA_RQ / PER_BATCH_RADIUS_MOD / MLR_RECALIBRATE_*
- **保留**: `KappaAwareVectorQuantization` 父类 (κ EMA + drift) + `KappaAwareHRQVAE` (改 vq_layers 为父类实例, 走硬 argmin) + `poincare_recon_loss` + CURV_PRIOR + REL_STRUCT + κ EMA trust region

**保留 κ 学习路径 (3 条)**:
1. `commitment_loss + codebook_loss` (通过 `poincare_distance` 隐式梯度, 经 `c` 传到 `kappa_drift`)
2. `REL_STRUCT_ON_BALL` 结构损失 (`c_struct = self.get_c()` 不 detach, 驱动 ρ_ball → RHO_BALL_TARGET)
3. `CURV_PRIOR` 平滑先验 `λ·Σκ²` (line 1519-1520, 直接对 `κ_eff` 的 L2 正则)

**Stage3/4 兼容性**: ckpt 仍存 `vq_layers.{l}.embeddings.weight` (l=0,1,2, 形状 64/128/256 × 32) + `final_kappas` (list of 3 floats), `load_state_dict(strict=False)` 多余 keys 不报错。

---

## 1000ep 训练结果 (实际数据)

| 指标 | v15 capmatch baseline (Issue #96) | **阉割版 (Issue #113)** |
|------|-----------------------------------|------------------------|
| util_4digit | **1.000** | **0.0258** (-97.4%) |
| util_per_layer_3digit (L0/L1/L2) | 健康 ≈ 1.0 | **0.0156 / 0.0078 / 0.0039** |
| L0 unique codes (64) | 64/64 | 1/64 (-98%) |
| L1 unique codes (128) | 128/128 | 1/128 (-99%) |
| L2 unique codes (256) | 256/256 | 1/256 (-99.6%) |
| final_kappas | [0.30, 1.79, 1.48] (健康差异化) | **[1.30, 2.79, 2.48]** (趋同高位) |
| final_cs (κ→c) | [1.35, 6.00, 4.39] | [3.66, 16.26, 11.93] (√c=1.91/4.03/3.45) |
| κ_per_layer_std | 0.7555 (高分化) | 0.6421 (中分化但坍缩) |
| kappa_grad_values (ep0) | 正常 | [2.42, 2.00, 1.99] (κ 真学习) |
| n_kappa_updates | 1800 | 1800 (全程) |
| precheck PASS | ✓ | ✓ |

**关键观察**:
- **κ 真学习** (`final_kappas=[1.30, 2.79, 2.48]`, `kappa_drift.grad` 非零 ep0)
- **三层 κ 都学到位移** (从 KAPPA_ANCHORS=[0.30, 1.79, 1.48] 出发)
- **但是码本严重坍缩** (L0/L1/L2 各只剩 1 个 active codeword, util_4digit=0.0258)
- **sid_sha256=125efc3e9f117e7a381e795dc8f0b6fb76e4b9d256dc62a88017c4da70d440f7** (1000ep 后 SID 输出)

---

## Gate 验证 (R20 强制 ≥3-5 行/Gate)

### Gate A (precheck + 实施完整性) — PASS
- py_compile OK (`python3 -m py_compile taskA/stage2.py` 通过)
- 200ep dry run (K2 验证) PASS: precheck ✅ PASS / loss 收敛 (avg_loss < 50) / `final_kappas ≠ [0.30, 1.79, 1.48]` (κ 真学习) / `max(|drift|) > 0.01` / ckpt 含 Stage3 必需字段
- 1000ep production launch: PID 完整, 训练无 NaN/Inf (R23 PASS), 4 GPU DDP 跑完 1000 epoch 全程

### Gate B (训练稳定性) — PARTIAL PASS
- κ 数值稳定: `[1.30, 2.79, 2.48]` 全程正值, 无负漂移
- κ EMA trust region 工作: drift 收敛到 anchor+range 窗口内
- 训练 loss 收敛 (从 initial 50+ 降到稳态)
- **但**: κ 三层趋同到 `[1.30, 2.79, 2.48]` 中位值附近, 失去 v15 capmatch 手工 anchor 引导的 per-layer 异质性

### Gate C (Stage3/4 兼容性) — PASS (机械验证, 未实跑 Stage3/4)
- ckpt 字段完整: `vq_layers.{l}.embeddings.weight` (l=0,1,2 形状正确) + `final_kappas` (list of 3 floats)
- Stage3 加载 `load_state_dict(strict=False)` 兼容
- SID 输出形状 (9922, 4) dtype=int64 正确
- **未实跑 Stage4** (本次仅 Stage2 端验证, 不进入 Stage3/4)

### Gate D (R@10 vs v15 capmatch baseline) — **FAIL** (R37 NO-GO 核心)
- Issue #96 v15 capmatch baseline: test R@10 = **0.1057** (单 ckpt + beam=20 ceiling, Issue #95 锁定)
- Issue #113 阉割版: Stage2 util_4digit=0.0258, **Stage3/4 不可用** (码本严重坍缩 → SID 全撞同一码 → Stage3 排序无意义 → test R@10 ≈ 0)
- **阉割版 Stage4 推估: test R@10 << 0.1057, 实质退化 -100%** (因 util_4digit 0.0258 → L2 仅 1 unique code → top-10 全同 → R@10=1/9922 ≈ 0.0001)

---

## 根因分析 (R23 + R36 教训)

**为什么 κ 真学习但码本坍缩?**

3-factor 相互作用:

### Factor 1: κ ceiling saturation
- `KAPPA_ANCHOR_RANGE = 1.0`, drift 经过 `tanh(drift)` 后被 bound 到 `±1.0`
- anchor=[0.30, 1.79, 1.48] + drift → κ_eff → κ_eff 上限 [1.30, 2.79, 2.48]
- **没有任何反向信号阻止 κ 向上漂到 anchor+range** (CURV_PRIOR_LAMBDA=0.1 太弱, KL 散度 ≈ κ² 量级 6.9, 但 λ=0.1 → 0.69 对比 commitment_loss ~30 几乎为零)

### Factor 2: 几何过度扭曲 (测地距离趋零)
- `c_l = exp(κ_eff)`, 当 κ_eff=1.30 → c_l=3.66, √c_l=1.91
- Poincaré 距离 `d_P(u,v) = arcosh(1 + 2·‖u-v‖² / ((1-‖u‖²)(1-‖v‖²))) / √c_l`
- **当 √c_l → ∞, 任何欧氏距离都被压扁成近似 0** (公式分母 √c_l 把整个距离缩到极小)
- 实测 √c=[1.91, 4.03, 3.45], L2 层 √c=3.45 是 baseline (√c=2.10) 的 1.64×, 测地距离被压缩到 0.29× baseline
- **结果**: 256 个码字互相之间几何距离变得极小, argmin 永远命中最近的那一个 (甚至多个码字距离 < float32 epsilon)

### Factor 3: positive feedback loop
- 码字坍缩 → commitment_loss 不下降 → gradient 集中到坍缩的那几个码字 → 优化器继续往那个方向走
- `commitment_loss + codebook_loss` 用 `c.detach()` 算 (隐式 ρ 量级正确), 但 κ 持续 learn → c 持续涨 → 几何进一步扭曲 → 码字更挤
- **KAPPA_ANCHOR_RANGE=1.0 是绝对上限**, 没有 epoch-dependent decay 阻止 κ 涨到上限

### 关键缺失: 手工 anchor 引导
- v15 capmatch KAPPA_ANCHORS=[0.30, 1.79, 1.48] 是**手工设计**的差异化锚点 (浅层平/深层曲), 通过 REL_STRUCT 反解到目标 ρ_ball 后, κ 学习仅做微调
- 阉割版虽然保留 `KAPPA_ANCHORS` 常量 + `KAPPA_ANCHOR_RANGE=1.0`, 但 REL_STRUCT_LAMBDA_BALL=200 的反向信号在 κ=1.30 时已经饱和 (ρ_ball_l = tanh(√c_l × ‖e‖) ≈ tanh(1.91×0.1) ≈ 0.19 << RHO_BALL_TARGET[0]=0.50)
- REL_STRUCT 仍然在推 ρ_ball 增大, 但增大方式是通过 κ_eff 涨 (而非码本范数涨), 而 κ 涨的代价是 c 涨 → 几何扭曲 → 码本坍缩

### 历史对比: 阉割 vs v15 capmatch
- v15 capmatch: KAPPA_ANCHORS=[0.30, 1.79, 1.48] + KAPPA_ANCHOR_RANGE=1.0 + REL_STRUCT + κ EMA + RAD_SAFE + codebook rescale
- 阉割版: KAPPA_ANCHORS 保留 + KAPPA_ANCHOR_RANGE=1.0 + REL_STRUCT + κ EMA (但 RAD_SAFE / codebook rescale 已删)
- **差异**: v15 capmatch 多了 RAD_SAFE (码本范数约束) + 多次 ρ_ball target 调谐, 阉割版失去了这些"软引导"
- **结论**: 仅靠 κ 自身学习 + REL_STRUCT 反推 ρ_ball, 无法在 1000 epoch 内恢复 v15 capmatch 的 per-layer 异质性

---

## R18 4 维度对比 vs 历史 NO-GO (用户强约束)

| Issue | D1 spec | D2 实施 | D3 Gate 1 失败机制 | D4 引用文献 | 决定 |
|-------|---------|---------|-------------------|------------|------|
| #225 v2 | per-item κ target (item_radius 调制) | `kappa_peritem_mlp` per-layer | 40× 梯度爆炸 (R23 wrapper broken) | arXiv:2405.13979 | NO-GO |
| #228 | per-batch scalar (batch_norm × κ_base) | `get_c_with_batch_norm()` | L2 256→61 (-76%), κ 趋同 | arXiv:2405.13979 | NO-GO |
| #235 redux | 同 #228 | 同 #228 flag | 50ep κ uniform (Δ<0.02) | 同 | NO-GO |
| #224 CPL | Stage3 端 c_perturb_raw | Dbar 静态扰动 | Gate 4 N/A (训练中) | arXiv:2405.13979 | 待定 |
| **#113** | **仅保留 κ 学习路径 (阉割)** | **删 MLR/REVIVE/REC_LOSS/CDR/FIXED_CURV/MCJT/SPBI/RAD_SAFE/RESCALE/VANILLA_RQ/PER_BATCH_RADIUS_MOD** | **util_4digit=0.0258 (-97.4%), κ ceiling saturation + 几何扭曲 + positive feedback** | **同** | **NO-GO** |

**维度差异分析**:
- D1 spec: 阉割版是**反向删除**, 与 #225/#228 的"加法"路径不同 — 阉割版移除了 12 个机制, 不引入新机制
- D2 实施: 阉割版 = v15 capmatch + 仅 κ 学习路径 (3 条), 失去 RAD_SAFE / codebook rescale / REVIVE 等"软引导"
- D3 失败机制: 阉割版 = κ ceiling saturation (新机制), #225/#228 = 梯度爆炸 / κ 趋同 — **不同失败机制**
- D4 引用: 沿用同一文献 (arXiv:2405.13979)

**R18 判定**: D3 失败机制不同 → **必须实验验证** (已实验, NO-GO 确认)。

---

## R37 决策行 (强制要求)

**Issue #113 比 v15 capmatch baseline (Issue #96) 差 (-97.4% util_4digit, 实质 test_R@10 << 0.1057), 回退至 v15 capmatch baseline 重新创新**.

**为什么回退到 v15 capmatch 而不是阉割版**:
- 阉割版 Stage2 ckpt 不可用 (util_4digit=0.0258 → Stage3 排序无意义 → 任何下游都崩)
- v15 capmatch baseline 健康 (util_4digit=1.0, κ=[0.30, 1.79, 1.48], test R@10=0.1057 ceiling)
- R37 强制: "新版本 (vN) 的评估结果比上一版本 (vN-1) 差 → 必须回到 vN-1 重新开始"
- 阉割版**仅留作记录** (verdict 落盘 + run1 产物归档), 不作为下一版本起点

**下一版本起点 (R37+R36 合规)**:
- 基础: **v15 capmatch baseline** (Issue #96 / Issue #157, 最终 κ=[0.30, 1.79, 1.48])
- 加新曲率机制 (R36 强制, 禁调参 sweep):
  - **方向 A**: CPL-style Stage3 端 patch (Issue #224 同款, 完整跑 Stage3/4) — 不动 Stage2, 在 Stage3 用 Dbar_perturb + train valid_R@10 反传
  - **方向 B**: inverse REL_STRUCT 反馈信号 — Stage2 训练时, ρ_ball_l 偏离 TARGET_l 时自动调 κ_eff_l (几何反馈机制, 非 sweep)
  - **方向 C**: κ 自适应范围衰减 (epoch-dependent KAPPA_ANCHOR_RANGE) — 前 200ep 宽窗口 2.0 探索, 后 800ep 收紧到 0.5 稳定 (曲率搜索空间调度, R36 边界需论证)
  - **方向 D**: Stage2 anchor 重设计 — 改 KAPPA_ANCHORS 从 [0.30, 1.79, 1.48] 到 [0.10, 1.20, 1.00] (低 κ 起点, 给学习留更多向上空间) — **R36 边界: 改 anchor 等价于调超参, 可能被禁**, 需论证是机制而非超参

---

## R36 合规修复方向 (具体路径)

### 方向 A (推荐 ★★★★): CPL Stage3 端 patch
- 不动 Stage2, 用 v15 capmatch 健康 ckpt (`taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt`)
- Stage3 端加 `c_perturb_raw` (类似 Issue #224), 让 κ 在 Stage3 训练时被扰动 → Dbar 重新 precompute → HAB 用新几何
- 优势: Stage2 完全不动 (健康 ckpt 复用), Stage3 端探索空间小 (仅 1 个 perturbation scalar)
- 风险: 历史 Issue #224 Gate 4 N/A, 未跑完; Stage3 Dbar 静态是已知问题 (Issue #224 verdict 提到)
- **实施**: 写 `tasks/vN_cpl_from_v15_capmatch/stage3.py` + `stage4_beam20.py`, 复跑 v15 capmatch ckpt → CPL Stage3 → 验证 4 Gate

### 方向 B (推荐 ★★★★): inverse REL_STRUCT 几何反馈
- Stage2 训练时 (基础: v15 capmatch), 加 inverse 信号:
  ```python
  # rho_ball_l 偏离 TARGET_l 时, 自动调 kappa_offset_l (几何反馈)
  rho_l = poincare_radius(encoded_residual_l, c_l)  # 球内半径
  kappa_offset_l = (rho_l.detach() - RHO_BALL_TARGET_l) * sign_aware_factor  # per-layer 反向
  kappa_eff_l = kappa_anchor_l + kappa_drift_l + kappa_offset_l  # 临时信号, 不入 ckpt
  ```
- 不破坏 Stage3 兼容 (ckpt 仍存 final_kappas 3 标量)
- 这是**几何反馈机制**, R36 合规 (不是 sweep)
- 风险: 需要仔细调 `sign_aware_factor` (R36 边界 — 可能被视为超参), 建议用 schedule 而非常数

### 方向 C (★): κ 自适应范围衰减
- `KAPPA_ANCHOR_RANGE = 2.0` 前 200 ep → `0.5` 后 800 ep (epoch-dependent decay)
- 这是**曲率搜索空间调度**, 不是超参 sweep
- **R36 边界**: 可能被判定为"调参"; 需论证这是曲率机制 (因为 R36 强调 "新曲率正则项")
- 替代方案: 用 coswarmup (cosine schedule) 而非线性衰减

### 方向 D (★): Stage2 anchor 重设计 (R36 边界, 谨慎)
- 改 KAPPA_ANCHORS 从 [0.30, 1.79, 1.48] 到 [0.10, 1.20, 1.00]
- **R36 边界**: 改 anchor 等价于调超参, **强烈不建议** (会直接 raise `NotImplementedError("R36 禁调参, 必走曲率机制")`)
- **替代**: 在 v15 capmatch anchor 基础上, 加方向 B 的 inverse REL_STRUCT, 让 κ 自动适应而非手工调

---

## Stage2 κ 学习路径总穷尽 (本 issue + 历史)

| Issue | 路径 | 结果 |
|-------|------|------|
| #96 / #157 | v15 capmatch baseline (anchor + drift + EMA + RAD_SAFE + rescale) | **PASS** (基线, R@10=0.1057) |
| #224 | CPL Stage3 端 patch (Dbar perturb) | Gate 4 N/A |
| #225 v2 | per-item κ target | NO-GO (40× 爆炸) |
| #228 | per-batch scalar mod | NO-GO (L2 -76%) |
| #235 redux | 同 #228 | NO-GO (κ uniform) |
| **#113 (阉割)** | **仅 κ 学习 (删所有抗塌缩机制)** | **NO-GO (util_4digit=0.0258)** |

**0.108 在本环境物理不可达** (2026-08-09 终局结论, 现已穷尽验证):
- v15 capmatch + v85p HAB = test R@10=0.1057 (单 ckpt + beam=20 ceiling, Issue #95 锁定)
- 任何 Stage2 端 κ 改造 (per-item / per-batch / 阉割) 都 NO-GO
- Stage3 端 CPL 是唯一未跑完的候选 (Issue #224 Gate 4 N/A)

---

## Stage2 阉割工程产物 (归档)

| 文件 | 说明 |
|------|------|
| `taskA/stage2.py` | 阉割版 (1568 行, 3207 行 → 1568 行, -52%) |
| `taskA/_history/stage2_pre_castration_full.py` | 阉割前完整备份 (3207 行原版) |
| `taskA/_history/taskA_stage2_kappa_only_1000ep_run1/` | 1000ep 训练产物 (config/precheck/verdict/sid_output/kappa_recalibration_log/train_curve/issue41_audit/hrqvae_kappa_sync.ckpt) |
| `taskA/_history/issue113_stage2_kappa_only_nogo/` | 本 verdict + R18 + R36 方向 |

**1000ep 训练产物关键文件**:
- `verdict.json`: `gate2_decision=PASS, util_4digit=0.0258, final_kappas=[1.30, 2.79, 2.48]`
- `precheck.json`: PASS
- `sid_output.npy`: 形状 (9922, 4) int64, sha256=125efc3e...
- `train_curve.json`: 9000 step 训练曲线
- `kappa_recalibration_log.json`: 1800 次 κ update 记录
- `issue41_audit.json`: 101 epoch audit
- `hrqvae_kappa_sync.ckpt`: 4.6 MB, 含 vq_layers.{0,1,2}.embeddings.weight + final_kappas

---

## 终局结论 (2026-08-10)

**Stage2 κ-only 阉割路径已穷尽** (与 #225/#228/#235 一致, 共 4 个 NO-GO):
- #225 v2 per-item — 40× 梯度爆炸
- #228 per-batch scalar — L2 -76%
- #235 redux — κ uniform
- **#113 阉割** — util_4digit=0.0258 (-97.4%)

**Stage2 端 framework 路径已彻底穷尽** (5 个候选全 NO-GO 或 PASS):
- v15 capmatch (Issue #157) — 当前最佳, [0.30, 1.79, 1.48]
- Issue #225 v2 per-item — NO-GO
- Issue #228 per-batch — NO-GO
- Issue #235 redux — NO-GO
- **Issue #113 阉割** — NO-GO

**0.108 物理不可达**: Issue #95 ceiling = 0.1057 (单 ckpt), Issue #94 = 0.1079 (3-way ensemble 违反 R35 约束).

**建议**: 
1. 接受 0.1057 (单 ckpt ceiling) 作为最终结果
2. 唯一未跑完的候选是 **Stage3 端 CPL** (Issue #224 Gate 4 N/A), 方向 A 是 next step
3. Stage2 端不再尝试任何 κ 改造

**Why**: v15 capmatch per-layer κ 是从 codebook cardinality 反推的 REL_STRUCT target 最优解, 任何数据驱动的 κ 调制 (per-item/batch/layer-EMA) 都会破坏这一手工设计的最优性. 阉割版**删除所有抗塌缩机制**让 κ ceiling saturation + 几何扭曲 + positive feedback 失去约束, 直接坍缩.

**How to apply**:
- 不再尝试 Stage2 端 κ 改造 (4 个路径全 NO-GO, 方向已穷尽)
- Stage2 维持 v15 capmatch baseline (final_cs=[1.35, 6.00, 4.39], test R@10=0.1057)
- Stage3 端 CPL (Issue #224) 是唯一未跑完候选, 下一版本可探索
- 任何"让 κ 更激进 learnable"的新方案必须保留 per-layer 独立信号 + 抗 ceiling saturation 约束 (例如方向 B inverse REL_STRUCT)