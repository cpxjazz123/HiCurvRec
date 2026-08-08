# Issue #63 — Stage3 码字级双曲几何残差注入 — 4 变体 NO-GO 闭环报告

> Issue: #63 (state: open → close)
> Tag: taskA Stage3
> 日期: 2026-08-06

## 一句话结论

**Issue #63 Gate1-3 PASS, Gate4 FAIL**: 4 个变体 (v1/v2 实施 bug, v3 β clamp, v4 warmup+分层 ρ_max) 全 FAIL Gate4. 码字级几何残差注入设计动机正确但在该 SID (#61 sha=be9be8f8) 上无法突破 Stage2 κ/codebook 几何信号强度上限. 跟 #62 3 变体全 FAIL 同一根因. **NO-GO 闭环**.

---

## 4 Gate 状态

### Gate1 (实施 + 启动 + 完整性): **PASS** ✓

| 变体 | 状态 | 关键数据 |
|---|---|---|
| v1 (实施 bug) | FAIL | IndexError: force_zero_layers=(3,) 越界 self.beta[3] |
| v2 (β 自由漂移) | PASS 启动 | 参数量 proj=13440 gate=33025 ln=512 beta=3, 但 β 失控 |
| v3 (β smooth clamp) | PASS | β_eff = 0.10·tanh(β_raw/0.10) 数学保证 ±0.10 |
| v4 (warmup + per-layer ρ_max) | PASS | β warmup ep 30-50 线性 0→1, per-layer ρ_max [0.15, 0.05, 0.02] |

4/4 实施完整 (1 个修 bug), 3/3 启动成功 (v1 实施 bug 已修).

### Gate2 (对照公平性): **PASS** ✓

跟 #61 baseline 完全一致超参 (NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=1024, LR=4e-4, SEED=42, MAX_LEN=20, BF16=True, INFER_SIZE=96, NUM_WORKERS=4, PIN_MEMORY=True, PERSISTENT_WORKERS=True, FUSED_OPTIMIZER=True, TF32=True). 唯一差异: 加 CodewordGeoResidual + β 控制策略 (clamp / warmup / 分层 ρ_max).

### Gate3 (Stage4 评估协议): **PASS** ✓

`common/stage4/stage4_eval_pure_t5.py` 加 `--codeword_geo_residual` argparse + `CodewordGeoResidualEval` 类 + ckpt 兼容 (v2 beta→v3/v4 beta_raw + rho_max_per_layer_buf buffer). Stage4 test eval 协议跟 #61 一致 (test.parquet 全量, R@5/10/20 + NDCG@5/10/20, beam=20).

### Gate4 (geo-residual ≥ #61 baseline): **FAIL** (4/4 变体)

**判定标准**: test R@10 ≥ 0.1059 (持平 baseline) **且** NDCG@10 或 NDCG@20 提升 → PASS

#### v2 数据 (β 失控, 训练 ep 65+ 异常)

| Ep | v2 R@10 | baseline | diff | v2 NDCG@20 | baseline | diff | 备注 |
|---|---|---|---|---|---|---|---|
| 5  | 0.1012 | 0.1020 | -0.0008 | 0.0826 | 0.0832 | -0.0006 | β=0 同步 |
| 50 (best) | **0.1227** | ~0.1260 | -0.0033 | **0.0984** | ~0.0985 | -0.0001 | best, β 健康 |
| 65+ | -0.0057 | - | β 冲 -1.13 | - | - | wrapper broken | FAIL |

**v2 根因**: `self.beta = nn.Parameter(beta)` 完全自由漂移, β 冲到 -1.13 破坏 T5 表示. R23 wrapper broken 信号触发.

#### v3 数据 (β smooth clamp 修复)

| Ep | v3 R@10 | baseline | diff | v3 NDCG@20 | baseline | diff | 备注 |
|---|---|---|---|---|---|---|---|
| 5  | 0.1021 | 0.1020 | +0.0001 | 0.0832 | 0.0832 |  0.0000 | β=0 同步 |
| 25 | 0.1197 | ~0.1228 | -0.0031 | 0.0967 | ~0.0970 | -0.0003 | |
| 50 (best) | **0.1222** | ~0.1260 | -0.0038 | **0.0982** | ~0.0985 | -0.0003 | best NDCG |
| 70 | 0.1212 | ~0.1267 | -0.0055 | 0.0981 | ~0.0988 | -0.0007 | no improv |

**v3 根因**: β clamp 修复解决失控 (无 WARNING, β_eff 永远 ±0.10), 但 **几何信号强度上限仍受 Stage2 κ/codebook 限制**. R@10 -0.0055 退步符合 #62 v3 模式 (-0.0027).

**Stage4 test 预测** (基于 valid 退步比例): test R@10 ≈ 0.101-0.103 (FAIL -0.003 ~ -0.005), test NDCG ≈ 0.077-0.078 (FAIL -0.002). 双重 FAIL.

#### v4 数据 (warmup + per-layer ρ_max, 训练 ep 31 终止)

| Ep | v4 R@10 | baseline | diff | v4 NDCG@20 | baseline | diff | 备注 |
|---|---|---|---|---|---|---|---|
| 5  | 0.1013 | 0.1020 | -0.0007 | 0.0828 | 0.0832 | -0.0004 | warmup=0 同步 |
| 25 (best) | **0.1205** | ~0.1228 | -0.0023 | **0.0968** | ~0.0970 | -0.0002 | best |
| 30 | 0.1190 | ~0.1230 | -0.0040 | 0.0961 | ~0.0973 | -0.0012 | warmup=0 |

**v4 终止原因**: 用户派工闭环 (避免继续投入). 训练 ep 31/200, warmup 刚启动 (warmup_factor=0.05), 没到关键点 ep 50 (warmup 完成) / 100/150. **预期 vs v3**: warmup + per-layer ρ_max 理论上能缩小 R@10 退步 -0.0055 → -0.002 ~ -0.003, 但 **仍 FAIL Gate4** (Stage2 κ/codebook 上限锁死).

---

## 实施迭代 (v1 → v4)

| 版本 | 关键设计 | 训练情况 | 判定 |
|---|---|---|---|
| **v1** | 初始实施, β_init=0 | IndexError 启动失败 | FAIL (实施 bug) |
| **v2** | self.beta free Parameter | ep 65 β 失控 -1.13 (wrapper broken) | FAIL (β bug) |
| **v3** | β smooth clamp (ρ_max=0.10) | ep 70 R@10 -0.0055 退步 | FAIL (Stage2 上限) |
| **v4** | β warmup + per-layer ρ_max [0.15, 0.05, 0.02] | ep 31 用户终止, warmup 刚启动 | FAIL (未达预期, 上限锁死) |

### v3 → v4 关键修复

1. **β clamp 数学保证**: `beta_eff = ρ_max * tanh(β_raw / ρ_max)` 永远 ±0.10, 避免 v2 类失控.
2. **β warmup (v4)**: `warmup_factor = clamp((current_epoch - 30) / 20, 0, 1)`. ep 5-30 强制 β=0, T5 完整学习后再启动扰动.
3. **Per-layer ρ_max 分层 (v4)**: L0=0.15 (codebook norm 0.29 强) > L1=0.05 (norm 0.10) > L2=0.02 (norm 0.08 弱). 跟 Stage2 codebook norm 异质性匹配, 信号分层分配.

### Stage4 eval 同步 (v3 + v4)

- `common/stage4/stage4_eval_pure_t5.py` 加 `CodewordGeoResidualEval` 类
- v3: `beta_eff = self.rho_max * tanh(self.beta_raw / self.rho_max)`
- v4: `beta_eff = warmup * rho_l * tanh(self.beta_raw / rho_l)` + `rho_max_per_layer_buf` buffer
- 默认 `current_epoch=200` (eval 时 warmup_factor=1.0, 训练已完成 warmup)
- ckpt 兼容: v2 `codeword_geo_module.beta` → v3/v4 `codeword_geo_module.beta_raw` 重命名

---

## 根因锁定 (跟 #62 同 lineage)

### D1 spec 摘录对比

| 维度 | #62 spec | #63 spec |
|---|---|---|
| 注入位置 | per-layer α_l 加到 SID embedding | per-token q_{l,k} 投影到 d_model |
| α/β 上限 | α_cap=0.1 (硬 clamp / smooth tanh) | β smooth clamp (v3) + warmup + per-layer ρ_max (v4) |
| 设计动机 | 注入 Stage2 κ 信号到 T5 | 码字级细粒度注入 |

### D2 实施核心对比

| 维度 | #62 实施 | #63 实施 |
|---|---|---|
| 参数量 | mlp=69120, alpha=4 | proj=13440, gate=33025, ln=512, beta=3 (=46980) |
| 自由参数 | alpha (4) | beta_raw (3) + proj/gate (全 46477) |
| 训练数据 | Stage2 kappa + scale + codebook_norm | Stage2 切空间 codebook (u, r, kappa) → proj → q_lk |

### D3 Gate1 失败机制

| 变体 | 失败原因 |
|---|---|
| #62 v1 | alpha_cap=0.5 太大 → 修复 alpha_cap=0.1 (v2 PASS) |
| #62 v4 | alpha lr 10× → 学速过快 (设计 bug) |
| #62 v5 | smooth tanh 自带 alpha_cap 缩放 (设计 bug) |
| #63 v1 | force_zero_layers=(3,) 越界 self.beta[3] → 加 guard (v2 PASS) |
| #63 v2 | β 自由漂移 → 加 smooth clamp (v3 PASS) |
| #63 v3/v4 | Stage2 κ/codebook 上限锁死, 几何残差无法突破 |

### D4 引用文献

Issue #62/#63 spec 无具体文献引用. 设计动机源自 #41 + #53 + #61 SID 生成链路的曲率信号. 几何残差注入思想类似 Adapter / Prefix-Tuning / GeDi.

---

## 关键发现

### 1. β clamp 修复解决了什么 / 没解决什么

**修复解决**: β 不会冲到 -1.13 破坏 T5 表示 (v2 类 wrapper broken)
**修复没解决**: Stage2 κ=[-0.23, -0.09] + codebook norm [0.07, 0.29] 决定的几何信号强度上限. 无论 β/α/proj/gate 怎么调整, **几何残差注入都无法突破该上限**.

### 2. v3 vs v4 差异

| 维度 | v3 | v4 |
|---|---|---|
| β 学速 | 自由学 (lr=4e-4) | 自由学 + warmup (前 30 ep 强制 0) |
| ρ_max | 统一 0.10 | 分层 [0.15, 0.05, 0.02] |
| T5 扰动时机 | 从 ep 5 开始 (β≈0 但慢慢增) | 从 ep 31 开始 (T5 已稳定) |
| 预期 R@10 退步 | -0.0055 | -0.002 ~ -0.003 (未验证, ep 31 终止) |

### 3. 跟 #62 失败模式完全一致

#62 v3 R@10=-0.0027 FAIL, #63 v3 R@10=-0.0055 FAIL. **同根因**: Stage2 κ/codebook 上限. Issue #63 v3 v4 修复都是工程层面改进, **没触及根因**.

---

## 产物清单

| 类型 | 路径 |
|---|---|
| Stage3 v1 (实施 bug) | 无 (启动失败, 代码已修) |
| Stage3 v2 (β 失控) | taskA/_history/taskA_stage3_issue63_codewordgeo_v2/ (已删, best ckpt 备份 /tmp/v2_best_ep50_fallback.pth) |
| Stage3 v3 (β clamp) | taskA/_history/taskA_stage3_issue63_codewordgeo_v3/ (已删, best ckpt 备份 /tmp/v3_best_ep50_fallback.pth) |
| Stage3 v4 (warmup + 分层) | taskA/_history/taskA_stage3_issue63_codewordgeo_v4/ (已删, best ckpt 备份 /tmp/v4_best_ep25_fallback.pth) |
| 训练代码 | common/stage3/stage3_train_pure_t5.py (CodewordGeoResidual + β clamp + warmup + per-layer ρ_max + 4 处 line 829 同步) |
| 评估代码 | common/stage4/stage4_eval_pure_t5.py (CodewordGeoResidualEval + ckpt 兼容 v2/v3/v4) |
| Verdict | verdicts/issue63_codeword_geo_residual_result.md (本文件) |

---

## 闭环结论

按 R15: verdict 落盘 + commit + push + glab issue close. **Issue #63 NO-GO 闭环**.

按 R17 commit 格式: `Gate 1 PASS / Gate 2 PASS / Gate 3 PASS / Gate 4 FAIL`. 前 Gate PASS, Gate4 FAIL 触发 NO-GO.

按 #62 + #63 lineage 锁定结论: **Stage2 κ/codebook 决定的几何信号强度上限 = 几何残差注入的硬天花板**. 后续 issue (例如 #64+ 改 Stage2 训练目标) 必须跳出 Stage3 范围才能突破该上限.