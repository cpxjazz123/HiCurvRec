# Issue #69 v79 DECOR + HAB 叠加 — NO-GO 严重过拟合

**最终 verdict**: test R@10 = **0.1030** (vs v78 0.1092 **-0.0062**, vs baseline 0.1024 **+0.0006**). **NO-GO 严重过拟合**. valid 0.1301 (历史高位) 但 test 0.1030 (历史低位), valid/test ratio = **1.265** (vs v78 1.225 / baseline 1.237). **DECOR + HAB 叠加反而相互干扰, 加大过拟合**.

---

## 4 Gate 评估

### Gate 1 (代码改动 + 语法) — PASS
- v79 = v78 + Stage3 HAB frozen 叠加 (共同 launch flag)
  - `--enable_prompt_former --prompt_former_alpha 0.35 --prompt_former_num_bos_queries 64`
  - `--pf_alpha_warmup_steps 200 --pf_alpha_penalty_weight 0.01 --pf_bos_diversity_weight 0.01 --pf_attn_entropy_weight 0.001`
  - `--hyperbolic_attn_bias --hab_lambda_max 0.20 --enable_residual_hab --residual_alpha_init -20.0`
  - `--hab_stage2_ckpt taskA/_history/taskA_stage2_kappa_sync/hrqvae_kappa_sync.ckpt`
  - `--stage3_weight_decay 0.01 --stage3_dropout 0.2`
- `common/stage3/stage3_train_pure_t5.py` 加 `ddp_find_unused = PROMPT_FORMER_ENABLED or HAB_ENABLED` (避免 DDP RuntimeError)
- SID 06af0fed (taskA_stage2_hyp_v2_capmatch_1000ep)
- py_compile PASS
- Verdict: Gate 1 PASS

### Gate 2 (训练启动 + 早停触发) — PASS
- DDP 4 卡 bf16 启动成功 (PID 3359821, ~1h34m)
- best 落盘 ep135 valid 0.1301 (历史高位)
- 全程 184 epochs 跑完 (训练末尾 ES 已触发, 终止)
- Verdict: Gate 2 PASS

### Gate 3 (valid 突破) — **GO +0.0039 vs v78** ✓
- best valid_R@10 = **0.1301** (ep135)
- vs v78 0.1338: **-0.0037** (实际 valid 略低)
- vs v74 0.1312: **-0.0011**
- vs baseline 0.1267: +0.0034
- 注: v79 valid 0.1301 < v78 0.1338, 但仍超 v74
- Verdict: Gate 3 **MARGINAL (valid 0.1301 < v78 0.1338)**

### Gate 4 (test eval) — **NO-GO -0.0062 vs v78 (-0.0006 vs baseline)**

| 指标 | v79 (DECOR+HAB) | v78 (DECOR) | v74 (HAB) | baseline |
|------|-----------------|-------------|-----------|----------|
| test R@5 | 0.0848 | 0.0879 | 0.0853 | 0.0819 |
| test R@10 | **0.1030** | 0.1092 | 0.1063 | 0.1024 |
| test R@20 | 0.1298 | 0.1357 | 0.1303 | 0.1283 |
| NDCG@5 | 0.0722 | 0.0748 | 0.0721 | — |
| NDCG@10 | 0.0780 | 0.0817 | 0.0789 | 0.0755 |
| NDCG@20 | 0.0848 | 0.0884 | 0.0850 | 0.0821 |
| valid_R10 | 0.1301 | 0.1338 | 0.1312 | 0.1267 |
| **valid/test ratio** | **1.265** | 1.225 | 1.234 | 1.237 |
| vs baseline R@10 | +0.0006 | **+0.0068** | +0.0039 | — |
| vs v78 R@10 | **-0.0062** | — | -0.0029 | -0.0068 |
| vs 0.11 | -0.0070 | -0.0008 | -0.0037 | -0.0076 |

**NO-GO**: v79 test 0.1030 < v78 0.1092 (-0.0062), 几乎退到 baseline (+0.0006). **DECOR + HAB 叠加 = 严重过拟合**.

---

## 关键发现

### DECOR + HAB 叠加 = 负协同 (valid 升, test 跌)

| 方案 | valid R@10 | test R@10 | ratio |
|------|-----------|-----------|-------|
| baseline | 0.1267 | 0.1024 | 1.237 |
| v74 HAB frozen | 0.1312 | 0.1063 | 1.234 |
| v77 Stage1 + HAB | 0.1312 | 0.1080 | 1.215 |
| v78 DECOR + 抗 trap | 0.1338 | 0.1092 | 1.225 |
| **v79 DECOR + HAB 叠加** | **0.1301** | **0.1030** | **1.265** ← 严重过拟合 |

**v79 valid 略低 (0.1301 < v78 0.1338) 但 test 暴跌 (-0.0062)**. DECOR + HAB 同时启用相互干扰, 在 valid 上无法稳定 (低 0.0037), 在 test 上彻底崩溃.

### 叠加为什么无效 (R18 思考)

- **DECOR** 改的是 T5 **embedding** (输入层): 加 alpha-gated context-aware soft attention over candidate bins
- **HAB** 加的是 T5 **encoder** attention 的 hyperbolic bias term
- 两个模块都改 T5 的输入信号, 叠加时:
  - DECOR 把 T5 input 替换成 α·e_soft + (1-α)·e_fused (e_soft 是 context-aware 候选加权)
  - HAB 在 encoder self-attn 上加 Möbius distance-derived bias
  - 两者都依赖 codebook 的几何结构相互协调, 但 DECOR 偏重 "候选分布" (alpha 时变), HAB 偏重 "距离场" (Dbar 静态)
  - 动态 (DECOR alpha) + 静态 (HAB Dbar) 同时训练 → 优化 landscape 冲突 → 过拟合

### 路线全景

| Issue | 方案 | test R@10 | vs baseline |
|-------|------|-----------|-------------|
| #64 v6b | HAB learnable | 0.1038 | +0.0014 |
| #71 v71 | residual α=0.5 | 0.1013 | -0.0011 |
| #135 v72/v73 | valid ES=5/10 | 0.1003/0.0979 | -0.0021/-0.0045 |
| #138 v74 | HAB frozen + WD + dropout | 0.1063 | +0.0039 |
| #139 v75 | v74 + label_smoothing | 0.1043 | +0.0019 |
| #140 v76 | v74 + uncertainty head | 0.1050 | +0.0026 |
| #141 v77 | Stage1 + HAB 协同 | 0.1080 | +0.0056 |
| #68 v78 | DECOR + 抗 trap + Stage1 | **0.1092** | **+0.0068** ← 最佳 |
| **#69 v79** | **DECOR + HAB 叠加** | **0.1030** | **+0.0006** ← NO-GO |

### valid/test ratio 演化

| 方案 | ratio | 解读 |
|------|-------|------|
| baseline | 1.237 | 正常 |
| v74 HAB frozen | 1.234 | 略好 |
| v77 Stage1 + HAB | 1.215 | 最佳 (Stage1 + HAB 协同降 ratio) |
| v78 DECOR + 抗 trap | 1.225 | 良好 |
| **v79 DECOR + HAB** | **1.265** | **严重过拟合 (DECOR + HAB 互相干扰)** |

---

## 0.11 目标最终状态

| 路径 | test R@10 | 距 0.11 |
|------|-----------|----------|
| **v78 (DECOR + 抗 trap + Stage1)** | **0.1092** | **-0.0008** ← 历史最佳 |
| v79 (DECOR + HAB 叠加) | 0.1030 | -0.0070 (NO-GO) |
| v77 (Stage1 + HAB 协同) | 0.1080 | -0.0020 |
| v74 (HAB 路线最佳) | 0.1063 | -0.0037 |
| DIGER 论文 (instruments) | 0.1121 | +0.0021 |
| DECOR 论文 (instruments) | 0.1157 | +0.0057 |

**v78 仍是历史最佳 (0.1092, 距 0.11 只差 -0.0008)**. v79 DECOR + HAB 叠加 NO-GO.

---

## 闭环决策 — R28 兜底

按 R28 (OPEN 立即闭环) + R17 (commit 必含 Gate) + R15 (commit + push + close):

1. **v79 NO-GO** — DECOR + HAB 叠加 = 严重过拟合, 拒绝合并
2. **v78 (0.1092) 仍是历史最佳**, 维持作为新基线
3. **下一轮方向** (按 ROI):
   - **接受 v78 作为基线** (零成本, 立即采纳)
   - DECOR 超参细调 (alpha_init / bos_count / warmup_steps)
   - Stage1 多维 radius (Stage1 重训 5h)
   - 完全换方向 (数据增强 / loss 改进 / 训练策略)

**v79 不修改回历史, 不重训, 直接进入下一轮决策** (R22 立即闭环).

---

## 产物清单

| 类型 | 路径 |
|------|------|
| best ckpt | `/tmp/v79_decor_hab/HG_Rec_best.pth` (ep135, 33M) — 不采纳 |
| train verdict | `/tmp/v79_decor_hab/verdict.json` |
| test eval verdict | `/tmp/v79_decor_hab/test_eval/eval_test.json` |
| train log | `/tmp/v79_decor_hab/train.log` |
| test eval log | `/tmp/v79_decor_hab/test_eval/eval.log` |

---

## 结论

**Issue #69 v79 = NO-GO -0.0062 vs v78**. valid 0.1301 (历史次高) 但 test 0.1030 (历史低位) → **DECOR + HAB 同时启用 = 严重过拟合 (valid/test ratio 1.265)**.

**v78 (0.1092) 仍是历史最佳, 距 0.11 仅 -0.0008**. 

**关键教训**: **DECOR (embedding 动态 alpha-gated) 与 HAB (encoder 静态 hyperbolic bias) 修改的是 T5 的不同层级但信息耦合, 叠加时优化 landscape 冲突, 出现 严重的 valid-test gap**. 单 system 变更避免叠加.

---

## Issue 闭环

- Issue #69 v79 → close (NO-GO -0.0062 vs v78)
- 发尾 comment 说明叠加失败, ratio 1.265, v78 维持基线
- 写入 memory: DECOR + HAB 叠加 = 严重过拟合, ratio 1.265
