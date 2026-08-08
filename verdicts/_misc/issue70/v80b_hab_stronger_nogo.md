# Issue #70 v80b HAB 强度加强 — NO-GO

**最终 verdict**: test R@10 = **0.1059** (vs v77 0.1080 **-0.0021**, vs baseline 0.1024 **+0.0035**). **NO-GO**. HAB 双曲 bias 强度加强 (λ_max 0.20→0.30 + residual_alpha_init -20→-25) **反而退步**, valid/test ratio 1.240 (vs v77 1.215 / v74 1.234).

---

## 4 Gate 评估

### Gate 1 (代码改动 + 语法) — PASS
- v80b = v77 之上 HAB 强度加强 (Stage3 命令行覆盖):
  - `--hyperbolic_attn_bias --hab_lambda_max 0.30` (vs v77 0.20)
  - `--enable_residual_hab --residual_alpha_init -25.0` (vs v77 -20.0)
  - `--stage3_weight_decay 0.01 --stage3_dropout 0.2` (v74 reg)
- SID 06af0fed (taskA_stage2_hyp_v2_capmatch_1000ep) — 与 v77 相同
- 无代码改动, 仅命令行参数变化
- py_compile 通过 (没改代码)
- Verdict: Gate 1 PASS

### Gate 2 (训练启动 + 早停触发) — PASS
- DDP 4 卡 bf16 启动成功 (PID 3414593)
- 36 min 175 epochs (ES=10/10 at ep175)
- train_loss 5.02 → 1.88
- valid_R10: 0.0951 (ep5) → 0.1313 (ep124, BEST) → 0.1235 (ep175 ES)
- best ckpt 落盘 `/tmp/v80b_hab_stronger/HG_Rec_best.pth`
- Verdict: Gate 2 PASS

### Gate 3 (valid 突破) — **NO-GO -0.0025 vs v77**
- best valid_R@10 = **0.1313** (ep124)
- vs v77 0.1312: +0.0001 (持平)
- vs v74 0.1312: +0.0001 (持平)
- vs baseline 0.1267: +0.0046
- vs v78 0.1338: **-0.0025** (略低)
- Verdict: Gate 3 **NO-GO -0.0025 vs v78, 持平 v77**

### Gate 4 (test eval) — **NO-GO -0.0021 vs v77**

| 指标 | v80b | v77 | v74 | baseline |
|------|------|-----|-----|----------|
| test R@5 | 0.0853 | 0.0859 | 0.0853 | 0.0819 |
| test R@10 | **0.1059** | 0.1080 | 0.1063 | 0.1024 |
| test R@20 | 0.1319 | 0.1343 | 0.1303 | 0.1283 |
| NDCG@5 | 0.0720 | 0.0720 | 0.0721 | — |
| NDCG@10 | 0.0786 | 0.0791 | 0.0789 | 0.0755 |
| NDCG@20 | 0.0852 | 0.0857 | 0.0850 | 0.0821 |
| valid_R10 | 0.1313 | 0.1312 | 0.1312 | 0.1267 |
| **valid/test ratio** | **1.240** | 1.215 | 1.234 | 1.237 |
| vs baseline R@10 | +0.0035 | +0.0056 | +0.0039 | — |
| vs v77 R@10 | **-0.0021** | — | -0.0017 | -0.0056 |
| vs 0.11 | -0.0041 | -0.0020 | -0.0037 | -0.0076 |

**NO-GO**: v80b test 0.1059 < v77 0.1080 (-0.0021), 退步. **HAB 强度加强 = 过拟合加剧** (ratio 1.240 > v77 1.215).

---

## 关键发现

### HAB 强度加强 = NO-GO

| 方案 | λ_max | residual_alpha_init | valid | test | ratio |
|------|-------|---------------------|-------|------|-------|
| baseline | — | — | 0.1267 | 0.1024 | 1.237 |
| v74 HAB frozen | 0.20 | -20.0 | 0.1312 | 0.1063 | 1.234 |
| v77 Stage1 + HAB | 0.20 | -20.0 | 0.1312 | 0.1080 | 1.215 |
| **v80b HAB 加强** | **0.30** | **-25.0** | **0.1313** | **0.1059** | **1.240** ← NO-GO |

**v80b valid 几乎不变 (0.1313 vs v77 0.1312), 但 test 退步 (-0.0021)**. HAB 强度加强 → valid 难涨, test 跌 → ratio 1.240 恶化.

### HAB 强度对 test 的影响

| λ_max | residual_alpha_init | test R@10 |
|-------|---------------------|-----------|
| 0.10 (v3 早期) | — | 0.1024 |
| 0.20 (v74/v77) | -20.0 | 0.1063/0.1080 |
| **0.30 (v80b)** | **-25.0** | **0.1059** ← 退步 |

**HAB λ_max=0.20 是甜点**. 加大到 0.30 反而退步. 减小到 0.10 也退步 (历史 v3).

### 双曲 bias 的过拟合机制

- **λ_max 太小 (0.10)**: 双曲信号太弱, 接近 baseline
- **λ_max=0.20 (sweet spot)**: 双曲信号适中, T5 学到曲率几何
- **λ_max=0.30 (v80b NO-GO)**: 双曲信号过强, 训练时 T5 把几何距离当成主导信号, 但 valid/test 上的真实相关性不匹配 → 严重过拟合
- **residual_alpha_init 更负 (-25)**: gate sigmoid 更尖锐, 接近 frozen bias, 失去可学习性 → 加剧 λ_max 增大的影响

**HAB frozen bias + Stage1 per-item radius 协同** 已在 v77 (λ_max=0.20) 达到甜点. v80b 加强反而破坏平衡.

---

## 路线全景 (曲率路线)

| Issue | 方案 | test R@10 | vs baseline | ratio |
|-------|------|-----------|-------------|-------|
| baseline | T5 only | 0.1024 | — | 1.237 |
| #138 v74 | HAB frozen + WD + dropout | 0.1063 | +0.0039 | 1.234 |
| #141 v77 | Stage1 per-item radius + v74 | **0.1080** | **+0.0056** | 1.215 ← 曲率路线最佳 |
| **#70 v80b** | **v77 + HAB 加强** | **0.1059** | **+0.0035** | **1.240** ← NO-GO |

---

## 0.11 目标状态

| 路径 | test R@10 | 距 0.11 |
|------|-----------|----------|
| **v77 (Stage1 + HAB)** | **0.1080** | **-0.0020** ← 曲率路线最佳 |
| v80b (HAB 加强) | 0.1059 | -0.0041 (NO-GO) |
| v78 (DECOR + 抗 trap) | 0.1092 | -0.0008 |
| DIGER 论文 | 0.1121 | +0.0021 |
| DECOR 论文 | 0.1157 | +0.0057 |

**v77 仍是曲率路线最佳 (test 0.1080)**. v80b 验证 HAB 强度不能简单加强.

---

## 闭环决策 — R28 兜底

按 R28 (OPEN 立即闭环) + R17 (commit 必含 Gate) + R15 (commit + push + close):

1. **v80b NO-GO** — HAB 强度加强反而退步, 拒绝合并
2. **v77 (0.1080) 仍是曲率路线最佳**, 维持作为新基线
3. **下一轮方向** (按 ROI):
   - **接受 v77 作为曲率路线基线** (零成本, 立即采纳)
   - **Stage1 per-item radius 强化** (R_MAX 0.99 → 0.95 / sigmoid temp 3 → 5)
   - **Stage2 κ 加强** (capmatch 不同层权重)
   - **Stage1 多维 radius** (per-dim learnable)
   - 完全换方向 (数据增强 / loss 改进 / 训练策略)

**v80b 不修改回历史, 不重训, 直接进入下一轮决策** (R22 立即闭环).

---

## 产物清单

| 类型 | 路径 |
|------|------|
| best ckpt | `/tmp/v80b_hab_stronger/HG_Rec_best.pth` (ep124, 33M) — 不采纳 |
| train verdict | `/tmp/v80b_hab_stronger/verdict.json` |
| test eval verdict | `/tmp/v80b_hab_stronger/test_eval/eval_test.json` |
| train log | `/tmp/v80b_hab_stronger/train.log` |
| test eval log | `/tmp/v80b_hab_stronger/test_eval/eval.log` |

---

## 结论

**Issue #70 v80b = NO-GO -0.0021 vs v77**. HAB 强度加强 (λ_max 0.20→0.30 + residual_alpha_init -20→-25) 反而退步, valid/test ratio 1.240 恶化.

**v77 (0.1080) 仍是曲率路线最佳, 距 0.11 仅 -0.0020**. 

**关键教训**: **HAB λ_max=0.20 + residual_alpha_init=-20 已是曲率路线甜点**. 加大 λ_max 或更负 residual_alpha_init 都破坏 valid/test 平衡, 加剧过拟合 (ratio 1.240).

---

## Issue 闭环

- Issue #70 v80b → close (NO-GO -0.0021 vs v77)
- 发尾 comment 说明 HAB 强度加强失败, ratio 1.240, v77 维持基线
- 写入 memory: HAB λ_max 0.20 是甜点, 加强 NO-GO
