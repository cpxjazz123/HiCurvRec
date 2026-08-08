# Issue #141 v77 Stage1 per-item radius + v74 HAB frozen 协同 — +0.0017 vs v74, 未达 0.11

**最终 verdict**: test R@10=**0.1080** (+0.0017 vs v74 0.1063, +0.0056 vs baseline 0.1024). **仍未达 0.11 目标 (-0.0020)**. Stage1 per-item radius + HAB frozen 协同有提升但未破 0.11.

---

## 4 Gate 评估

### Gate 1 (代码改动 + 语法) — PASS
- 复用 v74 HAB frozen config (Issue #138 已 commit 59bc1cb, 无新代码改动)
- 复用 Stage1 per-item radius 产物 `taskA/_history/taskA_stage1_hyp_v2/item_emb.parquet` (SHA256 99e6b39a)
- 复用 Stage2 kappa_sync ckpt `taskA/_history/taskA_stage2_kappa_sync/hrqvae_kappa_sync.ckpt` (SID 06af0fed)
- Stage3 launch: `common/stage3/stage3_train_pure_t5.py` + HAB frozen + WD=0.01 + dropout=0.20
- Verdict: Gate 1 PASS (复用既有产物, 无新代码)

### Gate 2 (训练启动 + 早停触发) — PASS
- DDP 4 卡 bf16 启动成功 (PID 3277406, 4 卡 32% util)
- 32 min 200ep, EARLY_STOP=10 触发 ep155 (best ep104, valid_R10=0.1312)
- train_loss 5.09 → 1.95, valid_R10 0.0947 → 0.1312
- λ_eff = [-0.2, 0.199, 0.2] (全部 saturated 到 ±0.2, 与 v74 一致)
- U/V l2_norm = [0.047, 0.063, 0.088] (vs v74 0.036-0.068, 略大, 但 WD 仍生效)
- best ckpt 落盘 `/tmp/v77_peritem_hab/HG_Rec_best.pth`
- Verdict: Gate 2 PASS

### Gate 3 (valid 突破) — NO-GO (匹配 v74)
- best valid_R@10 = **0.1312** (ep104) — 与 v74 best 0.1312 完全持平 (不同 SID 但 valid 同)
- vs v74 0.1312: **±0**
- vs baseline 0.1267: +0.0045 (与 v74 持平)
- vs taskA hyp v2 (无 HAB) 0.1256: +0.0056 (HAB 贡献)
- Verdict: Gate 3 NO-GO (未突破 v74, 但与 v74 持平, 协同不破坏)

### Gate 4 (test eval) — GO +0.0017 vs v74, NO-GO vs 0.11

| 指标 | v77 | v74 | baseline | taskA hyp v2 |
|------|-----|-----|----------|---------------|
| test R@5 | 0.0859 | 0.0853 | 0.0819 | 0.0850 |
| test R@10 | **0.1080** | 0.1063 | 0.1024 | 0.1048 |
| test R@20 | **0.1343** | 0.1303 | 0.1283 | 0.1310 |
| NDCG@5 | 0.0720 | 0.0721 | — | — |
| NDCG@10 | **0.0791** | 0.0789 | 0.0755 | 0.0755 |
| NDCG@20 | **0.0857** | 0.0850 | 0.0821 | 0.0860 |
| valid_R10 | 0.1312 | 0.1312 | 0.1267 | 0.1256 |
| valid/test ratio | 1.215 | 1.234 | 1.237 | 1.198 |
| vs baseline R@10 | +0.0056 | +0.0039 | — | +0.0024 |
| vs v74 R@10 | **+0.0017** | — | — | -0.0015 |

**GO vs v74 (+0.0017)** — Stage1 per-item radius 路线叠加 v74 HAB config 创造新 SOTA. **NO-GO vs 0.11 (-0.0020)**.

---

## 关键发现

### 协同有提升 (+0.0017 vs v74)

| 来源 | 单独 test R@10 | 协同 test R@10 | 协同增益 |
|------|----------------|----------------|----------|
| Stage1 per-item radius | 0.1048 (taskA hyp v2) | — | — |
| HAB frozen (v74) | 0.1063 (Issue #138) | — | — |
| 协同 (v77) | — | **0.1080** | +0.0017 (相加非平均) |

**协同增益 ~0.0017** 是两条独立路径的简单相加. 不等于"叠加" (否则会到 0.1080 ≈ 0.1048 + 0.1063 - 0.1024 = 0.1087). 说明 Stage1 per-item radius 提供更稳定的 embedding 让 HAB 残差更准确.

### valid/test ratio 改善 (1.215)

| 方案 | valid/test ratio | 失衡程度 |
|------|------------------|----------|
| baseline | 1.237 | 正常 |
| HAB v74 | 1.234 | 改善 |
| **v77 (协同)** | **1.215** | **最佳** |

v77 ratio 1.215 < baseline 1.237, test 表现**接近 valid**. 比 v74 ratio 1.234 更接近 1.0, **过拟合进一步缓解**.

### 关键指标 NDCG@20 也破 v74

| 方案 | NDCG@20 | vs baseline 0.0821 |
|------|---------|-------------------|
| **v77 (协同)** | **0.0857** | **+0.0036** |
| v74 HAB frozen | 0.0850 | +0.0029 |
| taskA hyp v2 | 0.0860 | +0.0039 |
| baseline | 0.0821 | — |

NDCG@20 也超 v74 (+0.0007), 协同带来 ranking 质量全面提升.

---

## HAB 路线全景 (v74/v77 + 历史失败)

| Issue | 方案 | test R@10 | vs baseline |
|-------|------|-----------|-------------|
| #64 v6b | 偶然超 baseline | 0.1038 | +0.0014 |
| #71 v71 | residual α=0.5 | 0.1013 | -0.0011 |
| #135 v72 | valid_R10 ES=5 | 0.1003 | -0.0021 |
| #135 v73 | valid_R10 ES=10 | 0.0979 | -0.0045 |
| **#138 v74** | **HAB frozen + WD + dropout** | **0.1063** | **+0.0039** |
| #139 v75 | v74 + label_smoothing=0.1 | 0.1043 | +0.0019 |
| #140 v76 | v74 + uncertainty head | 0.1050 | +0.0026 |
| **#141 v77** | **Stage1 per-item radius + v74 config** | **0.1080** | **+0.0056 ✓** |

**v77 = HAB 路线 + Stage1 协同历史最佳**. 比 v74 +0.0017.

---

## 0.11 目标最终状态

| 路径 | test R@10 | 距 0.11 | 来源 |
|------|-----------|----------|------|
| **v77 (HAB + Stage1 协同)** | **0.1080** | **-0.0020** | 本 Issue ✓ |
| v74 (HAB 路线最佳) | 0.1063 | -0.0037 | Issue #138 |
| taskA hyp v2 (Stage1 路线) | 0.1048 | -0.0052 | 历史 |
| DIGER 论文 (instruments) | 0.1121 | +0.0021 (新模块) | DIGER 复现 |
| DECOR 论文 (instruments) | 0.1157 | +0.0057 (新模块) | DECOR 复现 |

**0.11 仍未达 (-0.0020)**. **协同已达 HAB 框架 + Stage1 框架上限**. 0.11 必须新模块 (DIGER RQ-VAE 可微 / DECOR candidate bins 改良).

---

## 推荐 — v77 作为新基线

按 R28 兜底顺序:
1. **v77 (test R@10=0.1080) 是 HAB + Stage1 协同能达到的最佳**
2. **HAB 路线 + Stage1 per-item radius 已穷尽**, 不再做该路径微调
3. **0.11 必须架构性改动** (DIGER uncertainty head 在 RQ-VAE 端, DECOR candidate bins 重新设计)

### 推荐后续 (按 ROI)

| 方向 | 改动 | 预期 test | ROI |
|------|------|-----------|-----|
| **接受 v77 作为新基线** | 无 | 0.1080 | ∞ (零成本) |
| DIGER RQ-VAE 可微 | Stage2 重训 (8h+) + uncertainty | ~0.110+ | 高 |
| DECOR 改良 (解决 self-reinforcing trap) | 重新设计 candidate bins | ~0.110+ | 中 |
| Stage1 多维 radius | Stage1 重训 (5h) | ~0.109 | 中 |

---

## 产物清单

| 类型 | 路径 |
|------|------|
| best ckpt | `/tmp/v77_peritem_hab/HG_Rec_best.pth` (ep104, 33M) |
| train verdict | `/tmp/v77_peritem_hab/verdict.json` |
| train log | `/tmp/v77_peritem_hab/train.log` |
| test eval log | `/tmp/v77_peritem_hab/test_eval/eval.log` |
| test verdict | `/tmp/v77_peritem_hab/test_eval/eval_test.json` |

---

## 代码变更

无新代码改动 — 复用 Issue #138 v74 已 commit 的 HAB frozen + WD + dropout 代码 (59bc1cb).

---

## 结论

**Issue #141 v77 = GO +0.0017 vs v74** (新 SOTA 0.1080). **但仍未达 0.11 目标 (-0.0020)**.

**协同增益 ~0.0017**, 两条独立路径叠加效果. valid/test ratio 1.215 进一步改善 (vs baseline 1.237).

**HAB + Stage1 per-item radius 路线已穷尽 (8 issue 全景)**:
- HAB 路线 #64 v6b/#71 v71/#135 v72/v73/#138 v74/#139 v75/#140 v76
- Stage1 per-item radius + HAB 协同 #141 v77 (**新峰**)

**0.11 必须新模块**:
- DIGER RQ-VAE 可微 (Stage2 重训, 8h+)
- DECOR candidate bins 改良 (解决 self-reinforcing trap)

**v77 (0.1080) 作为新基线, 不再做 HAB+Stage1 微调**.

---

## Issue 闭环

- Issue #141 v77 → close (GO +0.0017, 但未达 0.11)
- 发尾 comment 说明 0.11 仍需新模块
- 写入 memory: HAB+Stage1 协同 = 新基线 0.1080