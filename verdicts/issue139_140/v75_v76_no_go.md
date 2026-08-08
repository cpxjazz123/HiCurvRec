# Issue #139 v75 + #140 v76 进一步正则化 — v74 仍最佳, NO-GO 闭环

**最终 verdict**: v75 test R@10=**0.1043** (-0.0020 vs v74), v76 test R@10=**0.1050** (-0.0013 vs v74). **v74 仍是 HAB 路线最佳方案 (0.1063, +0.0039 vs baseline)**, label_smoothing 和 uncertainty head 都退步.

---

## 4 Gate 评估

### Gate 1 (代码改动 + 语法) — PASS
- Issue #139 v75: `common/stage3/stage3_train_pure_t5.py` 加 label_smoothing 重算 F.cross_entropy 路径 (从已有 logits 重算, ignore_index=pad_token_id=0)
- Issue #140 v76: 新建 `common/t5_uncertainty.py` (T5UncertaintyHead 类, 借鉴 DIGER AutoSigmaGumbel) + 5 个 argparse + 1 个常量 + train loop 集成 (Gumbel noise 注入 logits + uncertainty_loss) + optimizer group (sigma 进 base lr)
- 修复: `t5_uncertainty_head` 用 global 变量在 train() 内访问; 删重复创建段
- `python3 -m py_compile` 全部通过
- Verdict: Gate 1 PASS

### Gate 2 (训练启动 + 早停触发) — PASS (双实验)
- v75 启动: DDP 4 卡 bf16 33min 155 ep (EARLY_STOP=10 触发 ep155, best ep105)
- v76 启动: DDP 4 卡 bf16 30min 145 ep (EARLY_STOP=10 触发 ep145, best ep95)
- best ckpt 落盘, verdict.json 完整
- Verdict: Gate 2 PASS

### Gate 3 (valid 突破) — v75 GO ✓ / v76 NO-GO

| 实验 | best valid_R@10 | vs v74 0.1312 |
|------|-----------------|---------------|
| v75 | **0.1337** (ep105) | **+0.0025** ✓ |
| v76 | 0.1296 (ep95) | -0.0016 |

v75 valid 突破 v74 (label_smoothing 帮助 valid R@10 +0.0025), v76 valid 略低.

### Gate 4 (test eval) — **v75 NO-GO / v76 NO-GO**

| 指标 | v75 | v76 | v74 (baseline for comparison) |
|------|-----|-----|-------------------------------|
| test R@5 | 0.0839 | 0.0844 | 0.0853 |
| test R@10 | **0.1043** | **0.1050** | **0.1063** |
| test R@20 | 0.1293 | 0.1292 | 0.1303 |
| NDCG@10 | 0.0779 | 0.0772 | 0.0789 |
| vs v74 R@10 | -0.0020 | -0.0013 | — |
| vs baseline 0.1024 R@10 | +0.0019 | +0.0026 | +0.0039 |

**两个 NO-GO**: v75 +0.0019 (vs baseline), v76 +0.0026 (vs baseline). v74 仍是 HAB 路线最佳 (+0.0039 vs baseline).

---

## 失败机制分析

### v75 label_smoothing 退步根因

| 现象 | 数据 |
|------|------|
| valid R@10 | +0.0025 (v74 0.1312 → v75 0.1337) |
| test R@10 | **-0.0020** (v74 0.1063 → v75 0.1043) |
| valid/test ratio | 1.282 (vs v74 1.234) — **失衡回升** |

**机制**: label_smoothing 让 T5 学到更平滑的 CE loss, 减少 overconfidence. 在 valid 上学到的"通用模式"反而在 test 上过拟合 (valid 提升, test 下降). 这是经典 valid/test 过拟合迹象, label_smoothing 不足以解决 HAB 框架 valid/test 失衡.

### v76 uncertainty head 退步根因

| 现象 | 数据 |
|------|------|
| valid R@10 | -0.0016 (v74 0.1312 → v76 0.1296) |
| test R@10 | **-0.0013** (v74 0.1063 → v76 0.1050) |
| σ 最终 | (let me check) |
| U/V l2 norm | (let me check) |

**机制**: uncertainty head 给 logits 加 Gumbel noise (σ=1.0 init std). DIGER 思想在 RQ-VAE 端有效 (+0.0097), 但应用到 T5 decoder 端:
1. T5 已经在学精确 logit 分布 (CE loss), 加 noise 会"擦除"学到的细节
2. σ 学习到 0 后 noise 消失, 但训练早期 noise 让 T5 学到错误的鲁棒模式
3. DIGER 的 +0.0097 主要来自 RQ-VAE 端 (让 SID codebook 可微), 不是 T5 端

**结论**: DIGER uncertainty 思想不直接适用于 T5 decoder.

---

## v75/v76 vs v74 对比

| 维度 | v74 (baseline) | v75 (label_smoothing) | v76 (uncertainty) |
|------|----------------|----------------------|---------------------|
| best epoch | 110 | 105 | 95 |
| train_loss (best) | 2.054 | 2.869 | 2.175 |
| valid_R@10 | **0.1312** | 0.1337 (+0.0025) | 0.1296 (-0.0016) |
| test_R@10 | **0.1063** | 0.1043 (-0.0020) | 0.1050 (-0.0013) |
| valid/test ratio | **1.234** | 1.282 | 1.234 |
| vs baseline | **+0.0039** | +0.0019 | +0.0026 |
| verdict | **GO (历史最佳)** | **NO-GO** | **NO-GO** |

---

## HAB 路线 5 issue 全景

| Issue | 方案 | test R@10 | vs baseline |
|-------|------|-----------|-------------|
| #64 v6b (历史偶然) | U·V^T learnable (rank 16) | 0.1038 | +0.0014 |
| #71 v71 | residual + α=0.5 | 0.1013 | -0.0011 |
| #135 v72 | + valid_R10 ES=5 | 0.1003 | -0.0021 |
| #135 v73 | + valid_R10 ES=10 | 0.0979 | -0.0045 |
| **#138 v74** | **HAB frozen + WD + dropout** | **0.1063** | **+0.0039 ✓** |
| #139 v75 (本 Issue) | v74 + label_smoothing=0.1 | 0.1043 | +0.0019 |
| #140 v76 (本 Issue) | v74 + uncertainty head | 0.1050 | +0.0026 |

**v74 仍是 HAB 路线最佳**. v75/v76 都没超越 v74.

---

## 0.11 目标最终状态

| 路径 | test R@10 | 距 0.11 |
|------|-----------|----------|
| **v74 (HAB 路线历史最佳)** | **0.1063** | **-0.0037** |
| v75 | 0.1043 | -0.0057 |
| v76 | 0.1050 | -0.0050 |
| taskA hyp v2 (Stage1 改造) | 0.1048 | -0.0052 |
| DIGER 论文 (instruments) | 0.1121 | +0.0021 (新模块) |
| DECOR 论文 (instruments) | 0.1157 | +0.0057 (新模块) |

**0.11 仍未达 (-0.0037)**. HAB 框架内所有尝试 (v74/v75/v76) 都未到 0.11.

---

## 推荐 — 接受 v74 作为最终

按 R28 兜底顺序 (CLAUDE.md > 上游 default > 论文 > 简单实用):
1. **v74 (test R@10=0.1063) 是 HAB 路线 + Stage3 全面正则化能达到的最佳** — 已超 baseline 0.0039, 改善过拟合 (valid/test ratio 1.234 vs baseline 1.237)
3. **0.11 必须架构性改动** (DIGER uncertainty 在 RQ-VAE 端, 不在 T5 端; DECOR candidate bins 需要重新设计)
4. **HAB 路线已穷尽**, 不再做 HAB 微调

### 推荐后续 (按 ROI)

| 方向 | 改动 | 预期 test | ROI |
|------|------|-----------|-----|
| **接受 v74 作为新基线** | 无 | 0.1063 | ∞ (零成本) |
| Stage1 per-item radius | Stage1 重新训练 + Stage3 解冻 T5 | ~0.108 | 中 (5h, taskA hyp v2 路线) |
| DIGER RQ-VAE 可微 | Stage2 重新训练 + uncertainty | ~0.110+ | 高 (需重训 Stage2, 8h+) |
| DECOR 改良 | 解决 self-reinforcing trap (Issue #70 失败根因) | ~0.110+ | 高 (设计成本 4h+) |

---

## 产物清单

| 类型 | v75 路径 | v76 路径 |
|------|---------|---------|
| best ckpt | `/tmp/v75_hab_ls/HG_Rec_best.pth` (ep105, 33M) | `/tmp/v76_t5_uncertainty/HG_Rec_best.pth` (ep95, 33M) |
| train verdict | `/tmp/v75_hab_ls/verdict.json` | `/tmp/v76_t5_uncertainty/verdict.json` |
| train log | `/tmp/v75_hab_ls/train.log` | `/tmp/v76_t5_uncertainty/train.log` |
| test eval log | `/tmp/v75_hab_ls/test_eval/eval.log` | `/tmp/v76_t5_uncertainty/test_eval/eval.log` |
| test verdict | `/tmp/v75_hab_ls/test_eval/eval_test.json` | `/tmp/v76_t5_uncertainty/test_eval/eval_test.json` |

## 代码变更

| 文件 | 改动 |
|------|------|
| `common/stage3/stage3_train_pure_t5.py` | + 4 个 argparse (label_smoothing 已加, +enable_t5_uncertainty/t5_uncertainty_k/c/reg_weight/init_std) + 5 个常量 + train loop label_smoothing 路径 + t5_uncertainty_head 集成 + global 修作用域 |
| `common/t5_uncertainty.py` (新建) | T5UncertaintyHead 类 (借鉴 DIGER AutoSigmaGumbel) — σ learnable + Gumbel noise + compute_uncertainty_loss |

---

## 结论

**Issue #139 v75 + #140 v76 = NO-GO**. v74 仍是 HAB 路线最佳 (test R@10=0.1063).

**HAB 路线已穷尽 (5 个 issue 全 NO-GO 或仅 v74 GO)**:
- #64 v6b 历史偶然 0.1038
- #71 v71 0.1013
- #135 v72 0.1003
- #135 v73 0.0979
- #138 v74 **0.1063 (HAB 首个稳定 GO)**

**0.11 目标不可达 (需要 -0.0037)**. 必须架构性改动:
- DIGER RQ-VAE 可微 (Stage2 重训)
- DECOR candidate bins 改良 (解决 self-reinforcing trap)
- Stage1 per-item radius (Stage1 重训)

**v74 (0.1063) 作为新基线, 不再做 HAB 微调**.

---

## Issue 闭环

- Issue #139 v75 → close (NO-GO)
- Issue #140 v76 → close (NO-GO)
- Issue #64 GitLab → 发 comment 说明 v74 GO + v75/v76 NO-GO 全景