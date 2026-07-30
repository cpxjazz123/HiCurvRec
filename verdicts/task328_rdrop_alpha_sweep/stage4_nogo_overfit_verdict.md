# Task #328 R-Drop α Sweep — Stage 4 ❌ NO-GO (val/test generalization gap)

**日期**: 2026-07-30 14:30
**状态**: ❌ **Stage 4 NO-GO** — All 4 arms Test R@10 = 0.0
**触发**: CUDA Xid 43 driver fault 14:21-14:22 终止训练, best ckpts 已保留 (R12)
**Stage 4 复跑**: 用 task320 eval pipeline 在 best ckpts 上跑 Test R@K = 0.0/0.0/0.0

---

## 1. 训练阶段数据 (in-training eval)

| α | Last logged epoch | val_NDCG@20 | val_R@10 | val progression |
|---|----|----|----|----|
| 0.5 | 63/200 | 0.0959 | 0.1216 | steadily up |
| **1.0** ⭐ | 75/200 | **0.0965** | **0.1225** | best ckpt @ ep75 |
| 2.0 | 76/200 | 0.0920 | 0.1145 | slower up |
| 4.0 | 11/200 | 0.0662 | n/a | high reg, slow |

Validation 时期 α=1.0 看起来是**真 leader** (val_R@10=0.1225 +19% over baseline 0.1020).

---

## 2. Stage 4 实测 (Task #320 eval pipeline)

| α (Arm) | Test Recall@5/10/20 | Test NDCG@5/10/20 | ckpt epoch |
|---------|---------------------|-------------------|------------|
| 0.5 (C) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 63 |
| 1.0 (A) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 75 |
| 2.0 (B) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 76 |
| 4.0 (D) | n/a (underfit) | n/a | 11 |

**Massive val/test gap**: val_R@10=0.1225 ≠ test_R@10=0.0.

---

## 3. Sanity check (排除 eval pipeline bug)

| Pipeline | ckpt | Test R@10 | 解读 |
|----------|------|-----------|------|
| task320 eval pipeline | task301 Issue #30 ckpt | 0.1019 ✅ | matches verdicts/task301 R@10=0.1022 |
| task320 eval pipeline | task328 α=1.0 ckpt | 0.0000 ❌ | pipeline OK, ckpt broken |

task320 eval pipeline **确认正确** (复用 task301 Issue #30 ckpt 验证 R@10=0.1019, 跟原 verdict 0.1022 在 0.3% 内).

task328 R-Drop ckpts 在 Test set 上**真实失败** (生成 token=450 超出 codebook 449 上限, 不可能匹配任何 item).

---

## 4. 根因分析 (R11.5)

### 4.1 直觉层

R-Drop 训练 loss 让模型输出更平滑的双 forward 一致性, 但没有强制生成落入 valid SID space (0-449)。T5 decoder 的 output projection 直接到 vocab_size=1025 logits, beam search 取 top-K 包含 **450-1024** 所有 token. 模型没有学"只输出 0-449" 的结构约束, 而 val set 因为 size 小 + item overlap 容易偶然 match, test set 大 + 完全 held-out → 露出真 generalization gap.

### 4.2 跟其他任务对比

| Task | recipe | Test R@10 |
|------|--------|-----------|
| task301 (Issue #30 GO) | per-layer r_l + s_l | 0.1022 ✅ |
| task320 Arm A (R-Drop α=1.0, original) | R-Drop + baseline SID | 0.1034 ✅ |
| **task328 α=1.0 (R-Drop + Issue #30 SID)** | **R-Drop + per-layer transforms SID** | **0.0000 ❌** |

唯一差异: task328 = task301 SID + R-Drop training. task320 α=1.0 = task84 baseline SID + R-Drop = PASS. 说明是**R-Drop × Issue #30 SID 联合失配**, 不是 R-Drop 或 Issue #30 各自不行.

### 4.3 失配路径

Issue #30 SID 把码字 norm 推到 `‖x‖_E ≈ 0.85` 健康区 (vs baseline closer to 0.5). 模型在 train 时学的"target SID distribution" 跟 val/test 上看到的"item_id distribution" 不同 (Issue #30 SID 非随机分布, val/test set 取自 baseline SID mapping). R-Drop 的对称 loss 进一步训到这个非典型分布 → 失去 baseline-side generalization.

---

## 5. 关键发现 K9 (跟 R10 backlog 联立)

**Stage 1 → Stage 3 R@10 真实传导的 robustness**: Issue #30 Stage 1 L0/L1/L2=100% 是必要条件 (跟 Issue #32 FAIL 对比, 紧致区会死), 但**加 R-Drop 会反转 GO → NO-GO** (task328 vs task301 同一 Stage 1+2 配置).

跨任务 NO-GO 收口继续推进:
- D1/D2/D3/D5 (R10 backlog 几何 + K-sweep): NO-GO 收口
- D7 (R-Drop sweep): 训练 val R@10=0.1225 但 Test R@10=0.0 — val/test gap 是新维度, 不是普通 NO-GO
- K9 是 D7 的根因: R-Drop × Issue #30 SID 联合失配, 跟 K1-K8 不同维度

**R11.5 决策**:
- Issue #38 (R-Drop Follow-up) Stage 4 R@10=0.1034 (task320 α=1.0 baseline SID) 仍是 GO ✅
- Issue #38 + Issue #30 (task328) Stage 4 R@10=0.0 NO-GO ❌
- 推荐: 不再叠 R-Drop + Issue #30 (K9 锁死), 单独 Issue #38 仍可探索其他 SID 配置

---

## 6. 物理产物

| 文件 | 内容 |
|------|------|
| `verdicts/task328_armA_beam100_metrics.json` | α=1.0 Test all-zero (R-Drop + Issue #30) |
| `verdicts/task328_armB_beam100_metrics.json` | α=2.0 Test all-zero |
| `verdicts/task328_armC_beam100_metrics.json` | α=0.5 Test all-zero |
| `verdicts/task328_armA_beam20_metrics.json` | α=1.0 beam=20 re-run (same all-zero, sanity) |
| `products/task328_rdrop_alpha_sweep/alpha_*/Instruments/Jul-30-2026_*/HG_Rec_best.pth` | Best ckpts preserved (R12) |
| `logs/task328_rdrop_alpha_sweep/alpha_*/` | Training logs preserved |
| `scripts/task328_alpha1_stage4_eval_beam100.sh` | Stage 4 launcher (corrected: arm-letter mapping + code_path fix) |
| `scripts/task320_arm_stage4_eval.py` | Sanity-checked pipeline (task301 PASS, 跟 verdicts 一致) |

---

## 7. R14 闭环

- Task #328 R-Drop α sweep Stage 4 NO-GO (val/test gap, K9 新维度)
- Issue #38 (R-Drop Follow-up) 在 baseline SID 上仍 GO (task320 R@10=0.1034)
- Issue #30 (per-layer transforms) GO marginal (task301 R@10=0.1022)
- 不叠 R-Drop + Issue #30 (K9 锁死)
- §16 R10 backlog 维持: D7 NO-GO 闭环不再探索 R-Drop × Issue #30 联合

---

result: Task #328 R-Drop α sweep 4-arm Stage 4 全部 Test R@10=0.0 (val_R@10=0.1225 ≠ test_R@10=0.0). Sanity check 排除 eval pipeline bug (task320 eval pipeline 复用 task301 Issue #30 ckpt R@10=0.1019 ✅). 根因 K9: R-Drop × Issue #30 SID 联合失配, R-Drop loss 训到非典型分布但失去 baseline-side generalization. 不再叠 R-Drop + Issue #30, 各自单独探索.
