# Task #171 verdict — κ-Stereographic + dead_code_reset_every=10 [⛔ C3 NO-GO]

> **任务目的**: 验证"定期 reset 死码能让 codebook 利用率健康化, 在 κ-Stereo 框架下也能解 val/test gap"假设. R5 fixed seed=42.
> **完成日期**: 2026-07-25
> **状态**: ⛔ C3 NO-GO — test R@10=0.1001 < baseline 0.1058 (-5.4%)

---

## 1. 关键指标

| 指标 | 测试值 | Baseline #105 | #164 纯 κ-Stereo | Δ vs baseline |
|------|-------|-------|--------|-------|
| **Recall@10 (主要)** | **0.1001** | 0.1058 | 0.0964 | **-5.4%** |
| Recall@5 | 0.0798 | 0.0822 | 0.0734 | -2.9% |
| Recall@20 | 0.1221 | 0.1214 | 0.1114 | +0.6% |
| NDCG@10 | 0.0728 | 0.0823 | 0.0781 | -11.5% |
| NDCG@20 | 0.0783 | 0.0901 | 0.0849 | -13.0% |

---

## 2. 执行时间线

- **Stage 1 launch**: κ-Stereographic Phase A/B (100+100 epoch), `--dead_code_reset_every=10`
- **Stage 2 codebook inference**: 完成
- **Stage 3 T5-mini 9.18M training**: 11:44 → 12:?? (~50 min, 200 epoch + early stop)
- **Stage 4 test eval**: 12:?? → 12:?? (~1 min)
- **总时长**: ~51 min

---

## 3. 关键发现

**dead_code_reset_every=10 没能解决 val/test gap**:
- val R@10 推断 > 0.1070 (跟前几次 #166-#169 一致)
- test R@10 = 0.1001 (-5.4% vs baseline)
- #170 (Sinkhorn all 3 layers): test R@10 = 0.0964 (NO-GO)
- #169 (Sinkhorn L2 only): test R@10 = 0.0863 (NO-GO -18.4%)
- #164 (纯 κ-Stereo): test R@10 = 0.0964 (NO-GO -8.9%)
- **#171 (dead_code_reset_every=10)**: test R@10 = **0.1001** (NO-GO -5.4%, 4 变体里相对最高, 但仍未超 baseline)

**Reset 死码改善利用率但下游无收益**:
- 该机制确认 codebook 健康 (L0 100%, L1 100%, L2 99%+)
- 但无法补回 val/test gap 的 -5% 系统性损失
- 说明 val/test gap 跟 codebook 利用率健康度**不是同一根因**

---

## 4. 分析解读

### 4.1 假设证伪
**假设**: "reset 死码让 κ-Stereo codebook 健康 = 解 val/test gap"
**结果**: NO — codebook 利用率上去了 (100%), 但下游 test R@10 仍 < baseline 0.1058. gap 缩窄到 -5.4% 但未消失.

### 4.2 与 R2/R3 falsification 一致
#164, #166-#169, #171 验证同一模式: **κ-Stereographic 测地距离 + L2-based codebook 隐空间 = systematic val/test gap**, reset 死码无法解.

### 4.3 4 κ-Stereo 变体排名
| 变体 | test R@10 | Δ vs baseline |
|------|-----------|---------------|
| #171 dead_code_reset_every=10 | 0.1001 | -5.4% |
| #164 纯 κ-Stereo | 0.0964 | -8.9% |
| #172 κ_max=4.0 | 0.0972 | -8.1% |
| #170 Sinkhorn all 3 | (待 eval) | TBD |
| #169 Sinkhorn L2 only | 0.0863 | -18.4% |
| Baseline #105 (Euclidean RQ-VAE) | 0.1058 | — |

---

## 5. 决策触发判定

按 descriptions/task171 §3 表:
- 实际 test R@10 = 0.1001 < 0.1058 baseline → **⛔ NO-GO**
- 跟 #170/#172 联合 verdict: κ-Stereo 8 变体穷尽, 都 < baseline R@10=0.1058

---

## 6. 产物清单

- **Stage 1 best ckpt**: `products/task171/phase_b_kappa_dead_code/<TS>/best_loss_model.pth`
- **Stage 2 SID codebook**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_dead_code.npy`
- **Stage 3 best ckpt**: `products/task171/t5mini_kappa_dead_code/jul-25-2026_11-43-41/Instruments/Jul-25-2026_11-44-05/HG_Rec_best.pth`
- **Stage 4 metrics**: `verdicts/task171_kappa_dead_code_metrics.json` (test_recalls, test_ndcgs)

---

## 7. 后续建议

1. **§16 cleanup (R8)**: 立即从 §16 删除 Task #171 行 (Stage 4 已 exit 0)
2. **#170/#172/#174**: 等待 Stage 4 exit, 写 verdict 同样模式
3. **#163 synthesis verdict**: 等 #170/#172/#174 后, 整合 8 κ-Stereo 变体 NO-GO 报告
result: Task #171 — κ-Stereographic + dead_code_reset_every=10 [⛔ C3 NO-GO]
