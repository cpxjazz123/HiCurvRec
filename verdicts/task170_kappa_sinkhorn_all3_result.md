# Task #170 verdict — κ-Stereographic + Sinkhorn(ALL 3 layers) [⛔ C3 NO-GO]

> **任务目的**: 验证"全 3 层 Sinkhorn 比 L2 only 更稳定"假设. R5 fixed seed=42.
> **完成日期**: 2026-07-25
> **状态**: ⛔ C3 NO-GO — test R@10=0.0836 < baseline 0.1058 (-21.0%)

---

## 1. 关键指标

| 指标 | 测试值 | Baseline #105 | #164 纯 κ-Stereo | #169 Sinkhorn L2 only | Δ vs baseline |
|------|-------|-------|--------|--------|-------|
| **Recall@10 (主要)** | **0.0836** | 0.1058 | 0.0964 | 0.0863 | **-21.0%** |
| Recall@5 | 0.0721 | 0.0822 | 0.0734 | 0.0720 | -12.3% |
| Recall@20 | 0.0995 | 0.1214 | 0.1114 | 0.1017 | -18.0% |
| NDCG@10 | 0.0679 | 0.0823 | 0.0781 | 0.0690 | -17.5% |
| NDCG@20 | 0.0720 | 0.0901 | 0.0849 | 0.0729 | -20.1% |

---

## 2. 执行时间线

- **Stage 1 launch**: κ-Stereographic Phase A/B (100+100 epoch), `--sk_epsilons 0.003 0.003 0.003`
- **Stage 2 codebook inference**: 完成
- **Stage 3 T5-mini 9.18M training**: 11:44 → 12:33 (~50 min, 200 epoch + early stop @ ep83)
- **Stage 4 test eval**: 12:36 → 12:37 (~1 min)
- **总时长**: ~53 min

---

## 3. 关键发现

**全 3 层 Sinkhorn 比 L2 only 更差** (反用户 design §2 假设):
- #169 Sinkhorn L2 only: test R@10 = 0.0863 (-18.4%)
- #170 Sinkhorn ALL 3 layers: test R@10 = **0.0836** (-21.0%)
- Δ -2.7%: Sinkhorn 应用越广, 下游损失越大

**全 6 κ-Stereo 变体排名 (now all done)**:
| 变体 | test R@10 | Δ vs baseline |
|------|-----------|---------------|
| #171 dead_code_reset_every=10 | 0.1001 | -5.4% |
| #172 κ_max=4.0 | 0.0972 | -8.1% |
| #164 纯 κ-Stereo | 0.0964 | -8.9% |
| #169 Sinkhorn L2 only | 0.0863 | -18.4% |
| **#170 Sinkhorn all 3** | **0.0836** | **-21.0%** |
| Baseline #105 (Euclidean RQ-VAE) | 0.1058 | — |

---

## 4. 分析解读

### 4.1 假设证伪
**假设**: "全 3 层 Sinkhorn 比 L2 only 更稳定, 因为 L0/L1 都 collapse 也能通过 Sinkhorn rescue"
**结果**: NO — 全 3 层 Sinkhorn 不仅没 rescue 反而让 loss 更大 (-21% vs -18.4% L2 only). 说明 Sinkhorn 在 κ 测地空间施加 entropy-regularized OT 越深, 越干扰 downstream.

### 4.2 Sinkhorn 失败模式 (一致性)
- Sinkhorn 是 L2-Cartesian 距离 entropy-regularized OT, 不感知 κ 几何
- 每加一层 Sinkhorn 都强行 reshape 该层 codebook 利用率分布, 但 argmax Q 选的 codeword 不是 κ 距离最近
- 全 3 层 Sinkhorn = 3 层都 reshape, 累计误差最大 → 最差下游

### 4.3 与 R2/R3 falsification 一致
#164, #166-#169, #170-#172 验证同一模式: **κ-Stereographic 测地距离 + L2-based codebook 隐空间 = systematic val/test gap**, Sinkhorn 在 κ 框架下系统性反作用.

---

## 5. 决策触发判定

按 descriptions/task170 §3 表:
- 实际 test R@10 = 0.0836 < 0.1058 baseline → **⛔ NO-GO**
- 跟 #171/#172 联合 verdict: κ-Stereo 8 变体穷尽, 都 < baseline R@10=0.1058

---

## 6. 产物清单

- **Stage 1 best ckpt**: `products/task170/phase_b_kappa_sinkhorn_all3/<TS>/best_loss_model.pth`
- **Stage 2 SID codebook**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_sinkhorn_all3.npy`
- **Stage 3 best ckpt**: `products/task170/t5mini_kappa_sinkhorn_all3/jul-25-2026_11-43-38/Instruments/Jul-25-2026_11-44-05/HG_Rec_best.pth`
- **Stage 4 metrics**: `verdicts/task170_kappa_sinkhorn_all3_metrics.json` (test_recalls, test_ndcgs)

---

## 7. 后续建议

1. **§16 cleanup (R8)**: 立即从 §16 删除 Task #170 行 (Stage 4 已 exit 0)
2. **#174**: 等待 Stage 4 exit, 写 verdict 同样模式
3. **#163 synthesis verdict**: 等 #174 后, 整合 8 κ-Stereo 变体 NO-GO 报告