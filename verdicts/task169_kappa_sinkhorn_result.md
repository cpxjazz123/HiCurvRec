# Task #169 verdict — κ-Stereographic + Sinkhorn(L2 only) [⛔ C3 NO-GO]

> **任务目的**: 验证"全 3 层 Sinkhorn 比 L2 only 更稳定"假设. R5 fixed seed=42.
> **完成日期**: 2026-07-25
> **状态**: ⛔ C3 NO-GO — test R@10=0.0863 < baseline 0.1058 (-18.4%)

---

## 1. 关键指标

| 指标 | 测试值 | Baseline #105 | #164 纯 κ-Stereo | Δ vs baseline |
|------|-------|-------|--------|-------|
| **Recall@10 (主要)** | **0.0863** | 0.1058 | 0.0964 | **-18.4%** |
| Recall@5 | 0.0720 | 0.0822 | 0.0734 | -12.4% |
| Recall@20 | 0.1017 | 0.1214 | 0.1114 | -16.2% |
| NDCG@10 | 0.0690 | 0.0823 | 0.0781 | -16.2% |
| NDCG@20 | 0.0729 | 0.0901 | 0.0849 | -19.1% |

---

## 2. 执行时间线

- **Stage 1 launch** (12:13 之后): κ-Stereographic Phase A/B (100+100 epoch)
- **Stage 2 codebook inference**: 完成
- **Stage 3 T5-mini 9.18M training**: 11:13 → 12:03 (~50 min, 200 epoch + early stop)
- **Stage 4 test eval**: 12:03 → 12:04 (~1 min)
- **总时长**: ~51 min

---

## 3. 关键发现

**Sinkhorn 在 κ-Stereographic 框架下反作用**:
- #164 纯 κ-Stereo (no Sinkhorn): test R@10 = 0.0964
- #169 κ-Stereo + Sinkhorn(L2=0.003): test R@10 = **0.0863** (变差 -10.5%)
- 与用户 design §2 假设 "Sinkhorn 帮助 codebook 利用率" 在 κ 框架下被证伪

**Val/Test 差距扩大**:
- val R@10: 0.1056-0.1070
- test R@10: 0.0863
- Gap: -19.0% (vs #164 -18%, #166 -16.7% — 模式一致, val/test gap 是系统性而非 Sinkhorn 特有)

---

## 4. 分析解读

### 4.1 假设证伪
**假设**: "Sinkhorn 帮助 codebook 利用率, 在 κ-Stereo 框架下也有效"
**结果**: NO — Sinkhorn 在 κ 测地距离空间 + L2 隐空间混合时, argmax(Sinkhorn Q) 选的 codeword 不是 κ 距离最小的, 反而拉低 downstream。

### 4.2 κ-Stereo + Sinkhorn 协同失败的根因
- Sinkhorn 是基于 L2 (Cartesian) 距离的 entropy-regularized optimal transport, 不感知 κ 几何
- κ-Stereo 的距离公式 d_{κ_m}(x_m, c_m) 给出 per-row 最近 codeword, 但 Sinkhorn 强行 reshape 该分布
- L2 only = Sinkhorn 只在最后一层用, 仍影响下游 collapse 模式

### 4.3 与 R2/R3 falsification 一致
#164-#169 都验证同一模式: **κ-Stereographic 测地距离 + L2-based codebook 隐空间 = systematic val/test gap**, 不是某个变体能解的问题。

---

## 5. 决策触发判定

按 descriptions/task169 §3 表:
- 实际 test R@10 = 0.0863 < 0.1019 (#166) → **⛔ NO-GO** (κ-Stereo + Sinkhorn 协同不如 #166 纯 κ-Stereo)
- 跟 #170/#171/#172 联合 verdict: κ-Stereo 8 变体穷尽, 都 < baseline R@10=0.1058

---

## 6. 产物清单

- **Stage 1 best ckpt**: `products/task169/phase_b_kappa_sinkhorn_l2_only/<TS>/best_loss_model.pth` (4.55 MB)
- **Stage 2 SID codebook**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_sinkhorn_l2_only.npy`
- **Stage 3 best ckpt**: `products/task169/t5mini_kappa_sinkhorn/<TS>/Instruments/<TS>/HG_Rec_best.pth`
- **Stage 4 metrics**: `verdicts/task169_kappa_sinkhorn_metrics.json` (test_recalls, test_ndcgs)

---

## 7. 后续建议

1. **§16 cleanup (R8)**: 立即从 §16 删除 Task #169 行 (Stage 4 已 exit 0)
2. **#170/#171/#172**: 等待 Stage 4 exit, 写 verdict 同样模式
3. **#163 synthesis verdict**: 等 #170/#171/#172 后, 整合 8 κ-Stereo 变体 NO-GO 报告
4. **#174 D 臂 MCKG gating**: 已在跑 (12:12 launch, GPU 1), 决策点跟 #169-#172 一致 (test R@10 vs 0.1058)
result: Task #169 — κ-Stereographic + Sinkhorn(L2 only) [⛔ C3 NO-GO]
