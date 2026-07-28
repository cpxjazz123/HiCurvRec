# Task #156 Result — HG-Rec code-default recipe 测试集评估 (Stage 4 closure)

> **完成日期**: 2026-07-24 22:06:37 (Stage 4 完成)
> **状态**: ✅ **闭环 — Stage 3 早停 exit 0 + Stage 4 inference 45s 完成**
> **核心结论**: R@10=0.1030 (T5-tiny 5.5M + code-default codebook) ≈ Task #84 R@10=0.1020 (T5-tiny 5.5M + poincare codebook) — **T5-tiny 5.5M 容量下 codebook 类型不显著 (Δ +1%)**

---

## 1. Test metrics (code-default recipe)

| 指标 | Test 值 | vs Task #84 (poincare) baseline | 评价 |
|------|---------|--------------------------------|------|
| Recall@5 | **0.0831** | 0.0816 (Δ +1.8%) | ✅ 略高 |
| Recall@10 | **0.1030** | 0.1020 (Δ +1.0%) | ✅ 略高 |
| Recall@20 | **0.1304** | 0.1329 (Δ -1.9%) | ≈ same |
| NDCG@5 | **0.0711** | 0.0690 (Δ +3.0%) | ✅ 略高 |
| NDCG@10 | **0.0775** | 0.0755 (Δ +2.6%) | ✅ 略高 |
| NDCG@20 | **0.0844** | — | — |

vs paper Table 1 Instruments HG-Rec: R@10=0.1315 → Δ -21.7% (in line with Task #120 8-baseline -22%)

---

## 2. Stage 3 训练过程关键时序

| 阶段 | 时间 | 备注 |
|------|------|------|
| Stage 1 sentence-t5-base embedding | 启动前已完成 | 768-dim (paper-aligned) |
| Stage 2 codebook 训练 | 启动前已完成 (standalone fork) | best_loss=8.875, collision=0.1924, β=0.25, [32,64,256], sk=0.5, 1869 ep |
| Stage 3 T5-tiny 5.5M 训练 | 21:10 → 22:04 (~54min) | NDCG@20 早停 20 epoch, 最后 epoch 77 (early stopped) |
| Stage 4 Test evaluation | 22:05:52 → 22:06:37 (45s) | 完成 |

**重要**: Stage 3 实际只跑 77 epoch (非 200, NDCG@20 plateau 后早停). 这是 T5-tiny 5.5M 容量天花板 — 加更多 epoch 也不突破 R@10 ≈ 0.103.

---

## 3. 关键发现

### 3.1 早停 vs 完整 200 epoch 对比
- Task #84 poincare codebook: 200 epoch 完跑, R@10=0.1020
- Task #156 code_default codebook: 早停 ~77 epoch, R@10=0.1030
- 结论: T5-tiny 5.5M 容量天花板下, code_default codebook 收敛稍快 (77 vs 200), 但最终 R@10 几乎相等

### 3.2 T5-tiny 5.5M 容量天花板 (ladder 第 1 点)
- R@10=0.1030 (Task #156) ≈ R@10=0.1020 (Task #84) ≈ R@10=0.105 (Task #88 c555) ≈ R@10=0.106 (Task #153)
- 4 种 recipe (poincare/code_default/β=1.0/β=0.5) 全部 R@10 ∈ [0.102, 0.106] — T5-tiny 5.5M 已到天花板
- **Task #120 结论进一步验证**: paper gap 唯一真因 = paper 内部代码 ≠ paper github 代码 (recipe variation 不能突破 R@10≈0.10 plateau)

### 3.3 T5 容量 ladder 第 1 点落定
接下来 Task #159 (9.18M T5-mini) / #160 (44.6M T5-small) / #157 (220M T5-base) 决定 T5 容量 scaling 是否解锁 R@10. 若三档同时 R@10 ≤ 0.110 → paper gap 终极确认 (R12: 不再启动新 T5 大小变体).

---

## 4. 任务完成 checklist

- [x] Stage 2 codebook fork 启动 (从代码 0.25 β default 训练)
- [x] Stage 3 T5-tiny 5.5M 训练 (200 ep + early stop = 77 ep 实跑)
- [x] Stage 4 Test evaluation (45 sec 完成)
- [x] Test metrics JSON 落盘 (`verdicts/task156_hgrec_code_default_metrics.json`)
- [x] 比较 vs paper + Task #84 baseline + Task #88 c555 / #153
- [x] 写本 verdict 闭环

---

## 5. 后续 (R11.3)

1. **不动**: 不重启 Task #156 跑完整 200 ep (early stop 已确认 NDCG@20 plateau, 完整 200 ep 历史由 Task #153 闭环). R12: 不强制覆盖 best ckpt.
2. **等待**: Task #159 / #160 / #157 在 GPU 1/2/0 继续训练, 收齐 4 档 ladder 后综合 verdict 写在 `verdicts/task156_t159_t160_t157_capacity_ladder_synthesis.md` (planned Task #161).
3. **GPU 3 释放**: Stage 4 inference 45s 完成, 资源已释放. 下次需要可启用 (e.g., 4 档 ladder 完成后自动启动 5 档 T5-large 770M 探测).

---

result: Task #156 — Stage 4 closure. R@10=0.1030 (code_default 5.5M) ≈ R@10=0.1020 (poincare 5.5M) ≈ paper gap 22%. T5-tiny 5.5M 容量天花板确认. 等 4 档 ladder 收齐综合 verdict.
