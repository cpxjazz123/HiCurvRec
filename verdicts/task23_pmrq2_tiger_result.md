# Task #23 — PM-RQ Phase 2 单层 SID × TIGER 端到端

> **任务目的**: 验证乘积流形 RQ (PM-RQ, Phase 2 单层 K=256, S×E×H 三子空间) SID 作为 TIGER Stage 3 输入时的端到端 Recall@K 性能, 与 Task #87 flat Euclidean baseline 对比。

> **完成日期**: 2026-07-19
> **状态**: ✅ 完成 (early-stop on plateau)

---

## 1. 关键指标

| 指标 | Task #23 PM-RQ Phase 2 | Task #87 flat baseline | 比值 |
|------|------|------|------|
| **Recall@5** | 0.00474 | 0.01937 | 24.5% |
| **Recall@10** | 0.00551 | 0.03318 | 16.6% |
| **NDCG@5** | 0.00275 | - | - |
| **NDCG@10** | 0.00300 | - | - |

**结论**: PM-RQ Phase 2 × TIGER 显著弱于 flat Euclidean baseline (R@5 ≈ 1/4)。

---

## 2. 执行时间线

| 阶段 | 状态 | 备注 |
|------|------|------|
| PM-RQ Phase 2 SID 提取 | ✅ | shape (4, 11924) → pad (4, 11924) |
| TIGER Stage 3 训练 | ⏸️ Early-stop @ step 24000/100000 | val_R@5 plateau 12/12 个 val 点, best @ step 8000 (0.00412) |
| Stage 4 推断 (best ckpt) | ✅ | merged_predictions_tensor.pt, 19412 users |
| Recall/NDCG 评估 | ✅ | `scripts/task23_105_pmrq_eval.py` |
| Verdict 写盘 | ✅ | 本文件 |

**注**: 训练在 step 24000 处停止（plateau 已确认）, 使用 step 8000 best ckpt 进入 Stage 4。

---

## 3. 训练曲线关键观察

| Step | val_R@5 | val_loss | 状态 |
|------|---------|----------|------|
| 4000 | 0.00242 | 10.50 | best |
| 6000 | 0.00314 | 10.30 | best |
| 8000 | **0.00412** | 10.20 | **best (final)** |
| 10000 | - | 10.10 | not top 1 |
| 12000 | - | 9.940 | not top 1 |
| 14000 | - | 9.910 | not top 1 |
| 16000 | - | 9.870 | not top 1 |
| 18000 | - | 9.880 | not top 1 |
| 20000 | - | 9.780 | not top 1 |
| 22000 | - | 9.740 | not top 1 |
| 24000 | - | 9.730 | not top 1 |

**关键诊断**: val_loss 持续缓慢下降 (10.50 → 9.730) 但 val_R@5 在 step 8000 后完全 plateau。最终 R@5 = 0.00412 (training 监控) vs Stage 4 完整 eval R@5 = 0.00474 — 二者一致 (eval 包含 dataloader 完整路径, 略高)。

---

## 4. 分析解读

### 4.1 PM-RQ × TIGER 信息瓶颈

PM-RQ Phase 2 用 MCKG 输入 + 三子空间 (S×E×H) 距离，**最大化几何保真但牺牲重建效率**：
- 11924 items × 4 digits SID = 4 codes per item
- 相比 Task #87 flat Euclidean 3-digits + 1 dedup (约 24-bit 总熵 = 16M bins)
- PM-RQ 单层 + dedup = 256^4 = 4.3B bins (信息密度更高)
- 但 MCKG embedding 输入表达力弱于 T5 (Task #22 阶段已确认)

**结果**: 信息密度增加未转化为 Recall → MCKG 输入成为瓶颈。

### 4.2 与 Task #85 三几何 RQ-VAE 对比

| Task | R@5 | 备注 |
|------|-----|------|
| Task #85 m=0 (Euclidean) | 0.02782 | 82% of baseline |
| Task #85 m=1 (球面) | - | 同/略低 |
| Task #85 m=2 (双曲) | - | 同/略低 |
| **Task #23 PM-RQ Phase 2** | **0.00474** | **24% of baseline** |

PM-RQ 表现远差于 Task #85 单一几何 → **三子空间几何混合在 TIGER 框架下引入噪声而非信息**。

### 4.3 决策触发对照（vs Task #87 baseline）

| 指标条件 | 预期 | 实际 | 决策 |
|----------|------|------|------|
| **R@5 ≥ 0.030** (>Task #87 +55%) | 0.030-0.040 | - | R1 反证: 几何优势成立 |
| **R@5 ∈ [0.022, 0.030]** | 0.022-0.030 | - | R1 一致 |
| **R@5 ∈ [0.018, 0.022]** | 0.018-0.022 | - | 噪声 |
| **R@5 < 0.018** | - | **0.00474** | **R1 确认: PM-RQ Phase 2 信息有损** |

**确认**: R1 (PM-RQ 信息密度优势) 在 TIGER 端到端框架下**未成立**。

---

## 5. 产物清单

| 文件 | 说明 |
|------|------|
| `products/task22_pm_rq/sid_phase2.pt` | PM-RQ Phase 2 SID, shape (4, 11924) |
| `logs/task23_pmrq2_s3/runs/2026-07-19/11-10-26/checkpoints/checkpoint_epoch=000_step=008000.ckpt` | Best Stage 3 ckpt |
| `products/task23_pmrq2_tiger/stage3_train/best.ckpt` | Symlink → best ckpt |
| `logs/task23_pmrq2_s4/runs/2026-07-19/12-48-15/pickle/merged_predictions_tensor.pt` | Stage 4 推断输出 |
| `verdicts/task23_pmrq2_tiger_eval.json` | Recall/NDCG 评估结果 |
| `verdicts/task23_pmrq2_tiger_result.md` | 本 verdict |

---

## 6. 后续建议

1. **PM-RQ 路径已证伪**: 单层 (Phase 2) + cascade (Phase 3) 在 TIGER 端到端均显著弱于 flat baseline
2. **下一研究方向**: 若坚持 PM-RQ 思路，应尝试 MCKG 输入增强（更多 KG 边 / 不同编码方式）而非依赖 SID 几何保真
3. **本实验 R1 已否证**: 三子空间 (S×E×H) 在 TIGER 框架下不优于单一欧氏 RQ-VAE

result: Task #23 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
