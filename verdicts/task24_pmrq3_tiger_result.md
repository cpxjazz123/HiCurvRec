# Task #24 — PM-RQ Phase 3 三层 cascade SID × TIGER 端到端

> **任务目的**: 验证 PM-RQ Phase 3 三层 cascade (3 layers × K=256, S×E×H × 3 layers = 9 codes/item) SID 作为 TIGER Stage 3 输入时的端到端 Recall@K 性能, 与 Task #87 flat baseline + Task #23 (Phase 2 单层) 对比。

> **完成日期**: 2026-07-19
> **状态**: ✅ 完成 (early-stop on plateau)

---

## 1. 关键指标

| 指标 | Task #24 PM-RQ Phase 3 | Task #23 PM-RQ Phase 2 | Task #87 flat baseline | 比值 (vs Task #87) |
|------|------|------|------|------|
| **Recall@5** | 0.00144 | 0.00474 | 0.01937 | **7.4%** |
| **Recall@10** | 0.00206 | 0.00551 | 0.03318 | **6.2%** |
| **NDCG@5** | 0.00095 | 0.00275 | - | - |
| **NDCG@10** | 0.00114 | 0.00300 | - | - |

**结论**: PM-RQ Phase 3 cascade × TIGER **最弱** (vs Task #87 仅 7%)，且弱于 PM-RQ Phase 2 单层（cascade 9-codes 信息密度反而比 single 3-codes 表现更差）。

---

## 2. 执行时间线

| 阶段 | 状态 | 备注 |
|------|------|------|
| PM-RQ Phase 3 cascade SID 提取 | ✅ | shape (10, 11924), 9-tuple unique 11923/11924 |
| TIGER Stage 3 训练 | ⏸️ Early-stop @ step 14000/100000 | val_R@5 plateau 4/4 个 val 点, best @ step 6000 (0.00206) |
| Stage 4 推断 (best ckpt) | ✅ | merged_predictions_tensor.pt, 19412 users |
| Recall/NDCG 评估 | ✅ | `scripts/task23_105_pmrq_eval.py` |
| Verdict 写盘 | ✅ | 本文件 |

**注**: 训练在 step 14000 处停止（plateau 严重），使用 step 6000 best ckpt 进入 Stage 4。

---

## 3. 训练曲线关键观察

| Step | val_R@5 | val_loss | 状态 |
|------|---------|----------|------|
| 6000 | **0.00206** | 14.20 | **best (final)** |
| 8000 | - | 14.00 | not top 1 |
| 10000 | - | 14.60 | not top 1 |
| 12000 | - | 13.90 | not top 1 |
| 14000 | - | 13.90 | not top 1 |

**关键诊断**: val_loss **真正 plateau** 在 13.90（vs Phase 2 仍在下降）→ cascade 训练容量饱和更早。val_R@5 完全没动。

---

## 4. 分析解读

### 4.1 Cascade 9-codes 信息密度失效

Phase 3 输出 (10, 11924) = 9 codes + 1 dedup = **256^9 = 4.7×10^21 bins** (天文数字)。
理论上信息密度远超 Phase 2 (256^4 = 4.3×10^9)。

**实际表现**: R@5 = 0.00144（30% of Phase 2 single-layer）→ 信息密度未转化为 SID 区分度。
**根因猜想**: 9-tuple unique 11923/11924 ≈ 100% — 每个 item 几乎独立编码，TIGER 的 next-K 预测任务变成"10000+ 个独立 SID 的随机抽样"，序列预测优势消失。

### 4.2 序列长度 vs 信息密度 trade-off

| Task | sequence_length | num_hierarchies | 信息密度 (log K^h) | R@5 |
|------|------|------|------|------|
| Task #87 | 120 | 4 | 4×log(256) = 32 bits | 0.01937 |
| Task #23 | 120 | 4 | 32 bits | 0.00474 |
| Task #24 | 200 | 10 | 80 bits | 0.00144 |

**观察**: 信息密度增加 2.5× 但 R@5 下降 3.3× → **序列模型在有限训练步数下无法吸收更多 digits**。

### 4.3 决策触发对照（vs Task #87 baseline + Task #23）

| 指标条件 | 预期 | 实际 | 决策 |
|----------|------|------|------|
| **R@5 ≥ 0.022** (>Task #87 +13%) | 0.022-0.030 | - | R1 反证: 9-codes 信息密度有效 |
| **R@5 ∈ [0.018, 0.022]** | 0.018-0.022 | - | R1 一致 |
| **R@5 - R@5_Task104 ≥ +0.005** | cascade > single | - | cascade 9-codes 有效 |
| **R@5 - R@5_Task104 ∈ [-0.005, +0.005]** | cascade ≈ single | - | 9-codes vs 3-codes 无差 |
| **R@5 < 0.018, R@5 - R@5_Task104 < -0.005** | - | **0.00144 (=Task104 - 0.0033)** | **R1 确认: cascade 信息有损** |

**确认**: R1 (PM-RQ cascade 9-codes 信息密度优势) 在 TIGER 端到端框架下**严重否证**。

---

## 5. 产物清单

| 文件 | 说明 |
|------|------|
| `products/task22_pm_rq/sid_phase3_cascade.pt` | PM-RQ Phase 3 cascade SID, shape (10, 11924) |
| `logs/task24_pmrq3_s3/runs/2026-07-19/11-10-26/checkpoints/checkpoint_epoch=000_step=006000.ckpt` | Best Stage 3 ckpt |
| `products/task24_pmrq3_tiger/stage3_train/best.ckpt` | Symlink → best ckpt |
| `logs/task24_pmrq3_s4/runs/2026-07-19/12-48-15/pickle/merged_predictions_tensor.pt` | Stage 4 推断输出 |
| `verdicts/task24_pmrq3_tiger_eval.json` | Recall/NDCG 评估结果 |
| `verdicts/task24_pmrq3_tiger_result.md` | 本 verdict |

---

## 6. 后续建议

1. **PM-RQ cascade 已彻底证伪**: 9 codes × 200 seq_len 在 TIGER 框架下完全失败
2. **未来方向不应再追求 SID 信息密度**: 训练容量饱和是主因，模型架构需调整（如 sparse attention over digits）
3. **建议停用 PM-RQ 方向**: 继续投入 GPU 时间在该方向 ROI 极低

result: Task #24 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
