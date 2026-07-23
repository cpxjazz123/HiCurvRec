# Task #23 — PM-RQ Phase 2 单层 SID × TIGER 端到端

> **任务目的**: 验证乘积流形 RQ (PM-RQ, K=256 单层, S×E×H) SID 作为 TIGER 输入时的端到端 Recall@K 性能, 与 Task #87 flat Euclidean baseline 对比。

> **完成日期**: in progress
> **状态**: 🟡 SID 已提取, TIGER Stage 3 待启动

---

## 1. 背景

Task #22 Phase 4 决策曾判断"不启动 8h Stage 3 重训 (信息增量 <5%)", 因信息已通过 Phase 0/1b/2/3/4a 各相位证。用户 2026-07-19 11:00 重新指示: "完整做 PM-RQ × TIGER 端到端实验", 启动 2 个新实验 (本任务为 #23, 另一个 #24 是 Phase 3 cascade)。

### 1.1 上游 Phase 摘要 (Task #22)
- Phase 0 (D0 几何诊断): ✅ norm_cv spread=10.45 (vs 原 1.14, 9.2x), subspace cos=-0.17 (强独立)
- Task #99 重建 MCKG: ✅ κ ∈ [+5.05, -0.08, -5.04] (vs 原 ±1)
- Phase 2 (单层 K=256, 11924 items): ✅ util_s/e/h = 1.0/0.898/1.0, 训练 recon_loss=0.32

### 1.2 对照基线
- Task #87 (flat Euclidean RQ-VAE × TIGER): R@5=0.01937, R@10=0.03318, NDCG@5=0.01222, NDCG@10=0.01663
- Task #80 (fused KMeans × flat Stage 3): R@5=0.0383
- paper RQ-VAE Toys: R@5=0.034, R@10=0.051

---

## 2. 实验设计

**变量**: Stage 2 SID = PM-RQ Phase 2 单层 (idx_s, idx_e, idx_h, K=256, MCKG 输入)
**保持不变**:
- Stage 1: sentence-t5-base 11924×768 embedding (Task #87 复用共享数据)
- Stage 2.1: flat Euclidean RQ-VAE (K=256, num_hierarchies=3) — 与 Task #87 完全一致
- Stage 3 TIGER 配置: Adafactor lr=0.01, InverseSqrtScheduler warmup=10000, num_user_bins=2000, LSH Hashing Trick, dropout=0.1, max_steps=100000
- Stage 4: TIGER 推断 sequence_length=120, top_k=10
- seed=42
- 数据: Toys (data/amazon_data/toys)

**Stage 2 替换点**:
- 不重新训练 PM-RQ, 直接 forward 从已保存 `products/task22_pm_rq/phase2_full/phase2_model.pt` 提取 (11924, 3) SID
- pad 为 (4, 11924) (3 hierarchy + 1 dedup), 与 Task #87 sid_dedup.pt shape 一致
- TIGER `semantic_id_path` 指向 `products/task22_pm_rq/sid_phase2.pt`
- TIGER `num_hierarchies=4` (与 Task #87 一致)

**启动命令 (TIGER Stage 3 训练)**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# PM-RQ Phase 2 SID 路径
SEMANTIC_ID_PATH=products/task22_pm_rq/sid_phase2.pt

CUDA_VISIBLE_DEVICES=0 nohup python -m src.train experiment=tiger_train_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SEMANTIC_ID_PATH \
    sequence_length=120 \
    num_hierarchies=4 \
    task_name=task23_pmrq2_s3 \
    > logs/task23_pmrq2_s3.log 2>&1 &
```

**Stage 4 推断** (TIGER Stage 4 启动):
```bash
SEMANTIC_ID_PATH=products/task22_pm_rq/sid_phase2.pt
BEST_CKPT=products/task23_pmrq2_tiger/stage3_train/best.ckpt

CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=tiger_inference_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SEMANTIC_ID_PATH \
    ckpt_path=$BEST_CKPT \
    num_hierarchies=4 sequence_length=120 \
    task_name=task23_pmrq2_s4
```

**Recall/NDCG 评估**: 复用 `scripts/task87_s4_item_eval.py`, 修参数指向 task23 路径, ~5 min CPU

---

## 3. 决策触发 (vs Task #87 baseline)

| 指标条件 | 预期 R@5 | 决策 |
|----------|---------|------|
| **R@5 ≥ 0.022** (vs Task #87 0.01937, +13%) | 0.022-0.025 | **R1 假设置疑**: PM-RQ 是否在 TIGER 框架下不弱? |
| **R@5 ∈ [0.018, 0.022]** (噪声范围) | 0.018-0.022 | **R1 一致**: 单 κ SID 弱于 fused KMeans 但与 flat Euclidean 信息量等同 (无几何增益) |
| **R@5 < 0.018** | <0.018 | **R1 确认**: PM-RQ SID 在 TIGER 框架下表现更弱 (亏损 vs flat Euclidean) |
| **R@10 ≥ 0.035** | >paper 65% | 跨 baseline 一致性 |
| **NDCG@10 ≥ 0.018** | >paper 60% | 排序质量改进 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| PM-RQ SID 提取 | ✅ 已完成 (1 min CPU) |
| TIGER Stage 3 训练 (100k steps) | ~6-8 h GPU on cuda:0 |
| TIGER Stage 4 推断 | ~1 min |
| Recall/NDCG 评估 | ~5 min CPU |
| **总计 (剩)** | **~7-8 h** |

---

## 5. 风险与缓解

**风险 1**: PM-RQ SID 与 Task #87 flat Euclidean SID 信息熵接近, R@5 落在 [0.018, 0.022] 噪声范围 → 决策"无几何增益"
- 缓解: 与 Task #24 (Phase 3 cascade) 对比, 若两个 PM-RQ 实验都无增益 → R1 证实
- 若 Task #24 显示提升 → R1 部分支持, 需进一步信息

**风险 2**: PM-RQ MCKG 输入与 T5 输入不对齐, 导致 SID overlap 弱 → R@5 < Task #87
- 缓解: 本任务允许 R@5 < Task #87 (≤18%), 此时仍可作为 PM-RQ "损失度" 的度量

**风险 3**: TIGER 训练 100k steps 跑满未达到 plateau → 早停 (Task #87 经验: val_R@5 best 在 step 66000)
- 缓解: 监控 val_R@5, 若连续 5 个 val 点 (10k steps) 无提升 → `kill -TERM` 优雅停训

**风险 4**: Stage 4 推断 2× data bug (Task #87 经验)
- 缓解: 推断后立即 dedup tensor (`merged_predictions_dedup.pt`)

---

## 6. 完成度跟踪

- [x] PM-RQ Phase 2 SID 提取 → `products/task22_pm_rq/sid_phase2.pt` shape (4, 11924)
- [x] Verify SID: util s=256/256, e=230/256, h=256/256, 11255 unique 3-tuples
- [ ] TIGER Stage 3 训练 (~6-8h)
- [ ] Stage 3 best ckpt 保存到 `products/task23_pmrq2_tiger/stage3_train/best.ckpt`
- [ ] Stage 4 推断 + 2× data dedup
- [ ] Recall/NDCG 评估 JSON
- [ ] 写 `verdicts/task23_pmrq2_tiger_result.md`
- [ ] 与 Task #87 + Task #24 对比表 (`verdicts/task23_vs_105_comparison.md`)

---

## 7. 关联

- 上游: Task #22 Phase 2 (`verdicts/task22_phase2_full_result.md`)
- 同批并行任务: Task #24 (PM-RQ Phase 3 cascade × TIGER)
- baseline: Task #87 (`verdicts/task87_tiger_baseline_result.md`)
