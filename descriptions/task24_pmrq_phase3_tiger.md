# Task #24 — PM-RQ Phase 3 三层 cascade SID × TIGER 端到端

> **任务目的**: 验证乘积流形 RQ (PM-RQ, 3层 cascade, 每层 K=256, S×E×H × 3 layers) SID 作为 TIGER 输入时的端到端 Recall@K 性能, 与 Task #87 flat Euclidean baseline + Task #23 (Phase 2 单层) 对比。

> **完成日期**: in progress
> **状态**: 🟡 SID 已提取, TIGER Stage 3 待启动

---

## 1. 背景

与 Task #23 并行。同为用户 2026-07-19 11:00 重新指示 "完整做 PM-RQ × TIGER 端到端实验" 的一部分。

### 1.1 上游 Phase 摘要 (Task #22 Phase 3)
- Phase 3 cascade (3 layers × K=256): ✅ util_e L0/L1/L2 = 0.895/0.965/0.973 (递增)
- V-info 跨层递减 L1/L2/L3 = 0.926/0.896/0.871 (但 >0.7 未达深层压缩目标)
- 9 codes per item (3 layers × 3 sub) — 比 Phase 2 多 3× 信息密度

### 1.2 关键差异 vs Task #23 (Phase 2 single)
- SID 形状: (10, 11924) vs (4, 11924)
- num_hierarchies=10 vs 4
- sequence_length=200 vs 120 (9 codes/item 需更长序列)

### 1.3 对照基线 (同 Task #23)
- Task #87 flat Euclidean × TIGER: R@5=0.01937, R@10=0.03318
- paper RQ-VAE Toys: R@5=0.034, R@10=0.051

---

## 2. 实验设计

**变量**: Stage 2 SID = PM-RQ Phase 3 三层 cascade (9 codes/item, K=256, MCKG 输入)
**保持不变**: 
- Stage 3 TIGER 配置与 Task #23 一致 (Adafactor + InverseSqrt + 2000 user bins + LSH)
- Stage 4 sequence_length=200, top_k=10
- seed=42, 数据: Toys

**SID 提取** (已 ✅ 完成): `scripts/task23_extract_pmrq_sid.py`
- Forward 11924 items (3 layers, K=256 each)
- Stack → (9, 11924) → pad (10, 11924) (9 hierarchy + 1 dedup)
- util per layer: L0=256/229/256, L1=256/247/256, L2=256/249/256
- 9-tuple unique: 11923/11924 (99.99% 几乎全覆盖)

**启动命令 (TIGER Stage 3 训练)**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# PM-RQ Phase 3 cascade SID 路径
SEMANTIC_ID_PATH=products/task22_pm_rq/sid_phase3_cascade.pt

CUDA_VISIBLE_DEVICES=1 nohup python -m src.train experiment=tiger_train_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SEMANTIC_ID_PATH \
    sequence_length=200 \
    num_hierarchies=10 \
    task_name=task24_pmrq3_s3 \
    > logs/task24_pmrq3_s3.log 2>&1 &
```

**Stage 4 推断**:
```bash
SEMANTIC_ID_PATH=products/task22_pm_rq/sid_phase3_cascade.pt
BEST_CKPT=products/task24_pmrq3_tiger/stage3_train/best.ckpt

CUDA_VISIBLE_DEVICES=1 python -m src.inference experiment=tiger_inference_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SEMANTIC_ID_PATH \
    ckpt_path=$BEST_CKPT \
    num_hierarchies=10 sequence_length=200 \
    task_name=task24_pmrq3_s4
```

---

## 3. 决策触发 (vs Task #87 baseline + Task #23 Phase 2)

| 指标条件 | 预期 R@5 | 决策 |
|----------|---------|------|
| **R@5 ≥ 0.022** (>Task #87 0.01937 +13%) | 0.022-0.030 | **R1 反证**: 9-codes 信息密度战胜 flat Euclidean 单 κ SID |
| **R@5 ∈ [0.018, 0.022]** (噪声) | 0.018-0.022 | **R1 一致**: 信息密度未转化为 Recall 提升 |
| **R@5 < 0.018** (弱于 baseline) | <0.018 | **R1 确认**: PM-RQ cascade SID 在 TIGER 框架下信息有损 |
| **R@5 - R@5_Task104 ∈ [-0.005, +0.005]** | cascade ≈ single | 9 codes 与 3 codes 信息密度无差 |
| **R@5 - R@5_Task104 ≥ +0.005** | cascade > single | cascade 9-codes 信息密度有效 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| PM-RQ SID 提取 | ✅ 已完成 (1.6 min CPU) |
| TIGER Stage 3 训练 (100k steps, num_hierarchies=10) | ~8-10 h GPU on cuda:1 (与 Task #23 并行) |
| TIGER Stage 4 推断 | ~1 min |
| Recall/NDCG 评估 | ~5 min CPU |
| **总计 (剩)** | **~8-10 h** |

> 注: 9-codes 模型 sequence_length=200 → 100k steps 训练 + 每 step 的 sequence compute cost vs 120 约 +30%, 估 8-10h

---

## 5. 风险与缓解

**风险 1**: 9-codes 信息密度未转 Recall 提升 (信息熵上限受限)
- 缓解: 对比 Task #23 (Phase 2 3 codes), 若两者都 ≈ flat Euclidean → 信息熵上限由 MCKG 输入决定, 与 cascade 无关
- 若 cascade > single → cascade 9-codes 有效 (R1 反证)

**风险 2**: TIGER num_hierarchies=10 训练不稳定 (vocab 维度 +70%)
- 缓解: 监控 val_loss, 若 loss 不收敛 → 早停 + 检查 ce_loss

**风险 3**: SID 9-tuple uniqueness 11923/11924 ≈ 100% — 每个 item 有独立 SID, 与 Task #87 (3 层级 dedup) 不同 → Loss 行为可能不同
- 缓解: 检查 TIGER 训练时 target 是否能区分 11924 个独立 SID — 若全是 independent tokens, cross-entropy 应正常

**风险 4**: Stage 4 sequence_length=200 推断慢 (~2x)
- 缓解: 预估 2 min 推断仍 <5 min 阈值

---

## 6. 完成度跟踪

- [x] PM-RQ Phase 3 cascade SID 提取 → `products/task22_pm_rq/sid_phase3_cascade.pt` shape (10, 11924)
- [x] Verify SID: util per layer L0=256/229/256, L1=256/247/256, L2=256/249/256
- [ ] TIGER Stage 3 训练 (~8-10h)
- [ ] Stage 3 best ckpt 保存
- [ ] Stage 4 推断 + 2× data dedup
- [ ] Recall/NDCG 评估 JSON
- [ ] 写 `verdicts/task24_pmrq3_tiger_result.md`
- [ ] 与 Task #87 + Task #23 对比表 (`verdicts/task23_vs_105_comparison.md`)

---

## 7. 关联

- 上游: Task #22 Phase 3 (`verdicts/task22_phase3_cascade_result.md`)
- 同批并行任务: Task #23 (PM-RQ Phase 2 single-layer × TIGER)
- baseline: Task #87 (`verdicts/task87_tiger_baseline_result.md`)
