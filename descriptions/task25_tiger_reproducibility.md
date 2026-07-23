# Task #25 — TIGER baseline 可复现性测试 (重训 seed=123)

> **任务目的**: 验证 Task #87 TIGER baseline (R@5=0.01937, seed=42) 在不同 seed 下是否可复现, 排除"该数字是 lucky seed"的疑虑, 锁定 Toys 数据集上 flat Euclidean SID × TIGER 框架的 R@5 ceiling/floor 区间。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

- Task #87 baseline 完成于 2026-07-19 09:17, best ckpt @ step 66000, val_R@5=0.02050, **真实 test eval** R@5=**0.01937**, R@10=**0.03318**, NDCG@5=**0.01222**, NDCG@10=**0.01663** (9706 unique users)
- paper RQ-VAE Toys baseline 目标 R@5=0.034, 达成率 57%
- 单 seed 实验无法区分"信号 vs 噪声": 若 seed=42 是 outlier, 真实 R@5 应在 ±0.005 区间波动
- Task #23/#24 PM-RQ × TIGER 均在 seed=42 下做, 同样受单 seed 噪声影响
- 本任务用 **seed=123** 重跑 Task #87 全 pipeline (Stage 3 + 4 + eval), 不改任何其他超参, 仅隔离 seed 变量

---

## 2. 实验设计

**变量**: seed (42 → 123)
**保持不变**:
- Stage 2 SID: Task #87 baseline cluster_ids.pt (`logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt`)
- Stage 3 TIGER 配置: `experiment=tiger_train_tiger`, Adafactor lr=0.01, InverseSqrtScheduler warmup=10000, num_user_bins=2000, LSH Hashing Trick, dropout=0.1, max_steps=100000, val_check_interval=2000
- Stage 4 sequence_length=120, top_k=10
- 数据: Toys, num_hierarchies=4

**启动命令 (Stage 3 训练)**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CUDA_VISIBLE_DEVICES=0 nohup python -m src.train experiment=tiger_train_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 \
    num_hierarchies=4 \
    seed=123 \
    task_name=task25_baseline_seed123 \
    > logs/task25_baseline_seed123.log 2>&1 &
```

**Stage 4 推断**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=tiger_inference_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt \
    ckpt_path=products/task25_baseline_seed123/stage3_train/best.ckpt \
    num_hierarchies=4 sequence_length=120 \
    task_name=task25_baseline_seed123_s4
```

---

## 3. 决策触发

| 指标条件 | 实际 R@5 | 决策 |
|----------|---------|------|
| **R@5 ∈ [0.014, 0.024]** (seed 42 ±0.005) | 同区间 | ✅ 可复现, 锁定 Toys ceiling ≈ 0.0194 |
| **R@5 ∈ [0.024, 0.034]** (>seed 42 +25%) | 显著高 | 锁定真实 baseline 上限 |
| **R@5 < 0.014** (<seed 42 -28%) | 显著低 | 锁定真实 baseline 下限 |
| **R@5 ≥ 0.034** (=paper) | 同 paper | 验证 Task #87 数字有偏, paper 才是真实上限 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 | ~6h (单卡 cuda:0, 4.4 it/s × 100k steps) |
| Stage 4 推断 | ~1 min |
| Recall/NDCG 评估 | ~5 min CPU |
| **总计** | **~6h** |

---

## 5. 风险与缓解

**风险 1**: seed=123 训练在 100k 内不收敛 (Task #87 best @ 66000)
- 缓解: val_check_interval=2000 → 监控 val_R@5 趋势, 若 step 80000 仍 < 0.015 → 早停用现有 best

**风险 2**: 与正在运行的实验抢 cuda:0 (若有)
- 缓解: 启动前 `nvidia-smi` 核对 (R7)

**风险 3**: Stage 4 推断 ckpt_path 路径含 `=` (Task #87 教训)
- 缓解: symlink 到 `products/task25_baseline_seed123/stage3_train/best.ckpt` 后再传参

---

## 6. 完成度跟踪

- [x] Task #25 description 写盘
- [ ] Stage 3 启动 (cuda:0, seed=123)
- [ ] Stage 3 best ckpt 保存 (ModelCheckpoint monitor=val/recall@5)
- [ ] Stage 4 推断 + 2× data dedup
- [ ] Recall/NDCG 评估 JSON
- [ ] 写 `verdicts/task25_baseline_seed123_result.md`
- [ ] 与 Task #87 (seed=42) 对比表 → `verdicts/task25_vs_87_reproducibility.md`

---

## 7. 关联

- 基准: Task #87 (`verdicts/task87_tiger_baseline_result.md`, seed=42, R@5=0.01937)
- 上游 SID: Task #87 Stage 2 RQ-VAE v6 inference
- 关联: Task #23 / #24 PM-RQ × TIGER (同 seed=42, 本任务量化其噪声区间)
