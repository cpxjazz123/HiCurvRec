# Task #72 — RQ-VAE codebook_width 256 → 512 ablation

> **任务目的**: 验证 Stage 2 RQ-VAE codebook 容量翻倍（256 → 512）是否能产生更细粒度的 SID，进而提升 Stage 3 TIGER 推荐的 R@10

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #71（三流形 cascade RQ-VAE）Stage 4 R@10 全部崩塌 -72%~-81%。Task #68（L1 brand augmentation）Stage 4 R@10=0.0792（-18.4%）。**两次"扩展 SID 表达力"的尝试都失败**。

Task #71 核心发现：**Stage 2.2 numerical fidelity ≠ Stage 3 推荐 fidelity**（E 流 val_recon 最佳但 R@10 最差）。

可能的根因：baseline codebook_width=256 不够细，导致 item-level 区分度不足，量化误差传播到 Stage 3。如果把 codebook 翻倍到 512：
- 量化粒度提升 ~2x
- 重建 MSE 预计降低
- SID 共现模式更稳定

**假设 H1**: codebook_width=512 + cascade 重建误差更小 → Stage 3 R@10 提升

**对照基线**（Task #65）:
- Stage 2 RQ-VAE: num_hierarchies=3, codebook_width=256
- Stage 3 R@10 = **0.09710**

---

## 2. 实验设计

**唯一变量**: `codebook_width=512`（vs baseline 256）

**保持不变**:
- Stage 1 嵌入: `products/task65/sem_embeds/merged_predictions_tensor.pt` (11924, 2048)
- Stage 2.1 trainer 配置: rqvae_train_flat.yaml 其他 override 全部与 Task #65 一致
- Stage 2.2 → 3 bridge: 追加 1 列 dedup digit → (4, 11924)
- Stage 3 配置: num_hierarchies=4, sequence_length=120, seed=42
- Stage 4 + eval: 复用 Task #65 评估脚本 task388v4_s4_item_eval.py

**启动命令（Stage 2.1）**:
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/wenyu/GeneRec/GRID

EMB_PATH=/home/wlia0047/ar57/wenyu/GeneRec/logs/inference/runs/2026-07-16/21-56-20/pickle/merged_predictions_tensor.pt
DATA_DIR=/home/wlia0047/wenyu/GeneRec/GRID/data/amazon_data/toys

python -m src.train experiment=rqvae_train_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=512 \
    seed=42 \
    paths.root_dir=/home/wlia0047/wenyu/GeneRec/GRID
```

**Stage 2.2 推断 + bridge + Stage 3 + Stage 4 + 评估**: 复用 Task #65 pipeline 脚本，仅替换 codebook 路径。

---

## 3. 决策触发（vs baseline R@10=0.09710）

| 指标条件 | R@10 区间 | 决策 |
|----------|-----------|------|
| R@10 ∈ [0.090, 0.105] | — | ✅ baseline 复现（无效对照）|
| R@10 > 0.105 | R@10 ∈ (0.105, ∞) | ✅ **H1 confirmed** → codebook_size 是提升方向 |
| R@10 ∈ [0.080, 0.090) | — | ⚠️ 部分确认 → codebook 翻倍不够，需结合其他改进 |
| R@10 < 0.080 | — | ❌ H1 否证 → codebook 容量非瓶颈，转向其他方向 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 2.1 RQ-VAE 训练 (codebook_width=512, 3000 steps) | ~3h |
| Stage 2.2 rkmeans_inference_flat | ~5 min |
| Bridge (4, 11924) | ~10 min |
| Stage 3 TIGER 训练 (3000 steps) | ~3h |
| Stage 4 inference + R@10 eval | ~10 min |
| **总计** | ~6.5h（1 卡跑全流程）|

> 💡 **并行机会**: 4 GPU 全部空闲，可分阶段并行（Stage 2.1 占 1 卡，剩 3 卡待机；Stage 3 占 1 卡，剩 3 卡待机）。

---

## 5. 风险与缓解

**风险 1**: codebook 翻倍 → 训练时间翻倍（每 step kmeans 时间 O(K)，K=512 vs K=256）
→ **缓解**: val_check_interval=3000（仅 step 3000 验一次），实际是 end-of-training 验

**风险 2**: Stage 3 训练 2875 步过拟合（baseline 现象）
→ **缓解**: Stage 3 加 early stopping patience=3 或 val_check_interval=500，更细粒度监控

**风险 3**: R@10 与 baseline 几乎相同 → 浪费 GPU
→ **缓解**: 若 Stage 3 val/recall@5 在 1500 步仍 < baseline 同期 → kill 提前终止

---

## 6. 完成度跟踪

- [ ] Stage 2.1 launch (codebook_width=512)
- [ ] Stage 2.1 finish (3000 steps, val_recon 报告)
- [ ] Stage 2.2 launch (rkmeans_inference_flat)
- [ ] Bridge (4, 11924) dedup SID 生成
- [ ] Stage 3 TIGER 训练
- [ ] Stage 4 inference
- [ ] R@10 eval (vs baseline 0.0971)
- [ ] 写 verdict + 更新 P5 paper section