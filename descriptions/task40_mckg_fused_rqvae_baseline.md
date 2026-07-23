# Task #40 — fused_64d 直接喂标准 RQ-VAE (PM-RQ 对照组)

> **任务目的**: 把 MCKG fused_item (11924, 64) 直接作为 Stage 1 embedding 替代 flan-t5-xl (11924, 2048),跑标准 RQ-VAE,评估端到端 R@5。这是 PM-RQ 设计的对照组。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (cuda:0, ~30 min 训练 + 5 min 推断 + 5 min 评估)

---

## 1. 背景

承接 Task #36: MCKG fused_64d gap=+0.0298 (STRONG, 99% CI [+0.0264, +0.0353]),比 T5 baseline (Task #87) 的 Stage 1 embedding (flan-t5-xl 2048d) 的 dense retrieval 信号更强。

**关键问题**: "仅仅换成 MCKG embedding(不改 RQ-VAE 量化方式)能带来多少提升?" 

这是 PM-RQ 实证链的第二层:
```
T5-baseline RQ-VAE  <  MCKG-fused 标准 RQ-VAE (Task #40)  <  MCKG-fused geometry-aware RQ-VAE (Task #41)
```

**意义**: 如果 Task #40 端到端 R@5 > Task #87 baseline (0.01937),说明 MCKG embedding 是更优的 Stage 1 替代品(不需改 RQ-VAE);如果 Task #41 再有额外增益,则归功于 geometry-aware 设计本身。

## 2. 实验设计

**变量**: Stage 1 embedding 来源 (flan-t5-xl 2048d → MCKG fused_item 64d)
**保持不变**:
- RQ-VAE 配置 (snap-research 默认: [768,256,128]→64 arch + Adagrad lr=1e-3 + bf16 + batch=2048 + init_buf=3072)
- num_hierarchies=3 + codebook_width=256
- Stage 3 TIGER 配置 (与 Task #87 完全一致, 不引入新变量)

**调整项** (因 embedding dim 64 ≠ 2048):
- Encoder 第一层 input dim: 768 → 64 → `[64,32,16,8]→8` (decoder mirror: `[8,16,32,64]→64`)
- 数据 batch 处理: 把 fused_item 当作 "Stage 1 输出", 跳过 Stage 1

**norm 预处理 (Task #43 结论, 必须做)**:
```python
# fused_64d L2 normalize 到 p95 = 1.3962 (Task #43 建议)
norms = torch.norm(fused_item, dim=1, keepdim=True)
fused_normalized = fused_item / norms.clip(min=1e-6) * 1.3962
# 或更稳: fused_normalized = fused_item / norms.clip(min=1e-6)  # 单位球
```
⚠️ **不做这步**, codebook 会往 3-19 个 norm>50 的离群 item 坍缩,训练崩溃。

**启动命令**:
```bash
# Stage 2.1: RQ-VAE 训练 (3000 steps)
CUDA_VISIBLE_DEVICES=0 python3 -m src.train experiment=rqvae_train_flat \
  data_dir=data/amazon_data/toys \
  embedding_path=products/task40_mckg_fused_rqvae/fused_item_64d.pt \
  embedding_dim=64 num_hierarchies=3 codebook_width=256 \
  model.encoder.dim_per_layer=[64,32,16,8] \
  model.decoder.dim_per_layer=[8,16,32,64] \
  model.beta=0.25 \
  trainer.max_steps=3000 \
  data.batch_size=2048 \
  optim.lr=0.001 \
  task_name=task40_s2_train

# Stage 2.2: RQ-VAE 推断
CUDA_VISIBLE_DEVICES=0 python3 -m src.inference experiment=rkmeans_inference_flat \
  data_dir=data/amazon_data/toys \
  embedding_path=products/task40_mckg_fused_rqvae/fused_item_64d.pt \
  embedding_dim=64 num_hierarchies=3 codebook_width=256 \
  task_name=task40_s2_infer

# Stage 3: TIGER 训练 (复用 Task #87 config, 用 task40 SID)
CUDA_VISIBLE_DEVICES=0 python3 -m src.train experiment=tiger_train_flat \
  data_dir=data/amazon_data/toys \
  semantic_id_path=products/task40_mckg_fused_rqvae/stage2_infer/cluster_ids.pt \
  sequence_length=120 num_hierarchies=4 seed=42 \
  task_name=task40_s3_train

# Stage 4: 推断 + R@5/R@10 评估
CUDA_VISIBLE_DEVICES=0 python3 -m src.inference experiment=tiger_inference_flat \
  data_dir=data/amazon_data/toys \
  semantic_id_path=products/task40_mckg_fused_rqvae/stage2_infer/cluster_ids.pt \
  task_name=task40_s4_infer
```

## 3. 决策触发 (vs Task #87 flat baseline R@5=0.01937)

| Task #40 端到端 R@5 | 解读 | 后续 |
|---|---|---|
| R@5 ≥ 0.025 (≥ +29% vs Task #87) | MCKG embedding 是显著更优的 Stage 1, Task #41 有价值 | ✅ 进入 Task #41 |
| 0.014 ≤ R@5 ≤ 0.025 (baseline 区间) | MCKG 与 T5 等价, 信息已被 RQ-VAE 损失掉 | ⚠️ Task #41 改为 ablation (sphere/euclid/hyperbolic 单子空间) |
| R@5 < 0.014 (低于 baseline 下限) | MCKG 信息密度不够 64d, 或 fuse 损失曲率 | ❌ 跳过 Task #41, 退回 Task #22 报告 |

## 4. 预算

| 阶段 | 估算 | GPU |
|---|---|---|
| 写 fused_item_64d.pt (Task #38 已落盘) | (已完成) | — |
| Stage 2.1 RQ-VAE 训练 (3000 steps) | ~30 min | cuda:0 |
| Stage 2.2 RQ-VAE 推断 | ~5 min | cuda:0 |
| Stage 3 TIGER 训练 (10000 steps early-stop) | ~2.5 h | cuda:0 |
| Stage 4 推断 + 评估 | ~5 min | cuda:0 |
| **总计** | **~3.3 h (主在 Stage 3)** | cuda:0 单卡 |

## 5. 风险与缓解

**风险 1**: Encoder dim_per_layer [64,32,16,8] 与 snap-research 默认 [768,256,128] 不同, 可能需要调 beta/lr/init_buf → 缓解: 复用 Task #87 已验证的 v6 recipe (Adagrad lr=1e-3 bf16 init_buf=3072 batch=2048)
**风险 2**: fused_item L2 norm 不归一 (mean=0.79, max=381) 可能影响 RQ-VAE 数值稳定性 → 缓解: 在 Stage 2.1 之前 L2 normalize
**风险 3**: Stage 3 训练时间超预算 (2.5h) → 缓解: 早停 (val/recall@5 plateau 10 eval) 或 stage 3 max_steps=20000

## 6. 完成度跟踪

- [ ] Stage 2.1 RQ-VAE 训练 (3000 steps)
- [ ] Stage 2.2 RQ-VAE 推断 (cluster_ids.pt)
- [ ] Stage 3 TIGER 训练 (early-stop)
- [ ] Stage 4 推断 + 评估 (R@5/R@10/NDCG)
- [ ] 对比 Task #87 baseline, 写 verdict