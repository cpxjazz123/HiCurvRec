# Task #79 — Phonism RQ-VAE Gromov δ-Hyperbolicity (4-point, per-layer)

> **任务目的**: 复用 Task #92 的 4-point δ-hyperbolicity 算法, 对 phonism-style RQ-VAE (SINKHORN last layer, embed_dim=32) 测量每层残差 δ 值, 与 vanilla RQ-VAE (Task #92) 横向对比, 看 SINKHORN/32d embedding 是否让残差空间"更树状"
>
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #92 (vanilla RQ-VAE δ-hyperbolicity 测量):
- vanilla RQ-VAE 在 Musical_Instruments SASRec 128d 上, 残差空间 δ_max 严格递减 5.39 → 4.93 → 3.70 → 2.74 (-49%)
- 结论: RQ-VAE 逐层残差机制让空间结构越来越像树 (hyperbolic)
- 但 vanilla 用的是 128d SASRec embedding + e_dim=128

Task #126 复现过 phonism (genrec) RQ-VAE, 配置:
- input_dim=768, embed_dim=32, hidden=[512,256,128,64], codebook=256x3
- layer 0,1: STE (sk_epsilon=0)
- layer 2: SINKHORN (sk_epsilon≈0.003)
- Toys 下游 R@10=0.0298

**假设 H1**: phonism-style (e_dim=32, SINKHORN last) 比 vanilla (e_dim=128, pure VQ) 的 δ_max 更低, 因为:
- 32d 强制紧凑表示 → 减少 dead space → 残差更聚类
- SINKHORN 让 last codebook 利用率均匀 → 残差方差更小

**假设 H2**: 否则, 即 e_dim 才是 δ 的关键驱动, SINKHORN 影响有限

由于 phonism Toys checkpoint 已丢失 (目录 `/home/wlia0047/ar57/wenyu/genrec/` 不存在),
本任务改为: 用 Musical_Instruments sentence-t5-768d.npy 重新训练 phonism-style RQ-VAE (~5000 steps), 再做 δ 分析。

**重要约束**: 输入数据从 Toys (11924 items) 改为 Musical_Instruments (24588 items), 与 Task #92 输入 (Musical_Instruments SAS-128d) 数据集相同但 embedding 不同 (768d vs 128d)。结论应理解为"e_dim/SINKHORN 几何效应", 而非"vanilla vs phonism 严格 head-to-head"。

---

## 2. 实验设计

**变量**: RQ-VAE 架构差异
- Task #92 (vanilla): input=128, e_dim=128, hidden=[512,256], codebook=256×3, **sk_epsilons=[0,0,0]** (pure VQ)
- Task #79 (phonism-style): input=768, e_dim=32, hidden=[512,256,128,64], codebook=256×3, **sk_epsilons=[0,0,0.003]** (SINKHORN last layer)

**保持不变**:
- 4-point δ 算法 (Task #92 `gromov_delta_4point`, 5000 samples, seed=42)
- 输入数据集: Musical_Instruments items
- 训练: 5000 epochs (brief), batch_size=1024, lr=1e-3, AdamW, MSE loss

**启动命令**:
```bash
# Step 1: 训练 phonism-style RQ-VAE (brief, ~5-8 min on GPU 2)
cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/RQVAE
CUDA_VISIBLE_DEVICES=2 python -u main.py \
  --lr 1e-3 \
  --epochs 5000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 \
  --sk_epsilons 0.0 0.0 0.003 \
  --layers 512 256 128 64 \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device cuda:0 \
  --kmeans_init True \
  --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy \
  --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task79_phonism_rqvae_768d/rqvae_ckpt

# Step 2: δ-hyperbolicity 测量 (复用 Task #92 算法)
cd /home/wlia0047/ar57/wenyu/GeneRec
python scripts/task79_phonism_delta_hyperbolicity.py \
  --ckpt products/task79_phonism_rqvae_768d/rqvae_ckpt/<TIMESTAMP>/best_loss_model.pth \
  --emb external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy \
  --num_samples 5000 \
  --output verdicts/task79_phonism_rqvae_delta_hyperbolicity.json
```

---

## 3. 决策触发(vs Task #92 vanilla baseline)

| 指标条件 | δ_max L3 vs vanilla (Task #92: 2.74) | 决策 |
|----------|---------------------------------------|------|
| 大幅更低 (< 2.0) | -27% 或更多 | **H1 成立**: SINKHORN/e_dim=32 让残差更树状, 下游有几何基础优势 |
| 相当 (2.0-3.0) | ±20% | **部分成立**: e_dim/SINKHORN 影响有限, δ 主要由数据集本身决定 |
| 显著更高 (> 3.5) | +28% | **H2 成立**: 32d 强制压缩让空间欠表示, δ 上升 (注意 δ 上升可能是过度压缩, 不一定是好事) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 训练 phonism-style RQ-VAE (5000 epochs) | ~8 min on GPU 2 |
| δ-hyperbolicity 测量 (5000 samples × 4 layers) | ~2 min on GPU 2 |
| 写 verdict + 横向对比分析 | ~2 min |
| 总计 | ~12 min |

---

## 5. 风险与缓解

**风险 1**: phonism 配置 + 5000 epochs 可能未收敛, 导致 δ 曲线不显著
→ 缓解: 监控 train loss, 末 1000 epochs 仍在下降则延长到 8000 epochs

**风险 2**: e_dim=32 + SINKHORN 可能让 codebook collapse (利用率<30%), 影响 δ 解释
→ 缓解: 跑完后打印每层 codebook 利用率, 不 collapse 才纳入对比

**风险 3**: GPU 2 被新任务抢用 (R7 约束)
→ 缓解: 启动前 `nvidia-smi --query-gpu=utilization.gpu,memory.used` 确认空闲

---

## 6. 完成度跟踪

- [ ] Task #79 description 落盘 (本文档)
- [ ] 训练 launcher `scripts/task79_train_phonism_rqvae.sh`
- [ ] 训练完成 (`products/task79_phonism_rqvae_768d/rqvae_ckpt/<TS>/best_loss_model.pth` 落盘)
- [ ] δ-hyperbolicity 脚本 `scripts/task79_phonism_delta_hyperbolicity.py`
- [ ] 跑 δ 测量, 落盘 JSON
- [ ] 与 Task #92 vanilla 横向对比, 写 verdict `verdicts/task79_phonism_rqvae_delta_hyperbolicity_result.md`
- [ ] 更新 loop.md §16 (按 R8 清理已完成)