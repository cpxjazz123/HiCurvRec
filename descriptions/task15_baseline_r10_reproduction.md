# Task #65 — 基线 R@10 完整流水线复现

> **任务目的**: 从零开始跑通完整 GRID 三阶段流水线，在 Toys 数据集上复现 RQ-VAE 基线 R@10。确认恢复后的 configs 可用，建立干净的对照基准。

> **完成日期**: (in progress)
> **状态**: 🟢 在跑 — Stage 1 进行中

---

## 1. 背景

当前基线 R@10=0.0973（E-E-E-E，task17）来自 legacy 评估脚本。自 configs 目录恢复后，尚未从头到尾跑通过完整流水线。

所有几何类改进路线（T59-64）已全部否证。启动任何新方向前，需要先确认基础流水线可正常工作，并获得一个干净的对照基线。

**关键参数**：
- Stage 2: RQ-VAE，`num_hierarchies=3`，`codebook_width=256`，`trainer.max_steps=15000`
- Stage 2.2: 推断后追加去重 digit 列 → L=4
- Stage 3: TIGER encoder-decoder，`num_hierarchies=4`
- 数据：Toys（11924 items, 19412 users）

---

## 2. 实验设计

### Stage 1：LLM 语义嵌入

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

python -m src.inference experiment=sem_embeds_inference_flat \
    data_dir=data/amazon_data/toys
```

- 模型：`google/flan-t5-xl`（2048-dim）
- 输出：`merged_predictions_tensor.pt` shape `(11924, 2048)`

### Stage 2.1：RQ-VAE 训练

```bash
python -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=<Stage1_output> \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.max_steps=15000
```

### Stage 2.2：RQ-VAE 推断（SID 生成）

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/toys \
    ckpt_path=<Stage2.1_ckpt> \
    embedding_path=<Stage1_output> \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256
```

### Stage 2.2 → 3 桥接：追加去重 digit

SID tensor shape `(3, 11924)` → 追加去重列 → `(4, 11924)`，用 `num_hierarchies=4` 传入 Stage 3。

### Stage 3：TIGER 训练

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=<L4_SID_tensor> \
    num_hierarchies=4 \
    sequence_length=120
```

### Stage 4：TIGER 推断

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=<L4_SID_tensor> \
    num_hierarchies=4
```

### 评估

```bash
python scripts/task388v4_s4_item_eval.py
```

---

## 3. 决策触发（vs legacy R@10=0.0973）

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| R@10 ∈ [0.090, 0.105] | 基线可复现 | ✅ 流水线正常，基线确认 |
| R@10 ∈ [0.070, 0.090) | 部分接近 | ⚠️ 有差异但接近，需分析原因 |
| R@10 < 0.070 | 显著偏低 | ❌ 流水线/配置有问题，需诊断 |
| Stage 任一失败 | — | ❌ 流水线断点，需修复 |

---

## 4. 预算

| 阶段 | 估算时间 | GPU 显存 |
|------|---------|---------|
| Stage 1: 语义嵌入 | ~20-30 min | <8GB |
| Stage 2.1: RQ-VAE 训练 | ~1-1.5h | <8GB |
| Stage 2.2: SID 推断 | ~5 min | <4GB |
| Stage 3: TIGER 训练 | ~3-5h | ~29GB |
| Stage 4: TIGER 推断 | ~5 min | <16GB |
| R@10 评估 | ~3 min | CPU |
| 总计 | ~5-7h | — |

---

## 5. 风险与缓解

**风险 1**: Stage 1 需从 HuggingFace 下载 flan-t5-xl（~3B 参数），首次需联网。
→ 缓解：模型会缓存到 `~/.cache/huggingface/`，后续无需网络。

**风险 2**: RQ-VAE DDP 策略在多卡时可能崩溃（已知 `ddp_find_unused_parameters_true` 问题）。
→ 缓解：如果 DDP 失败，降级为 `trainer.strategy=auto trainer.devices=1`。

**风险 3**: TIGER 训练 320k steps 耗时过长。
→ 缓解：early_stopping patience=10 监控 val/recall@5，达标可提前停。

**风险 4**: 评估脚本路径依赖 legacy 产物结构。
→ 缓解：手动调整评估脚本输入路径指向本任务产物。

---

## 6. 完成度跟踪

- [ ] Stage 1: sem_embeds_inference_flat
- [ ] Stage 2.1: RQ-VAE 训练 15000 steps
- [ ] Stage 2.2: SID 推断
- [ ] Stage 2.2 → 3: 追加去重 digit 列
- [ ] Stage 3: TIGER 训练
- [ ] Stage 4: TIGER 推断
- [ ] R@10 评估
- [ ] 写 verdict → `verdicts/task65_result.md`
