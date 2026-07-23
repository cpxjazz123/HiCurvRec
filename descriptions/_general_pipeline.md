# GRID 流水线实现任务书（Toys 数据集 / RQ-VAE 专用版）

> **量化算法：仅使用 RQ-VAE**（Residual Quantization VAE）
> **数据集：仅 Amazon Toys**（最小数据集，2.4GB / 480 文件 / 约 12,000 items）
> 目标仓库：`/home/wlia0047/ar57/wenyu/GeneRec/GRID`
> 数据目录：`/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys`
> 环境：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`（**GeneRec 专用环境，已准备完毕**）
> 总入口：`python -m src.train` / `python -m src.inference`，配置由 Hydra 管理（`configs/experiment/*.yaml`）

---

## 总体目标

跑通 GRID 三阶段流水线（**LLM Embedding → RQ-VAE 量化 SID → TIGER 生成式推荐**），在 **Amazon Toys** 数据集上达到 TIGER 论文可复现基线水平。

总实验数：**1 数据集 × 4 阶段 = 4 个 run**（阶段 1 LLM 嵌入 → 阶段 2 RQ-VAE 训练+推断 → 阶段 3 TIGER 训练 → 阶段 4 TIGER 推理）。

---

## 阶段 1：LLM 语义嵌入生成

### 1.1 目标
用 HuggingFace 上的 LLM（默认 `google/flan-t5-xl`，dim=2048）把 Toys 数据集每个 item 的文本字段转成稠密向量，作为 RQ-VAE 量化的输入。

### 1.2 执行命令
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

python -m src.inference experiment=sem_embeds_inference_flat \
    data_dir=data/amazon_data/toys
```

可调超参：`embedding_model=google/flan-t5-xl`（默认，输出维度 2048）、`batch_size_per_device=8`（视显存调整）。

### 1.3 产物
- `logs/<timestamp>/pickle/merged_predictions_tensor.pt`：所有 item 的嵌入张量，shape `(N_items, 2048)`，约 `(12000, 2048)`
- `logs/<timestamp>/pickle/<item_id>.pkl`：每个 item 的嵌入分片

### 1.4 阶段 1 完成指标

| 指标 | 目标值 | 校验方式 |
|------|--------|----------|
| 覆盖率 | **100%**（无遗漏 item） | `tensor.shape[0] == 数据集 item 总数` |
| 嵌入维度 | **2048**（flan-t5-xl） | `tensor.shape[1] == 2048` |
| NaN/Inf | **0 个** | `torch.isnan(t).sum() == 0 and torch.isinf(t).sum() == 0` |
| L2 范数 | **全部 > 0** | `(tensor.norm(dim=1) > 0).all()` |
| 重复 item | **0** | `tensor.unique(dim=0).shape[0] == tensor.shape[0]` |
| Toys 数据 item 数 | 约 **12,000** | 与 P5 论文一致 |
| 推理耗时 | **< 4 hour**（8 卡 A100） | wall-clock time |
| 显存峰值 | **< 40GB / GPU** | `nvidia-smi` |

---

## 阶段 2：RQ-VAE 残差量化学习 Semantic ID

> **唯一算法**：本项目只用 Residual Quantization VAE（RQ-VAE），不再跑 RKMeans / RVQ。

### 2.0 RQ-VAE 结构（来自 `configs/experiment/rqvae_train_flat.yaml`）

- **Encoder**：MLP，`2048 → 768 → 256 → 128 → 64`，ReLU
- **Decoder**：MLP，`64 → 128 → 256 → 768 → 2048`，ReLU
- **前置归一化**：`BatchNorm1d(2048) + NormalizeLayer`（在 encoder 之前）
- **量化层**：`VectorQuantization` + `STEQuantization` + `MiniBatchKMeans` 初始化（KMeans++）
- **损失**：`BetaQuantizationLoss(beta=0.25)` + `MSELoss(reduction=mean)`，`reconstruction_loss_weight=1.0`
- **优化器**：Adagrad(lr=0.001, weight_decay=0.0) + `WarmupLinearScheduler(warmup_steps=1000, min_ratio=0.01)`

### 2.1 RQ-VAE 训练

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

EMB_PATH_TOYS=logs/<stage1_ts_toys>/pickle/merged_predictions_tensor.pt

python -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=$EMB_PATH_TOYS \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256
```

**训练超参**（取自 yaml，已按论文修正）：`max_steps=15000`（论文要求 15k steps，RQ-VAE 比 RKMeans 需要更多迭代）、`batch_size_per_device=2048`、`precision=bf16-mixed`、`strategy=ddp_find_unused_parameters_true`、`sync_batchnorm=true`。

### 2.2 RQ-VAE 推断（生成 SID）

> RQ-VAE 训练后**复用 `rkmeans_inference_flat` 配置**给每个 item 分配 SID（SID 推断只需码本，与训练算法无关）。

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=$EMB_PATH_TOYS \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=logs/<stage2_train_ts_toys>/checkpoints/checkpoint_*.ckpt
```

### 2.3 产物
- 训练：`logs/<ts>/checkpoints/checkpoint_*.ckpt`
- 推断：`logs/<ts>/pickle/merged_predictions_tensor.pt`，shape `(N_items, 4)`，最后一列是去重后追加的 digit

### 2.4 阶段 2 完成指标

#### 训练阶段（RQ-VAE）

| 指标 | 目标值 | 校验方式 |
|------|--------|----------|
| 最终 `train/loss` | **< 0.5**（3000 step 内下降 ≥ 70%） | `csv log` 末 50 步均值 |
| 最终 `train/reconstruction_loss` | **< 0.3**（占总 loss < 60%） | `csv log` |
| 最终 `train/quantization_loss` | **< 0.2** | `csv log` |
| Beta commitment loss 占比 | **< 40%** 总损失 | `csv log` |
| 训练无 NaN | **0 个 NaN step** | `grep nan logs/<ts>/csv/` |
| 训练无崩溃 | **DDP 同步正常** | `trainer.strategy = ddp_find_unused_parameters_true` |
| BatchNorm 同步正常 | **sync_batchnorm=true 生效** | `csv log` 中无 `unnormalized` 警告 |
| 训练耗时 | **< 2 hour**（8 卡 A100） | wall-clock time |
| 显存峰值 | **< 32GB / GPU** | `nvidia-smi` |

#### 推断阶段（SID 质量）

| 指标 | 目标值 | 校验方式 |
|------|--------|----------|
| item 覆盖率 | **100%** | 输出 tensor 行数 == 输入 item 数（约 12,000） |
| 每层码本使用率 | **> 70%**（4 层均需达标） | 每层 unique ID 数 ≥ 179/256 |
| 重复 SID 数（去重前） | **< 5%** | `unique_rows / total_rows` |
| 重复 SID 数（去重后） | **< 1%** | 经 `deduplicate_rows_in_tensor` 后处理 |
| SID 形状 | `(N, 4)`（num_hierarchies+1） | `tensor.shape` |
| SID 取值范围 | **每列 ∈ [0, 256)** | `tensor.max() < 256` |
| 推断耗时 | **< 30 min** | wall-clock time |

> ⚠️ **关键约定**：`num_hierarchies=3` 训练，生成 SID 时**末尾追加 1 列去重 digit**，所以最终 SID 列数 = 4。传给阶段 3 时必须用 `num_hierarchies=4`。

---

## 阶段 3：TIGER 生成式推荐训练

### 3.1 目标
训练 Encoder-Decoder Transformer（T5 架构），按用户历史序列自回归生成下一个 item 的语义 ID 序列。

### 3.2 执行命令

```bash
SID_PATH_TOYS=logs/<stage2_inference_ts_toys>/pickle/merged_predictions_tensor.pt

python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SID_PATH_TOYS \
    num_hierarchies=4 \
    sequence_length=120
```

可调超参：`max_steps=320000`（默认）、`accumulate_grad_batches=16`（有效 batch=512）、`precision=32-true`、`devices=-1`（全部 GPU）。

### 3.3 产物
- `logs/<ts>/checkpoints/checkpoint_*.ckpt`：监控 `val/recall@10` 保存 top-1 权重（**按论文 Section 4 修正**：论文原文 checkpoint with best validation Recall@10）
- `logs/<ts>/csv/metrics.csv`：训练曲线

### 3.4 阶段 3 完成指标

#### 训练过程

| 指标 | 目标值 | 校验方式 |
|------|--------|----------|
| 总训练步数 | **320,000** | `trainer.max_steps` |
| 验证频率 | **每 1,600 步一次** | `val_check_interval` |
| 早停 patience | **10 次验证无提升**（以 `val/recall@10` 为准，论文 Section 4 原文：checkpoint with best validation Recall@10） | `early_stopping.patience=10` |
| 训练 loss 趋势 | **持续下降，无 plateau** | `csv log` 滑动平均 |
| GPU 利用率 | **> 80%** | `nvidia-smi dmon` |
| 单卡显存峰值 | **< 32GB**（A100） | `nvidia-smi` |
| 训练耗时 | **< 24 hour**（8 卡 A100） | wall-clock time |

#### 最终性能（TIGER 论文基线对标，使用 RQ-VAE SID）

| 数据集 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|--------|----------|-----------|--------|---------|
| Toys（RQ-VAE） | **≥ 0.034** | **≥ 0.051** | **≥ 0.022** | **≥ 0.028** |

> ⚠️ **重要修正**：以上为**论文 Table 1 RQ-VAE Toys 原文值**（R@5=0.0342, R@10=0.0514, N@5=0.0224, N@10=0.0280）。注意 RK-Means Toys 值更高（R@5=0.0376, R@10=0.0577），**task 指定只用 RQ-VAE，故目标值应取 RQ-VAE 栏**。Tiger 原论文 Toys R@5=0.0446、R@10=0.0679 作为参考上限。

---

## 阶段 4：TIGER 推荐生成与评估

### 4.1 执行命令

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SID_PATH_TOYS \
    ckpt_path=logs/<stage3_train_ts_toys>/checkpoints/checkpoint_*.ckpt \
    num_hierarchies=4 \
    sequence_length=120
```

### 4.2 产物
- `logs/<ts>/pickle/<user_id>.pkl`：每个用户的 top-10 语义 ID 推荐

### 4.3 阶段 4 完成指标

| 指标 | 目标值 | 校验方式 |
|------|--------|----------|
| 测试集用户覆盖率 | **100%** | 输出 user 数 == testing 子集 user 数 |
| 每用户推荐数 | **= 10**（top_k_for_generation） | `output.shape[-1] == 10` |
| 推理耗时 | **< 30 min**（8 卡 A100） | wall-clock time |
| 显存峰值 | **< 16GB / GPU** | `nvidia-smi` |
| Recall@10 (test) | **≥ 0.051**（RQ-VAE Toys 论文值） | `SIDRetrievalEvaluator.compute()` |
| Recall@5 (test) | **≥ 0.034** | 同上 |
| NDCG@5 (test) | **≥ 0.022** | 同上 |
| NDCG@10 (test) | **≥ 0.028** | 同上 |

---

## 跨阶段验收（Pipeline End-to-End）

| 检查项 | 目标 |
|--------|------|
| Toys 流水线串通 | 阶段 1→2→3→4 全跑通，无 OOM/NaN |
| 端到端可复现 | 固定 `seed=42`，两次运行指标差异 < 2% |
| 产物完整性 | 每阶段输出符合 schema（`tensor.shape`、dtype） |
| Hydra 配置覆盖 | 超参通过 CLI 覆盖生效（`data_dir`、`ckpt_path` 等） |
| RQ-VAE 单一算法 | 全流程**不出现 RKMeans / RVQ** 配置调用（`rkmeans_inference_flat` 仅用于 SID 推断） |
| 数据集单一性 | `data_dir` 始终指向 `data/amazon_data/toys` |

---

## 风险与回退点

1. **OOM in Stage 1** → 减小 `batch_size_per_device`（8 → 4 → 2）或换 `google/flan-t5-base`（dim=768）
2. **RQ-VAE 不收敛** → 检查 `embedding_dim=2048` 是否与阶段 1 LLM 输出一致；增大 `codebook_width` 到 512
3. **Codebook collapse in Stage 2**（某层 unique < 50/256）→ 增大 `init_buffer_size`（3072 → 8192）或重设随机种子
4. **TIGER 不收敛** → 检查 SID 是否含 `-1` padding、确认 `num_hierarchies=4` 而非 `3`、确认 SID path 指向**推断后的** `merged_predictions_tensor.pt`
5. **显存不够训练 TIGER** → 启用 `precision=bf16-mixed`、`accumulate_grad_batches` 调到 32
6. **评估指标低于基线 ≥ 20%** → 检查 L2 范数归一化、`codebook_width=256` 是否生效
7. **DDP 同步失败** → 确认 `sync_batchnorm=true` 已设置、CUDA visible devices 正确

---

## 进度跟踪模板

```
[✅] Toys: 阶段 1 ✅ → 阶段 2.1 ✅ → 阶段 2.2 ✅ → 阶段 3 ✅ → 阶段 4 ✅
```

每个 run 完成时打勾 ✅。

---