# Task 13：Codebook 维度 (L×W) 消融实验（Toys, paper Table 3）

> **目的**：复现 GRID 论文 Section 4.2 Table 3 的 **Codebook L×W 维度** 消融结论
> **数据集**：仅 Amazon Toys（论文 Table 3 仅 Beauty，但我们 Toys 复用 RK-Means 同样可做）
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

---

## 总目标

论文 Table 3 Beauty 行（**注意 paper 跑的是 Beauty，不是 Toys**，但 Toys 同样可复现）：
| L × W | R@5 | R@10 | N@5 | N@10 |
|-------|-----|------|-----|------|
| 3 × 128 | 0.0412 | 0.0617 | 0.0273 | 0.0339 |
| **3 × 256** (default) | **0.0422** | **0.0639** | **0.0277** | **0.0347** |
| 3 × 512 | 0.0415 | 0.0631 | 0.0273 | 0.0342 |
| 2 × 256 | 0.0403 | 0.0618 | 0.0264 | 0.0333 |
| 4 × 256 | 0.0405 | 0.0609 | 0.0265 | 0.0331 |
| 5 × 256 | 0.0396 | 0.0596 | 0.0257 | 0.0321 |

**关键发现**（论文 Section 4.2 第 105 行原文）：
> "We vary the number of residual layers L and tokens per layer W in RK-Means, and observe that the default choice of (L, W) = (3, 256) leads to the best recommendation performance. **Surprisingly, performance drops substantially with more layers**, although additional layers convey more semantic information to the recommendation model. This points to a trade-off between SID sequence learnability and the amount of semantic information contained in the SIDs."

**任务目标**：在 Toys 数据集上验证 **(L=3, W=256) 是否最优**，并验证"层数增加性能下降"的反直觉发现。

**总 run 数**：5 × 3 = 15（5 个 (L×W) 变体 × {Stage 2.1 + Stage 2.2 + Stage 3}，复用 task.md 的 Stage 1 + Stage 4）

> 节省：5 个变体只需要 Stage 2 + Stage 3 训练/推断，Stage 1 嵌入可全程复用；Stage 4 评估可逐个跑（每个 ~30 min）。

---

## 复用约定

| 产物 | 来源 | 路径 |
|------|------|------|
| Stage 1 LLM embedding | 复用 task.md | `logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt` (11924, 2048) |

---

## 5 个 (L, W) 变体

| L | W | 对应 paper row | 新增 Stage 2.1 ckpt | 新增 Stage 2.2 SID | 新增 Stage 3 ckpt |
|---|---|----------------|---------------------|---------------------|-------------------|
| **3** | **128** | "3×128" | `<run1>/checkpoints/...` | shape (4, 11924) | `<run1_stage3>/checkpoints/...` |
| **3** | **256** (default) | "3×256" | 复用 task1 | 复用 task1 | 复用 task1 |
| **3** | **512** | "3×512" | `<run2>/checkpoints/...` | shape (5, 11924) | `<run2_stage3>/checkpoints/...` |
| **2** | **256** | "2×256" | `<run3>/checkpoints/...` | shape (3, 11924) | `<run3_stage3>/checkpoints/...` |
| **4** | **256** | "4×256" | `<run4>/checkpoints/...` | shape (5, 11924) | `<run4_stage3>/checkpoints/...` |
| **5** | **256** | "5×256" | `<run5>/checkpoints/...` | shape (6, 11924) | `<run5_stage3>/checkpoints/...` |

> ⚠️ 注意 Stage 2.2 输出的 SID shape = (L+1, 11924)（L 个 layer digit + 1 个 de-dup digit），不是固定的 (4, 11924)。Stage 3 必须传对应的 `num_hierarchies=L+1`。

**Stage 2.1 训练步数** = `L × 1000`（论文 Section 3.1: "RK-Means and R-VQ are trained layer-wise for 1k steps per layer"）

---

## Stage 2.1 run1：L=3, W=128

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

EMB_PATH=logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt

nohup python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=$EMB_PATH \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=128 \
    trainer.max_steps=3000 \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    > logs/$(date +%Y%m%d_%H%M%S)_task11_s21_l3w128.log 2>&1 &
```

---

## Stage 2.2 run1：L=3, W=128

```bash
nohup python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=$EMB_PATH \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=128 \
    ckpt_path=logs/<s21_run1_ts>/checkpoints/checkpoint_*.ckpt \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    > logs/$(date +%Y%m%d_%H%M%S)_task11_s22_l3w128.log 2>&1 &
```

---

## Stage 3 run1：L=3, W=128 TIGER

```bash
SID_L3W128=logs/<s22_run1_ts>/pickle/merged_predictions_tensor.pt

nohup python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=$SID_L3W128 \
    num_hierarchies=4 \
    sequence_length=120 \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    > logs/$(date +%Y%m%d_%H%M%S)_task11_s3_l3w128.log 2>&1 &
```

> Stage 3 的 `num_hierarchies` 永远是 L+1=4（W 不影响），但**每个 L 需要单独训练**（因为 L=3 vs L=4 给出不同长度的 SID 序列）。

---

## 4 个其他 (L, W) 变体

替换 `num_hierarchies` / `codebook_width` 即可：

| 变体 | num_hierarchies | codebook_width | trainer.max_steps | Stage 3 num_hierarchies |
|------|-----------------|----------------|-------------------|--------------------------|
| L=3, W=512 | 3 | 512 | 3000 | 4 |
| L=2, W=256 | 2 | 256 | 2000 | 3 |
| L=4, W=256 | 4 | 256 | 4000 | 5 |
| L=5, W=256 | 5 | 256 | 5000 | 6 |

---

## Stage 4：5 个变体的推断 + 评估

每个 Stage 3 ckpt 都需要 Stage 4 推断 + 评估，复用 `task4_free_form_eval.py`：

```bash
for VARIANT in l3w128 l3w512 l2w256 l4w256 l5w256; do
    NUM_HIER=$(case $VARIANT in
        l3w128) echo 4 ;;
        l3w512) echo 4 ;;
        l2w256) echo 3 ;;
        l4w256) echo 5 ;;
        l5w256) echo 6 ;;
    esac)
    nohup python -m src.inference experiment=tiger_inference_flat \
        data_dir=data/amazon_data/toys \
        semantic_id_path=logs/<s22_${VARIANT}_ts>/pickle/merged_predictions_tensor.pt \
        ckpt_path=logs/<s3_${VARIANT}_ts>/checkpoints/best_*.ckpt \
        num_hierarchies=$NUM_HIER \
        sequence_length=120 \
        trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
        > logs/$(date +%Y%m%d_%H%M%S)_task11_s4_${VARIANT}.log 2>&1 &
done
```

---

## 完成指标

| 指标 | 目标 | 验证 |
|------|------|------|
| 5 个 Stage 2.1 ckpt | 5 个 checkpoint_*.ckpt | ls checkpoints/ |
| 5 个 Stage 2.2 SID tensor | shape (L+1, 11924) | torch.load |
| 5 个 Stage 3 ckpt | 5 个 best_*.ckpt | ls checkpoints/ |
| 5 个 Stage 4 tensor | shape (19412, 10, L+1) | torch.load |
| (L=3, W=256) 是否最优 | 5 行 R@10 对比表 | 评估 JSON |
| L 增加性能下降趋势 | L=2→L=5 R@10 持续 ↓ | 评估 JSON |

---

## 风险与回退

1. **`codebook_width=128` / 512 / 1024 是否被 yaml 接受** → GRID 默认 256，可能 hardcode；先编辑 yaml 改 `codebook_width: ???` 为变量
2. **W=128 时码本利用率低**（仅 128 个码字）→ Layer 0 利用率 < 50% 是预期的，论文接受这点
3. **L=5, W=256 Stage 3 显存 OOM**（SID 序列长度 = 6）→ 降 `sequence_length=60` 或 `batch_size_per_device=8`
4. **`num_hierarchies=L+1` 在 Stage 3 不灵活** → 直接编辑 yaml
5. **总耗时评估**：5 个变体 × (Stage 2.1 5-10 min + Stage 2.2 2-3 min + Stage 3 早停 ~30-60 min + Stage 4 20 min) = **~5-8h 总耗时**（单 GPU 串行）

---

## 与 task1 的关系

| (L, W) | 实测 R@10 | 论文 R@10 (Beauty) | 是否 default |
|--------|-----------|---------------------|---------------|
| **3, 256** | 0.0647 (task1, val) | 0.0639 (Beauty) | ✅ default |
| 3, 128 | tbd | 0.0617 | |
| 3, 512 | tbd | 0.0631 | |
| 2, 256 | tbd | 0.0618 | |
| 4, 256 | tbd | 0.0609 | |
| 5, 256 | tbd | 0.0596 | |

> 复现判定：
> 1. (L=3, W=256) 是否在 Toys 上也是最优（论文 Beauty R@10=0.0639 是 6 个变体中最高）
> 2. L=5 vs L=2 的 R@10 差距是否 ≥ 0.002（论文 0.0022 差距）
> 3. W=512 vs W=256 是否有微跌（论文 0.0008 跌）
> 如果 3 个判定全部命中 → 完全复现 paper Table 3；如果有任何反转 → 反直觉发现
