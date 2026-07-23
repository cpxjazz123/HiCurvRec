# Task 17 (Idea 3 收官)：gain-shape 精细融合 vs MMQ 纯方向 三方对照

> **状态**: 🟡 backlog

> **目的**：在完整 GRID 流水线上比较 3 种残差量化策略，**直接回答 Idea 3 应该以什么故事线收尾**。
>
> **背景**：Task 14 (衰减均匀) + Task 15 (新层 100 步内稳定) + Task 16 (gain 1D 分类 ≈ top-class 占比) 三个诊断都倾向于"gain 信息微弱"。但所有这些诊断都止于统计层面，**没在最终推荐指标上直接验证**。Task 17 把统计上的"微弱"翻译成下游 Recall@10/NDCG@10 的实际差异。
>
> **数据集**：Amazon Toys（11924 商品，flan-t5-xl 2048 维 embedding）
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

---

## 总目标

三个量化策略在**完全相同的设置**（同一批 embedding、同一个 train/test 切分、同样的 TIGER 架构、同样的训练 seed=42）下分别跑完 Stage 2 → 3 → 4 → eval，**直接比较 Recall@10 / NDCG@10 / 碰撞率**。

| 组 | 名称 | 残差处理 | 量化距离 | 实现路径 |
|----|------|----------|----------|----------|
| **A** | 标准 RQ-VAE/RKMeans | 不归一化（magnitude 保留） | 欧氏距离 | `model.normalize_residuals=false`（已有 CLI 开关） |
| **B** | MMQ 纯方向 | 归一化到单位球面（gain 丢弃） | 归一化欧氏 = 余弦 | `model.normalize_residuals=true`（yaml 默认） |
| **C** | GSRQ gain-shape 融合 | 不归一化 + centroid update 显式跟踪 gain | 欧氏距离（centroid 保持 magnitude） | **新代码**：`MiniBatchKMeans.use_gain_tracking=True` |

**关键判断**：
- 组 C 比 B/A 显著更好（Recall@10 差 > 0.005，超出 3-4% 种子噪声幅度）→ Idea 3 故事线："提出新方法"
- 组 C ≈ B（差 < 0.003）→ Idea 3 故事线改为"审计/诊断型"：系统性检验残差量化的几个假设，证伪了 2 个（gain 携带大部分信息 / 深层更不平稳），部分确认了 1 个（方向主导）；并验证 MMQ 纯方向策略已逼近这个具体场景下的天花板。
- **碰撞率**无论 Recall@10 结论如何，都是一个独立的次要贡献（GSRQ 原始动机：防止质心收缩导致方向损失）。

---

## 输入

### 共享资源（3 组共用）
- **Stage 1 embedding**：`/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt` (11924, 2048)
- **TIGER Stage 3 训练配置**：`configs/experiment/tiger_train_flat.yaml`（直接复用，不改）
- **TIGER Stage 4 推断配置**：`configs/experiment/tiger_inference_flat.yaml`（直接复用）
- **评估脚本**：`task_artifacts/scripts/task4_free_form_eval.py`（扩展为加碰撞率字段）

### 仅 Group A 特殊
- Hydra override：`model.normalize_residuals=false`

### 仅 Group B 特殊
- Hydra override：默认（不传 `model.normalize_residuals`）

### 仅 Group C 特殊
- 新增 yaml：`configs/experiment/rkmeans_train_gsrq.yaml`（继承 `rkmeans_train_flat` 后改 model.use_gain_tracking=True）
- 新增 yaml：`configs/experiment/rkmeans_inference_gsrq.yaml`
- 改 `src/models/modules/clustering/mini_batch_kmeans.py`：
  - 新增参数 `use_gain_tracking=False`
  - 新增 tensor `cluster_gains: nn.Parameter(torch.ones(n_clusters))` 跟踪每个 cluster 的最近 gain
  - 改 `centroid_update()`：update 后用 `centroids = F.normalize(centroids, dim=-1) * cluster_gains.unsqueeze(-1)` 重新 scale
- 改 `src/modules/clustering/residual_quantization.py`：
  - `forward()` 里：当 `use_gain_tracking=True` 时，强制 `normalize_residuals=False`（因为要保留 magnitude 才能让 centroid update 有意义）

---

## 完整流水线（每组都要跑一遍）

```
                    ┌─────────────┐
[复用 Stage 1] ──→  │  Stage 2.1  │  RKMeans 训练 (max_steps=3000, num_hierarchies=3, W=256)
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │  Stage 2.2  │  SID 推断 + 碰撞率测量
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │  Stage 3    │  TIGER 训练 (复用 tiger_train_flat，num_hierarchies=4)
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │  Stage 4    │  TIGER 推断
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │   eval      │  R@10, NDCG@10, 碰撞率
                    └─────────────┘
```

每组 4 个 run × 3 组 = **12 个 run**（实际：Group A/B 复用现有 baseline 的 Stage 3/4 yaml，只有 Group C 需要全套新 run）。

---

## 具体做法（按组）

### Group A (baseline, `normalize_residuals=false`)
- **Stage 2.1**：`python -m src.train experiment=rkmeans_train_flat model.normalize_residuals=false ...`
- **Stage 2.2**：`python -m src.inference experiment=rkmeans_inference_flat model.normalize_residuals=false ckpt_path=$S21_CKPT ...`
- **Stage 3 / 4**：和 baseline 一样（tiger_train_flat / tiger_inference_flat）

### Group B (MMQ, `normalize_residuals=true`)
- 完全和 Group A 一样的脚本，只是不传 `model.normalize_residuals`（用 yaml 默认）

### Group C (GSRQ)
- **Stage 2.1**：`python -m src.train experiment=rkmeans_train_gsrq ...`（新 yaml）
- **Stage 2.2**：`python -m src.inference experiment=rkmeans_inference_gsrq ...`
- **Stage 3 / 4**：和 A/B 一样

---

## 评估指标（3 组共同）

| 指标 | 来源 | 含义 |
|------|------|------|
| **Recall@10** | `task4_free_form_eval.py` 输出 | 主指标 |
| **Recall@5** | 同上 | 副指标 |
| **NDCG@10** | 同上 | 排序质量 |
| **NDCG@5** | 同上 | 同上 |
| **碰撞率** | Stage 2.2 SID tensor 直算 | 1 - n_unique_SIDs / N |

**碰撞率公式**：
```python
sid = torch.load("merged_predictions_tensor.pt")  # (N, H)
n_unique = torch.unique(sid, dim=0).shape[0]
collision_rate = 1 - n_unique / N
```

---

## 完成指标

| 指标 | 目标 | 验证 |
|------|------|------|
| 3 组各自 Stage 2.1 ckpt | 3 个 checkpoint_000_003000.ckpt | 文件存在 |
| 3 组各自 Stage 2.2 SID | 3 个 merged_predictions_tensor.pt, shape (11924, 3) | torch.load |
| 3 组各自 Stage 3 TIGER ckpt | 3 个 best_*.ckpt | 文件存在 |
| 3 组各自 Stage 4 tensor | 3 个 merged_predictions_tensor.pt, shape (19412, 10, 4) | torch.load |
| 3 组各自 eval JSON | 3 个 task15_eval_{A,B,C}.json | 文件存在 |
| **三组对比表** | 1 个 task15_comparison.json 含 R@5/R@10/N@5/N@10/碰撞率 | 格式正确 |

---

## 关键判断标准（提前定好）

| 条件 | 结论 | 论文故事线 |
|------|------|------------|
| Group C R@10 > Group B + 0.005 且 > Group A | GSRQ gain-shape 融合有价值 | "提出新方法" |
| Group C ≈ Group B（差 < 0.003）| 纯方向已是最优 | "审计/诊断型"：验证 MMQ 是天花板 |
| Group A > Group B | magnitude 比 direction 更重要 | MMQ 不一定最优，需进一步探索 |
| Group B > Group A | direction 主导 | MMQ 是最优实践（验证论文） |
| Group C 碰撞率 < Group A/B - 0.05 | GSRQ 改善碰撞率 | 独立次要贡献 |

---

## 执行顺序（重要）

1. **先跑 Group B**（yaml 默认，0 新代码，~30 min）
2. **再跑 Group A**（只改 CLI 参数，~30 min）
3. **判断 A vs B**：如果 B > A（验证 MMQ），再投入 Group C 实现
4. **如果 B 不显著 > A**，可能该场景不适合"方向优先"，则 Group C 也大概率不优于 B；可考虑：
   - 直接出 A/B 对照表，把 Idea 3 定位成"诊断型"（已做）
   - 或继续跑 Group C 作为完整性（保守做法）

---

## 风险与回退

| 风险 | 回退方案 |
|------|----------|
| Group C 实现 bug | 退化为 Group A（不传 use_gain_tracking）跑，仍然能出 3 组对照 |
| Group C 训练时间过长 | 减少 max_steps=1500，复用 baseline 推理产物 |
| 评估 JSON 缺字段 | 用现有 task4_free_form_eval.py 跑，加 collision_rate 字段 |
| TIGER Stage 3 在新 SID 上训练失败 | 用 baseline TIGER ckpt 评估所有 3 组 SID（降级方案） |

---

## 与之前 task14/15/16 诊断的关系

| 之前结论 | Task 17 如何验证 |
|----------|------------------|
| Task 14: 衰减 CV ≈ 0.05 均匀 | Group B 残差全在 unit sphere 上，确实"形状"主导 |
| Task 15: 新层 100 步内稳定 | 三组都在 3000 step 训练，100 步远在收敛后 |
| Task 16: 1D gain 分类 acc ≈ top-class 占比 | **Task 17 进一步**：完整量化器中 gain 是否有价值？|

Task 17 是 task14/15/16 三个诊断的"下游验证"——如果统计上"gain 弱"，那么：
- 完整量化器里 gain 应该确实没用（Group A ≈ Group B）
- 即便精细融合（Group C），也救不回这个弱信号（Group C ≈ Group B）
- → 故事线：诊断 + 验证 = 审计/诊断型

如果 Task 17 反例（Group C > Group B 显著），则前面的诊断错了，需要重新审视 task16 的 1D 分类（也许是分类任务本身不够强，而不是 gain 没信息）。

---

## 输出示例（最终对比表 JSON 格式）

```json
{
  "task": "task15_gain_shape_fusion_vs_mmq",
  "data": "Toys (11924 items, 2048-dim embedding)",
  "groups": {
    "A_baseline": {
      "name": "Standard RKMeans (no normalize)",
      "stage21_ckpt": "logs/.../task15_a/checkpoints/checkpoint_000_003000.ckpt",
      "stage22_sid": "logs/.../task15_a_s2/pickle/merged_predictions_tensor.pt",
      "stage3_ckpt": "logs/.../task15_a_s3/checkpoints/best_tiger_*.ckpt",
      "stage4_tensor": "logs/.../task15_a_s4/pickle/merged_predictions_tensor.pt",
      "collision_rate": 0.15,
      "n_unique_sids": 10135,
      "n_total_items": 11924,
      "Recall@5": 0.034,
      "Recall@10": 0.052,
      "NDCG@5": 0.022,
      "NDCG@10": 0.028
    },
    "B_mmq": { ... same schema ... },
    "C_gsrq": { ... same schema ... }
  },
  "delta_C_minus_B_R10": 0.002,
  "verdict": "C ≈ B (delta < 0.003) → MMQ 纯方向已是最优；碰撞率独立观察..."
}
```

---

## 启动命令模板

### Group B (MMQ)

```bash
# Stage 2.1
cd /fs04/ar57/wenyu/GeneRec/GRID
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 codebook_width=256 \
    trainer.max_steps=3000 \
    trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task15_group_b_s21
```

### Group A (baseline)

```bash
# Stage 2.1
python -m src.train experiment=rkmeans_train_flat \
    model.normalize_residuals=false \
    ... (其他同上) \
    id=task15_group_a_s21
```

### Group C (GSRQ, 需先实现 yaml + mini_batch_kmeans.py 改动)

```bash
python -m src.train experiment=rkmeans_train_gsrq \
    ... (其他同上) \
    id=task15_group_c_s21
```

> 完整 4 阶段（Stage 2.1/2.2/3/4）+ eval 见 `task_artifacts/scripts/task15_*.sh`

---

## 复现判据

- ✅ Group B R@10 ≈ 论文 Toys RKMeans R@10 (0.0577) 复现
- ✅ Group A R@10 vs Group B R@10 差异 < 0.01（A/B 都是标准 RKMeans，差异仅在 normalize）
- ✅ Group C R@10 vs Group B R@10 差异 < 0.005（如果 GSRQ 无价值）
  - 或 > 0.005（如果 GSRQ 有价值，这本身就是论文主要贡献）

---

## 备注

- 三方对照中 Stage 3/4/eval 完全相同，**唯一变量是 Stage 2.1 的量化器**——确保 Stage 3 训练时用相同的 seed=42（yaml 已设）以排除随机波动
- TIGER 训练本身有早停 callback（`val/recall@10` 连续 10 次无提升即停），所以即使 max_steps=50000 也不会跑满
- 如果某组 Stage 3 早停后 R@10 极低（如 < 0.02），先检查该组的 SID tensor 是否合理（unique 数量是否够多）