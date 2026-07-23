# Task #68 — MCKG 原始（未修 norm）+ 192d RQ-VAE 验证维度独立性

> **任务目的**: 验证"维度提高到 192d 是否能独立解决 MCKG 在 RQ-VAE 上的坍缩问题"，填补 Task #67 与 64d 实证之间的关键认识论空白——现有数据中"高维"和"健康"两个变量绑在一起 (T5 2048d 健康)，缺乏"不健康但高维"的直接对照。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 已有数据点（构成"维度→坍缩"的间接曲线）

| 输入 | 维度 | 健康度 | L0 coverage | 是否坍缩 |
|------|------|--------|-------------|----------|
| MCKG 原始 (Task #100) | 64d | ❌ 不健康 | **0.11** | ❌ **严重坍缩** |
| S4 AE (Task #54/58/59 baseline) | 64d | ✅ 健康 | (历史) | ❌ **64d 也坍缩** |
| MCKG 修复 (Task #67 当前) | 192d | ✅ 健康 | (正在跑) | ⚠️ 仍未确认 |
| T5 flan-xl | 2048d | ✅ 健康 | (未测) | ✅ 不坍缩 (假设) |

3 个点连起来看似"维度越高坍缩越轻"，但**关键漏洞**：
- 64d 与 192d 之间数据点缺失
- 高维数据点 (T5) 恰好也健康，**维度高 ≠ 健康**这个变量无法分离

### 1.2 本任务的核心假设 R1

**R1**: MCKG 拼接提高 192d (不论是否修 norm) 都应该有显著高于 64d 的 L0 coverage，**因为 encoder 有了充足通道处理异常值**。

**对照假设 R2 (若 R1 否证)**：norm 病才是主因，提高维度本身无法解决，需要 norm 修复独立作用。

### 1.3 本任务的关键格子

| | 64d | 192d |
|---|---|---|
| **MCKG 原始 (未修 norm)** | ✅ Task #100 (cov=0.11) | **🔴 本任务 #68 — 最关键的"未测格子"** |
| **MCKG + norm 修复** | ❌ 未测 | ✅ Task #67 |

完成本任务即填上 2×2 矩阵中"MCKG 原始 + 192d"格子。基于 #68 的决策方向：
- 若 #68 coverage ≥ 80% → R1 ✅ GO，R2 否证 → 后续跑 "MCKG 修复 + 64d" 进一步定位 norm 修复独立性
- 若 #68 coverage 50-80% → R1 部分成立 → norm 修复仍是必要维度
- 若 #68 coverage < 50% → R1 否证，norm 修复不能"被维度替代"

---

## 2. 实验设计

### 2.1 变量

| 维度 | Task #67 (已设计) | **Task #68 (本任务)** |
|------|-------------------|------------------------|
| **输入 norm 修复** | exp_map_pre_clip + cap=5.0 各自修复 | **不修复，直接 raw concat** |
| **拼接方式** | 3 段拼接 192d | **相同: 3 段拼接 192d** |
| **Revival hook** | 启用 | **禁用**（需 override Hydra yaml） |
| **RQ-VAE 配置** | num_hierarchies=3, codebook_width=256, latent_dim=128 | **相同** |
| **max_steps** | 5000 | **相同** |
| **seed** | 42 | **相同** |
| **embedding_dim** | 192 | **相同** |

**唯一独立变量**: 是否对每个 subspace 做 norm 修复。
**次要差异**: 关闭 Revival hook，确保维度独立性的纯净对比。

### 2.2 Stage 0: 输入准备（不修 norm）

```python
import torch
entity = torch.load("products/task99_mckg_rebuild/entity_embedding.pt")
# entity shape: (3, 11924, 64) = (n_subspaces, N, dim_per_subspace)
raw_concat_192d = torch.cat([entity[0], entity[1], entity[2]], dim=-1)
# shape: (11924, 192), values 直来自 raw MCKG, 不修复
torch.save(raw_concat_192d, "products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt")
```

**输出**:
- `products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt` (11924, 192)
- 不保证 norm 范围，可能有 1000× 异常值（来自 euclid/hyperbolic subspace）

### 2.3 Stage 1: RQ-VAE 训练（不加 Revival）

**启动命令**:
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# 关闭 Revival hook (避免与"维度独立性"实验设计混淆)
python -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=/home/wlia0047/ar57/wenyu/GeneRec/products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt \
    embedding_dim=192 num_hierarchies=3 codebook_width=256 \
    trainer.max_steps=5000 \
    trainer.val_check_interval=100000000 \
    +callbacks.revive_dead_codes=null \
    task_name=task68_mckg_raw_192d \
    2>&1 | tee logs/task68_mckg_raw_192d.log
```

**关于 Revival 移除**: Hydra 标准做法是用 `~` prefix 删除 config key（在 run command 后追加 `hydra.job_logging=disabled` + 手写一个没有 revive_dead_codes 的 yaml）。简化做法：直接在 config 里注释掉 revival_dead_codes 这一段，恢复后再启用。

**最稳方案**: 在 `configs/experiment/rqvae_train_flat.yaml` 临时注释掉 `revive_dead_codes` 的 callback 配置，跑完 #68 再恢复。或者用一个 deepcopy 后的 v68 版 yaml。

### 2.4 Stage 2: 检查指标（不做 Stage 2.2/3/4 端到端）

本任务是**纯验证维度独立性**的对照实验，**不直接进入 Stage 2.2 推断**：本任务的目的是判断 L0 coverage，结论成 GO 后再叠加 norm 修复做 192d 端到端。

```
[skip Stage 2.2/3/4 unless coverage supports]
```

### 2.5 监控指标

- **`train/layer_0/frac_layer_coverages_step`**: 每 50 step log 一次的 batch-level coverage
- **`train/layer_0/id_entropy_step`**: id 分布熵 (max=ln(256)=5.55)
- **`train/reconstruction_loss_step`**: 重建损失 (不应 NaN/Inf)
- **`train/frac_unique_ids_step`**: 唯一 ID 比例

---

## 3. 决策触发（vs Task #100 64d cov=0.11）

### 3.1 主决策表（核心是 coverage 对比）

| L0 batch-level coverage | 决策 | 假设验证 |
|-------------------------|------|----------|
| **≥ 0.80** | ✅ **GO**: R1 完全成立 | 维度独立解决坍缩 → 后续跑 "MCKG 修复 + 64d" 确认 norm 独立作用 |
| **0.50 ~ 0.80** | ⚠️ **PARTIAL**: R1 部分成立 | 维度有帮助但不充分，需 norm 修复协同 |
| **0.20 ~ 0.50** | ⚠️ **R1 部分否证** | norm 修复仍然必需 |
| **< 0.20** | ❌ **否证 R1**: norm 病无法被维度替代 | 必须 norm 修复才能跑通 192d |
| **NaN / 训练崩溃** | ❌ **否证 R1**: 不修 norm 直接 RQ 会训练失败 | 必须 norm 修复才能跑通 |

### 3.2 辅助指标（次要）

| 指标 | 含义 | 阈值 |
|------|------|------|
| `recon_loss_epoch` | 训练稳定性 | < 0.1 (稳定); 0.1-0.5 (可接受); > 0.5 (退化) |
| `id_entropy L0` | centroid 利用 | ≥ 4.5 健康 |
| `frac_unique_ids_step` | SID 唯一性 | ≥ 0.95 健康 |
| 各层 centroids_norm | centroid 健康 | 1.0-100 区间健康 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 0 数据准备 (~ 1 min 写 entity_embedding_concat192d_raw.pt) | ~1 min |
| Stage 1 RQ-VAE 训练 (max_steps=5000, 1.85it/s 估计) | **~45 min** |
| **总计** | **~1 h GPU** |

> 本任务只跑 Stage 0 + Stage 1 (不动 Stage 2.2/3/4)，比 Task #67 端到端预算低很多。

---

## 5. 风险与缓解

### 风险 1: 不健康 192d 训练时 loss 出现 NaN / Inf

**根因**: MCKG euclid/hyperbolic subspace 有 100-1000× 异常值 (Task #99 实测)。
**缓解**:
- Hydra 默认 `bf16-mixed` 精度下自动 clamp
- 若爆 NaN，fallback 关闭 Revival hook 改用 float32 重跑 (但在 CLAUDE.md R2 下禁用 fallback，应当在 description 阶段就改环境)
- 建议 Stage 0 加一道 sanity check: 验证 `raw_concat_192d.abs().max()` 不超过 1e3

### 风险 2: Hydra Revival hook 关闭不彻底

**根因**: `+callbacks.revive_dead_codes=null` 不一定让 Hydra 完全跳过该 callback。
**缓解**:
- 用 `~callbacks.revive_dead_codes` 删除该 config key (Hydra delete mark)
- 或者直接修改 yaml，临时注释掉 revive_dead_codes block 跑完 #68 再恢复（最稳）
- 建议方案：在 `configs/experiment/` 下创建 `rqvae_train_flat_no_revive.yaml` 作为 #68 的启动 yaml

### 风险 3: GPU 0 被 #67 占满 (PID 3232809 在跑)

**当前状态**: Task #67 已经 launched 第二次重跑（PID 3232809 在 GPU 0，max_steps=5000，~30-45 min ETA）
**缓解**:
- 等 #67 跑完（约 ~30 min）再启动 #68
- 或并行启动 #68 在 GPU 1/2/3 任一空闲卡上（多卡隔离 RQ-VAE 训练）

### 风险 4: 实验数据落盘路径混乱

**缓解**:
- 训练产物明确落到 `logs/task68_mckg_raw_192d/runs/<date>/<time>/`
- inference 产物 (`merged_predictions_tensor.pt`, `cluster_ids.pt`) 暂不产生 — 本任务不做 Stage 2.2
- Stage 0 数据落到 `products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt`

---

## 6. 完成度跟踪

- [ ] Stage 0 数据准备 (~1 min)：从 `products/task99_mckg_rebuild/entity_embedding.pt` raw → `products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt`
- [ ] 创建 `rqvae_train_flat_no_revive.yaml` config（移除 Revival hook）或修改现有 yaml
- [ ] Stage 1 RQ-VAE 训练 (max_steps=5000, ~45 min)
- [ ] 提取 step 500/1000/2000/3000/5000 的 layer coverage / id_entropy / recon_loss
- [ ] 写 `verdicts/task68_result.md`（最终 verdict 必须含 `result:` 行）
- [ ] **如果** coverage ≥ 80% → 创建后续 task #69 跑 "MCKG 修复 + 64d" 进一步定位 norm 独立作用

---

**设计依据**: 该实验对应 2×2 矩阵中的"高维 + 不健康"格，是分离"维度"与"健康度"两个变量最直接的方法。
