# Task #67 v6 — norm 修复但禁用 Revival（隔离 Revival 独立贡献）

> **任务目的**: 隔离 Revival hook 的独立贡献 — v5 (norm fix + Revival) L0 cov=0.374 劣于 raw 192d (Task #68 = 0.651), 需 v6 验证是 Revival 单独反作用, 还是 norm 修复本身也反作用.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 已有数据点（2×2 矩阵）

| | 64d | 192d |
|---|---|---|
| **MCKG 原始（未修 norm）** | ❌ Task #100 (cov=0.11) | ✅ Task #68 (cov=0.651) |
| **MCKG + norm 修复** | ❌ 未测 | ✅ Task #67 v5 (cov=0.374, norm fix + Revival) |

3 个格子已有数据. 缺失的最后一格是 "MCKG 修复 + 64d", 但 ROI 较低（已知 64d 坍缩严重）.

### 1.2 v5 反直觉发现

| 实验 | 输入 | Revival | L0 cov epoch | L0 ent epoch |
|------|------|---------|--------------|--------------|
| Task #67 v5 | norm fix 192d | 启用 | **0.3745** | 4.126 |
| Task #68 | raw 192d | 禁用 | **0.6510** | 4.570 |

**v5 比 raw 差 42%** — 反直觉. 两种可能解释:
- (A) **Revival hook 是负贡献** — 在 192d 上过度干预, 1620 centroids 被强制替换
- (B) **norm 修复本身是负贡献** — cap=5.0 把 hyperbolic 异常值压平, 丢信息
- (C) **两者都负贡献** — 协同失效

### 1.3 v6 核心目标

**隔离 A vs B**: 用 norm 修复输入但禁用 Revival, 看 L0 cov 能否回升到 ≥ 0.651.

- 若 v6 L0 cov ≥ 0.651 → **A 成立 (Revival 是负贡献)** → 永久关闭 Revival, 用 raw 192d
- 若 v6 L0 cov ∈ [0.40, 0.65) → **A + B 都成立** → 既不用 Revival 也不用 norm fix
- 若 v6 L0 cov < 0.40 → v5 的 0.374 大部分来自 norm fix 信息损失, raw 192d 仍是最佳

---

## 2. 实验设计

### 2.1 变量

| 维度 | Task #67 v5 (norm fix + Revival) | **Task #67 v6 (本任务)** |
|------|--------------------------------|------------------------|
| **输入** | norm 修复 192d | **相同: norm 修复 192d** |
| **Revival hook** | 启用 | **禁用** |
| **embedding_dim** | 192 | 192 |
| **num_hierarchies** | 3 | 3 |
| **codebook_width** | 256 | 256 |
| **max_steps** | 5000 | 5000 |
| **seed** | 42 | 42 |

**唯一独立变量 vs v5**: 关闭 Revival hook
**唯一独立变量 vs Task #68**: 启用 norm 修复

### 2.2 Stage 0: 输入复用

直接复用 Task #99 的 norm 修复 192d embedding:
```
products/task99_mckg_rebuild/entity_embedding_concat192d.pt  (11924, 192)
```

### 2.3 Stage 1: RQ-VAE 训练（禁用 Revival）

**启动命令**:
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

python -m src.train experiment=rqvae_train_flat_no_revive \
    data_dir=data/amazon_data/toys \
    embedding_path=/home/wlia0047/ar57/wenyu/GeneRec/products/task99_mckg_rebuild/entity_embedding_concat192d.pt \
    embedding_dim=192 num_hierarchies=3 codebook_width=256 \
    trainer.max_steps=5000 \
    task_name=task67_v6_norm_fix_no_revival \
    2>&1 | tee logs/task67_v6_norm_fix_no_revival.log
```

直接复用 `configs/experiment/rqvae_train_flat_no_revive.yaml`（Task #68 已创建）.

### 2.4 监控指标

- `train/layer_0/frac_layer_coverages_step` (batch-level coverage)
- `train/layer_0/id_entropy_step`
- `train/frac_unique_ids_step`
- `train/reconstruction_loss_step`

---

## 3. 决策触发（vs v5=0.374 / Task #68=0.651）

| L0 cumulative coverage | 决策 | 假设验证 |
|------------------------|------|----------|
| **≥ 0.651** | ✅ **A 成立**: Revival 单独反作用 | 永久关闭 Revival, 用 raw 192d + 不 Revival |
| **[0.40, 0.65)** | ⚠️ **A + B 都成立**: Revival 反作用 + norm fix 也反作用 | 用 raw 192d + 不 Revival (Task #68 配置) |
| **[0.30, 0.40)** | ⚠️ **B 部分成立**: Revival 没作用, norm fix 反作用 | 用 raw 192d |
| **< 0.30** | ❌ **v5 退化主要来自其他因素** | 需要进一步诊断 (查 norm fix 是不是有问题) |
| **NaN / 崩溃** | ❌ norm 修复 192d 不稳定 | 不能用 norm fix |

### 辅助决策（vs Task #68 raw=0.651）

- v6 cov ≥ 0.651 → norm fix 信息损失比 Revival 负贡献小, 启用 norm fix + 不 Revival 更优
- v6 cov < 0.651 → raw 192d 仍是最佳

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 RQ-VAE 训练 (max_steps=5000, ~1.85it/s) | **~45 min** |
| 写 verdict | ~5 min |
| **总计** | **~50 min GPU** |

> 本任务只跑 Stage 1, 不做 Stage 2.2/3/4 端到端, 预算很低.

---

## 5. 风险与缓解

### 风险 1: GPU 占用冲突

**当前状态**: 4 张 A40 全部空闲 (2026-07-20 16:55 检查).
**缓解**: 直接用 GPU 0 启动, ~45 min 内不冲突.

### 风险 2: no_revive yaml 可能与 norm 修复输入不兼容

**根因**: rqvae_train_flat_no_revive.yaml 复用了 rqvae_train_flat.yaml 的所有配置, 只删了 Revival callback. 其它 (BatchNorm + NormalizeLayer) 都相同, 应该兼容.
**缓解**: 启动前先 `python3 -m py_compile` 检查 yaml 解析; 启动后看 step 1-100 是否 loss 健康.

### 风险 3: 与 Task #68 (raw 192d) Stage 1 的混淆

**缓解**: 用不同的 task_name `task67_v6_norm_fix_no_revival` 避免 logs/ 目录冲突.

---

## 6. 完成度跟踪

- [ ] 写 descriptions/task67_v6_norm_fix_no_revival.md (本文件)
- [ ] 启动 Stage 1 RQ-VAE 训练 (GPU 0, ~45 min)
- [ ] 监控 step 500/1000/2500/5000 的 L0 cov/id_entropy/loss
- [ ] 写 `verdicts/task67_v6_result.md` (含 result: 行)
- [ ] 更新 loop.md §16 (从活跃任务列表删除, 归档到历史已清理)
- [ ] 更新 memory (写入 Revival vs norm fix 隔离结论)

---

**设计依据**: 该实验是 Task #67 v5 (norm fix + Revival) 失败后的诊断性 v6, 通过关闭 Revival hook 隔离其独立贡献, 是 2×2 矩阵分析的关键 pivot.
