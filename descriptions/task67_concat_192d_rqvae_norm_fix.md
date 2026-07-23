# Task #67 — 拼接 192d RQ-VAE + 三段独立 norm 修复 + Dead Code Revival

> **任务目的**: 验证"3 段各自 64d 先 norm 修复 → cat 成 192d → 保守 RQ-VAE (192→128 latent, K=256, 3-layer) + Dead Code Revival"是否能把 Task #44 E4 baseline (1.75× recon_loss) 压到 ≤ 1.1×，同时 codebook coverage 全部 ≥ 0.95，**真正跑通 RQ-VAE 而不依赖 SimpleKMeans 兜底**。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 4 个前置结论：

| 结论来源 | 关键数字 | 含义 |
|---------|---------|------|
| **Task #44 E4** (concat-192d L2) | recon_loss = **1.75×** baseline | 所有独立量化方案里最接近 baseline，但 1.75× 仍退化 |
| **Task #45 Joint-Flat-3seg** | raw 1.390 (5.05×), +log1p **0.389 (1.41×)** | 训练耦合不带来额外增益，norm 修复是真正修复 |
| **Task #46 L1 G1 PASS (log1p)** | SCR 4.22× → **0.31×** | 后处理 log1p 端到端有效 |
| **Task #54 ⭐** | cov0 = 1.0 (SimpleKMeans) vs 0.11 (neural RQ-VAE) | 神经 RQ-VAE 架构在低维 64d 输入上**瓶颈压缩坍缩** |
| **Task #99 MCKG 重训** | sphere 1.9× ✓, euclid 仍 486× ❌, hyperbolic 仍 960× ❌ | 三段 norm 病态程度不同，需分别修复 |

**核心假设 H1**: 把 3 段独立修复 + 192d 拼接 + 保守压缩比 + LayerNorm + Dead Code Revival 这套组合搬到 RQ-VAE 上，能跑通且重建质量优于 E4 (1.75×)。

**核心假设 H2**: Dead Code Revival 机制能解决 Task #54 揭示的"cov0 坍缩到 0.11"问题（layer-0 codebook 256 桶至少用 244）。

**核心假设 H3**: 输入维度从 64d (Task #53/54) 提到 192d 后，encoder 192→128 的 1.5× 压缩比不会触发瓶颈坍缩（Task #54 根因是 64→64 同维 encoder 在 64d 输入上信息不足）。

---

## 2. 实验设计

### 2.1 Stage 0: 输入准备

```python
# 数据源：Task #99 重建后 MCKG (products/task99_mckg_rebuild/entity_embedding.pt)
# subspace shape: (3, 11924, 64)
subspace_sphere_fixed = repair_norm(subspace_sphere, method='exp_map_pre_clip', cap=5.0)
subspace_euclid_fixed = repair_norm(subspace_euclid, method='exp_map_pre_clip', cap=5.0)
subspace_hyperbolic_fixed = repair_norm(subspace_hyperbolic, method='exp_map_pre_clip', cap=5.0)
input_192d = torch.cat([subspace_sphere_fixed, subspace_euclid_fixed, subspace_hyperbolic_fixed], dim=-1)
# shape: (11924, 192)
```

**关键顺序**: 修复要在拼接之前做（Task #99 教训：三段 norm 病态程度不同，分别处理更精确）。

### 2.2 Stage 1: 编码器 (保守压缩比 + LayerNorm)

```python
class Encoder(nn.Module):
    def __init__(self, input_dim=192, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(192, 160),
            nn.LayerNorm(160),    # 稳定性保险
            nn.ReLU(),
            nn.Linear(160, 128),
        )
```

**压缩比 1.5×**（vs Task #53 64→64 同维 / Task #54 64→64 同维），给 decoder 足够重建空间。

### 2.3 Stage 2: 标准欧氏 RQ (3 层 K=256, 不带曲率)

```python
class ResidualQuantizer(nn.Module):
    n_layers=3, K=256, latent_dim=128
    # 直接欧氏 argmin + 残差减法（无 log/exp map）
    # 每个 epoch 末尾 revive_dead_codes(threshold_ratio=0.01)
```

**为什么不用 PM-RQ**: Task #44 V3 已证伪（κ 极端退化，fusion trivial）。

### 2.4 Stage 3: 解码器 + Loss

```python
Decoder: Linear(128→160) → ReLU → Linear(160→192)
Loss = MSE_recon + 0.25 * commit_loss + 0.1 * diversity_loss
```

**diversity_loss = -z.var(dim=0).mean()** 防止 latent 坍缩（保险机制）。

### 2.5 Stage 4: 训练循环

- 每个 epoch 末尾: revive_dead_codes(batch_residuals, threshold_ratio=0.01)
- 每个 epoch 输出 layer-0/1/2 coverage，目标 ≥ 0.95

**变量**: 输入拼接方式（Task #44 E4 直接 L2 归一化 vs 本任务 分段 norm 修复 + 拼接）
**保持不变**:
- 数据源: Task #99 重建后 MCKG (subspace shape (3, 11924, 64))
- num_hierarchies=3, codebook_width=256, K=256
- seed=42

**启动命令**:
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

python -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=/home/wlia0047/ar57/wenyu/GeneRec/products/task99_mckg_rebuild/entity_embedding_concat192d.pt \
    embedding_dim=192 num_hierarchies=3 codebook_width=256 \
    task_name=task67_concat192d_norm_fix
```

> ⚠️ 启动前需先写脚本 `scripts/task67_concat192d_rqvae.py`：
> 1) 加载 `products/task99_mckg_rebuild/entity_embedding.pt`
> 2) 拆出 3 个 subspace (3, 11924, 64)
> 3) 分别 norm 修复 (exp_map_pre_clip + cap=5.0)
> 4) cat → (11924, 192) 保存到 `products/task99_mckg_rebuild/entity_embedding_concat192d.pt`
> 5) 启动 src.train

---

## 3. 决策触发（vs Task #44 E4 baseline recon_loss = 1.75× = 0.4813）

| recon_loss 倍数 | 状态 | 决策 |
|----------------|------|------|
| **≤ 1.1× (0.30)** | ✅ 完全跑通 | H1+H2+H3 全部确认 → 进入 Stage 2.2 SID 推断 + Stage 3 TIGER |
| 1.1× ~ 1.5× (0.30 ~ 0.41) | ⚠️ 部分跑通 | norm 修复生效但 RQ 仍有压缩损失 → 调研 compression ratio |
| **1.5× ~ 1.75× (0.41 ~ 0.481)** | ⚠️ 与 E4 同水平 | 拼接方案无实质改进 → 关闭主线，回到 E4 |
| **> 1.75× (0.481)** | ❌ 比 E4 差 | 修复反而恶化 → 检查 norm 修复是否过激 / encoder 压缩比 |

**Codebook coverage 辅助判据**:
- Layer-0 coverage ≥ 0.95 → H2 确认（Dead Code Revival 起效）
- Layer-0 coverage < 0.5 → H2 否证 → 检查 encoder 是否仍坍缩

**联合判据**: recon_loss ≤ 1.1× AND Layer-0 coverage ≥ 0.95 → 完全 GO

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 0 数据预处理（写 entity_embedding_concat192d.pt） | ~5 min |
| Stage 2.1 RQ-VAE 训练（5000 steps） | ~30-60 min |
| Stage 2.2 SID 推断 + Stage 3 TIGER 训练（若 GO） | ~2-3 h |
| Stage 4 推断 + R@5/R@10 评估 | ~10 min |
| **总预算（仅 Stage 0 + Stage 2.1 验证决策阈值）** | **~1 h** |
| 总预算（若 GO 含 Stage 3+4） | ~4 h |

---

## 5. 风险与缓解

**风险 1**: Task #99 重建后 MCKG 的 sub_e / sub_h 仍分别有 486× / 960× 的 norm 长尾（仅 sphere 修到 1.9×），exp_map_pre_clip + cap=5.0 可能仍不够 → 缓解：先用 `print(subspace_X_fixed.norm(dim=-1).max())` 检查每段修复后 max norm ≤ 5.0；如 cap=5.0 不够，考虑 cap=3.0 + 二次 log1p。

**风险 2**: Dead Code Revival 触发时机不对（每 epoch 一次 vs 每 N step 一次）→ 缓解：先按每 epoch 一次跑，看 coverage 收敛曲线，必要时改每 500 step 一次。

**风险 3**: Task #54 揭示的 RQ-VAE 架构缺陷（bottleneck 64→64 坍缩）在 192→128 上仍存在 → 缓解：监控 diversity_loss，若 -z.var(dim=0).mean() 持续小于 0.01 → latent 已坍缩 → 改 latent_dim=160 或加 skip connection。

**风险 4**: 与 Task #66 P3（Sinkhorn cascade TIGER，PID 1029066）共享 GPU 资源冲突 → 缓解：本任务用 GPU 1/2/3（Task #66 P3 占 GPU 0/1/2/3 全 4 卡 DDP，等其自然结束或 kill 后再启动；如必要可申请新卡）。

---

## 6. 完成度跟踪

- [ ] Stage 0: 写 `scripts/task67_concat192d_rqvae.py`，生成 `products/task99_mckg_rebuild/entity_embedding_concat192d.pt`
- [ ] Stage 0: 验证每段 norm 修复后 max ≤ 5.0
- [ ] Stage 2.1: 启动 RQ-VAE 训练（5000 steps）
- [ ] Stage 2.1: 监控 layer-0/1/2 coverage ≥ 0.95
- [ ] Stage 2.1: 收敛判据 recon_loss 倍数落入 §3 决策表
- [ ] 若 GO：Stage 2.2 SID 推断 + Stage 3 TIGER 训练
- [ ] 若 GO：Stage 4 推断 + R@5/R@10 评估
- [ ] 写 `verdicts/task67_result.md`（含 result: 行）
- [ ] 更新 loop.md §16：归档 + 进入下一个任务

---

**任务目标 ROI**: 若 GO（recon_loss ≤ 1.1×），证明 Task #54 揭示的 RQ-VAE 架构缺陷可通过 "192d 输入 + 保守压缩 + LayerNorm + Dead Code Revival" 修复，**直接打开"MCKG 嵌入 → RQ-VAE → TIGER"完整闭环**，是论文"训练目标 × 可量化性"研究的关键缺口补充。

当前任务已完成，请做下一个任务的指示。
