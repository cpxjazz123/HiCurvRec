# Task #41 — PM-RQ geometry-aware RQ-VAE (3 段原始子空间作输入)

> **任务目的**: 用 3 段**原始子空间**(各 64d, 各自留其流形)作 PM-RQ 量化器的输入,验证"混合曲率量化"的设计价值。这是 PM-RQ 本体的真实验,不是 fused_64d 喂 PM-RQ。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (cuda:1, ~40 min 训练 + 5 min 推断 + 2.5 h Stage 3)

---

## 1. 背景

**用户 2026-07-20 关键修正**: "PM-RQ 真正需要的输入,应该是压平之前的三个子空间,不是 fused_64d。"

理由: PM-RQ 的核心机制 = **量化时分别处理三种曲率**。这要求量化器输入是"压平之前"的 3 段原始表示,各 64d,各在自身流形上 (sphere / euclid / hyperbolic)。

- fused_64d 是"压平之后"的产物 (tangent map + 平均),几何结构已被 flatten,无法在 PM-RQ 中体现
- Task #40 (对照组 fused → 标准 RQ-VAE) 仍然用 fused_64d — 它测的是"仅换信息源"的价值,不需要几何保持

承接 Task #36 (3 子空间均 STRONG signal) + Task #43 (norm clip 必须) + Task #22 (PM-RQ × TIGER 早期实验 R1 否证, 但那是 fused 输入)。

**关键差异**: 早期 Task #23/#24 用 fused_64d 喂 PM-RQ, R@5 仅 0.0014-0.0047。本任务用 **3 段原始** 喂 PM-RQ,是新的设计,预期结果可能不同。

## 2. 实验设计

**变量**: PM-RQ 量化器的输入 (fused_64d → 3 段原始子空间) + 距离度量 (L2 → 乘积流形距离)
**保持不变**:
- 码本: 3 段独立码本 (C_s ∈ R^64 sphere-constrained, C_e ∈ R^64, C_h ∈ R^64 hyperbolic-constrained), 各 K=256
- 可学习 κ ∈ [-8, +8], 初始为 κ_init=[+5.05, -0.08, -5.04] (MCKG 训练后的值)
- 可学习权重 w ∈ R^3, softmax 归一化,初始均匀 1/3
- Stage 3+4 TIGER 配置 (与 Task #87 一致)

**调整项** (因 3 段输入而非 fused):
- 输入维度: 64 (单段) → 64×3=192 (拼接 3 段作为编码器输入)
- encoder 第一层 dim: 64 → `[192,128,64,32]` (放大以容纳 3 段信息)
- decoder mirror: `[32,64,128,192]`
- 量化器: 3 段独立 quantize, 然后 concat 输出 (B, 192)
- 重建目标: 重建完整的 3 段拼接 (B, 192) 与原输入对比

**norm clip (前置, Task #43 结论)**:
```python
# 每个 subspace 单独 clip norm 到 p99
clip_vals = {'subspace_0': 0.5668, 'subspace_1': 4.7902, 'subspace_2': 10.2035}
for k, clip in clip_vals.items():
    norms = torch.norm(sub[k], dim=1, keepdim=True)
    sub[k] = sub[k] / norms.clip(min=1e-6) * norms.clip(max=clip)
# sphere 还应归一化到 1 (Poincaré 标准)
```

**启动命令**:
```bash
# Stage 2.1: PM-RQ 训练 (3000 steps)
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=rqvae_train_pm \
  data_dir=data/amazon_data/toys \
  embedding_path=products/task41_pm_rq/subspace_0.pt:products/task41_pm_rq/subspace_1.pt:products/task41_pm_rq/subspace_2.pt \
  embedding_dim=64 num_hierarchies=3 codebook_width=256 \
  model.input_mode=three_subspaces \
  model.encoder.dim_per_layer=[192,128,64,32] \
  model.decoder.dim_per_layer=[32,64,128,192] \
  model.codebook_type=product_manifold \
  model.pm_learnable_curvature=true \
  model.pm_learnable_weights=true \
  model.beta=0.25 \
  trainer.max_steps=3000 \
  data.batch_size=512 \
  optim.lr=0.001 \
  task_name=task41_s2_train

# Stage 2.2: SID 推断
CUDA_VISIBLE_DEVICES=1 python3 -m src.inference experiment=rkmeans_inference_flat \
  data_dir=data/amazon_data/toys \
  semantic_id_path=... task_name=task41_s2_infer

# Stage 3: TIGER 训练
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
  data_dir=data/amazon_data/toys \
  semantic_id_path=products/task41_pm_rq/stage2_infer/cluster_ids.pt \
  sequence_length=120 num_hierarchies=4 seed=42 \
  task_name=task41_s3_train

# Stage 4: 推断 + 评估
CUDA_VISIBLE_DEVICES=1 python3 -m src.inference experiment=tiger_inference_flat \
  data_dir=data/amazon_data/toys \
  semantic_id_path=products/task41_pm_rq/stage2_infer/cluster_ids.pt \
  task_name=task41_s4_infer
```

## 3. 决策触发 (vs Task #40 fused_64d 端到端 R@5)

| Task #41 端到端 R@5 | 解读 | 后续 |
|---|---|---|
| Task #41 R@5 ≥ Task #40 × 1.10 (+10%) | 3 段输入 + geometry-aware 设计显著优于 fused+标准 | ✅ 写 ablation + 论文 Section 5.3 |
| Task #40 × 0.95 ≤ Task #41 < Task #40 × 1.10 (±10%) | 3 段 vs fused 输入等价; PM-RQ 增益有限 | ⚠️ 改 ablation (sphere/euclid/hyperbolic 单子空间) |
| Task #41 R@5 < Task #40 × 0.95 (-10%) | 3 段输入伤害性能 | ❌ 退回 Task #22,标注 "PM-RQ 不优于标准 RQ-VAE" |

## 4. 预算

| 阶段 | 估算 | GPU |
|---|---|---|
| 写 PMCodebook + 3-段输入处理代码 + 单测 | ~45 min (代码) | — |
| Stage 2.1 PM-RQ 训练 (3000 steps, 输入 192d 更大) | ~50 min | cuda:1 |
| Stage 2.2 SID 推断 | ~5 min | cuda:1 |
| Stage 3 TIGER 训练 | ~2.5 h | cuda:1 |
| Stage 4 推断 + 评估 | ~5 min | cuda:1 |
| **总计** | **~3.7 h** | cuda:1 单卡 |

## 5. 风险与缓解

**风险 1**: 可学习 κ 在训练中发散到极端值 → 缓解: 显式 clamp κ ∈ [-8, +8], log 中监控 κ 轨迹
**风险 2**: PM-RQ loss 数值不稳定 (3 段距离量纲不同) → 缓解: 用 squared distance + softmax 权重,确保 loss 量纲一致
**风险 3**: 输入从 64d → 192d, RQ-VAE 编码器压力大 → 缓解: encoder hidden 同步放大 [192,128,64,32], 训练步数从 3000 → 4000
**风险 4**: 3 段码本对齐难 (3 段码字如何 "concat" 为一个 token?) → 缓解: 编码方式 (idx_s, idx_e, idx_h) → flat (idx_s * K^2 + idx_e * K + idx_h),与 RQ-VAE hierarchical 一致
**风险 5**: Task #23/#24 PM-RQ × TIGER 已否证 R1 → 缓解: 本任务的输入不同 (3 段原始,非 fused),属于新实验

## 6. 完成度跟踪

- [ ] 写 `src/components/pm_codebook.py` (PMCodebook + 可学习 κ + 可学习权重)
- [ ] 写 `configs/experiment/rqvae_train_pm.yaml` (input_mode=three_subspaces)
- [ ] 单测 PMCodebook (3 段独立 quantize + concat)
- [ ] 准备 3 段输入 + norm clip
- [ ] Stage 2.1 PM-RQ 训练
- [ ] Stage 2.2 SID 推断
- [ ] Stage 3 TIGER 训练 (early-stop)
- [ ] Stage 4 推断 + 评估
- [ ] 对比 Task #40, 写 ablation verdict

---

**Why 这次输入不同 = 这次结果可能不同**:
- 早期 Task #23/#24: fused_64d → PM-RQ (但 fused 已被 flatten,几何信息丢失)
- 本 Task #41: 3 段原始 → PM-RQ (每段保留曲率,可被 geodesic 距离正确处理)
- 这是 PM-RQ **真正的设计意图**,新实验预期可能优于早期 R1 否证链
