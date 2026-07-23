# Task #22 — Product Manifold RQ (PM-RQ): 球面×欧氏×双曲乘积流形量化

> **任务目的**: 验证 Product Manifold RQ (PM-RQ) 能否在深层信息集中问题上取得改进。若成功 → 投稿 CIKM/RecSys；若失败 → 进一步坐实"多几何 RQ-VAE 整体收益有限"的终局性结论。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (Phase 0)

---

## 1. 背景

承接 Task #85 三几何独立 RQ-VAE 实验结论：

- Task #85 m=0 球面 (sphere_fix) Recall@5 = 0.01932 (57% of baseline)
- Task #85 m=1 准欧氏 (euclid_full) Recall@5 = 0.02838 (83% of baseline)
- Task #85 m=2 双曲 (hyperbolic_full) trivial bias (失败)
- Task #85 Phase 4a 终局 verdict v2: `verdicts/task85_result.md`

**本任务假设**:
- **R1**: 单 κ 流形独立优化存在"几何信息浪费"——同一 embedding 既有球面结构也有欧氏结构，但当前实现只允许选一种
- **R2**: 乘积流形 (S × E × H) 用乘积度量 d² = d_s² + d_e² + d_h² 能在单层量化时同时保留三种几何信号
- **R3**: 三分量若 Kendall τ 接近 1 → 学的是冗余信息 → 重新设计；若都 < 0.7 → 三分量学不同信号 → 验证通过

**任务链**:
- 上游：Task #85 三几何独立 RQ-VAE（已完成）+ Task #15 baseline Toys（已完成）
- 下游：若成功 → Task #23 PM-RQ 三层 cascade + 端到端 Recall 评测
- 平行：Task #87 TIGER-aligned baseline（独立链路，不与本任务冲突）

---

## 2. 实验设计

**变量**: 几何结构 (单流形 → 乘积流形)
**保持不变**:
- 数据集：Amazon Toys (11924 items)
- Stage 1 embedding：sentence-t5-base 768d（Task #87 已生成，可复用）
- Stage 4 评估协议：Toys Recall@5/10, NDCG@5/10
- 码本大小 K=256（与 Task #85 一致）
- 种子：seed=42

**Phased 实验计划**:

### Phase 0 — 几何诊断 (1 周)

**目标**: 验证 embedding 确实需要混合曲率

```python
# D0 任务
# 1. δ-hyperbolicity 计算
#    工具: geoopt 或 hyperbolic_diagnostic
#    输出: δ_4-tree metric
# 2. sectional curvature 分布
# 3. UMAP 投影到 Poincaré ball / sphere / Euclidean
```

**D0 决策**:
- ✓ δ > 0.3 + 三方向非平凡几何特征 → proceed Phase 1
- ✗ embedding 本质平坦 → 停止

### Phase 1 — Toy Implementation (2 周)

**目标**: 单层 K=64，10K items subset，验证 K³ 搜索可行 + 三分量稳定

```python
# 1.1 数据: Toys 前 10000 items, dim=240 → (80, 80, 80) 三分割
# 1.2 ProductManifoldCodebook:
#     - C_s: K 码字 on S^79
#     - C_e: K 码字 on R^80
#     - C_h: K 码字 on H^79 (Lorentz model)
#     - product metric: d² = d_s² + d_e² + d_h²
# 1.3 训练: RSGD for 码本, SGD for encoder/decoder
# 1.4 诊断: 三码本利用率 + Kendall τ 矩阵 + 搜索时间
```

**D1.5 决策**:
- ✓ search_time < 50ms/item + utilization > 0.8 + Recall@10 ≥ HRQ baseline → proceed Phase 2
- ✗ Kendall τ 全 > 0.9 (冗余) → 重新设计分割
- ✗ 某码本 collapse → 调整 lr/L2

### Phase 2 — Full-Scale Single-Layer (2 周)

**目标**: K=256 全 Toys，与 HRQ 单层 / RQ-VAE 单层 / AQ 对比

```python
# baseline_1 = Task #15 RQ-VAE 单层
# baseline_2 = Task #7 HRQ 单层
# baseline_3 = AQ 单层
# current = PM-RQ 单层 (K=256)
# metrics: recall@10, codebook collapse rate per segment, cross-segment redundancy
```

### Phase 3 — Multi-Layer Cascade (2-3 周)

**目标**: 3 层 cascade，验证深层 V-info 改进

```python
# ProductManifoldRQ(num_layers=3, K=256, d_split=80)
# 跨层 residual: log/exp maps per segment
# V-info per layer per segment (用已有诊断工具)
# 对比 baseline: Phase 2 单层 PM-RQ + HRQ 3层
```

**D2 决策**:
- ✓ 深层 V-info 改进 vs RQ-VAE baseline → proceed Phase 4
- ✗ 没有改进 → 诊断三层 redundancy

### Phase 4 — Benchmarking & Ablation (1-2 周)

**目标**: 完整对标 + 消融

```python
# Baselines: RQ-VAE 3层 / HRQ 3层 / AQ 3层 / PM-RQ 3层
# Ablations:
#   A. 移除球面 → E×H
#   B. 移除双曲 → S×E
#   C. 用固定权重替代乘积流形
# 可视化: Kendall τ 矩阵演变, V-info per layer
```

**D3 决策**:
- ✓ PM-RQ Recall@10 显著高于 RQ-VAE/HRQ/AQ → 准备投稿 CIKM/RecSys
- △ 部分提升 → 调整 ablation 找最优组合
- ✗ 没有提升 → 终局性结论"多几何 RQ-VAE 整体收益有限"

**启动命令**（Phase 0 启动示例）:
```bash
# Phase 0: 几何诊断
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

python3 -c "
import torch
import geoopt
# δ-hyperbolicity on Toys embedding
emb = torch.load('logs/task87_s1_sentence_t5_base_inference/runs/task87_s1/pickle/merged_predictions_tensor.pt', weights_only=False)
print(f'emb shape: {emb.shape}')
# TODO: compute δ_4-tree
"
```

---

## 3. 决策触发 (vs baseline)

| Phase | 指标条件 | 决策 |
|-------|---------|------|
| **D0** | δ > 0.3 + 三方向非平凡几何 | ✓ proceed Phase 1 |
| **D0** | δ < 0.1 + 单方向主导 | ✗ 停止, 改做其他方向 |
| **D1.5** | search_time < 50ms/item + utilization > 0.8 + Recall@10 ≥ HRQ baseline | ✓ proceed Phase 2 |
| **D1.5** | Kendall τ 全 > 0.9 (冗余) | ✗ 重新设计分割或初始化 |
| **D1.5** | 某码本 collapse (utilization < 0.5) | ✗ 改进 lr 或加 L2 正则 |
| **D2** | 深层 V-info 显著改进 vs RQ-VAE | ✓ proceed Phase 4 |
| **D2** | 深层 V-info 无改进 | ✗ 诊断三层 redundancy |
| **D3** | PM-RQ Recall@10 > RQ-VAE + 3pp | ✓ 准备投稿 |
| **D3** | PM-RQ Recall@10 与 HRQ 持平 | △ 调整 ablation 找最优组合 |
| **D3** | PM-RQ Recall@10 < RQ-VAE | ✗ 终局结论: 多几何 RQ-VAE 收益有限 |

baseline 对标：
- Toys flat RQ-VAE Recall@5 = 0.0342 (paper Table 1 RQ-VAE Toys 行)
- Task #85 m=1 最佳 = Recall@5 = 0.02838 (83% of baseline)
- Task #87 TIGER-aligned baseline（若完成）作为新 baseline

---

## 4. 预算

| Phase | 周数 | GPU 时间 | 关键产出 |
|-------|------|---------|---------|
| 0 几何诊断 | 1 | CPU only | D0 报告 |
| 1 Toy test | 2 | ~6h single GPU | toy_metrics.json |
| 2 Full single-layer | 2 | ~12h single GPU | phase2_metrics.json |
| 3 Multi-layer cascade | 2-3 | ~24h single GPU | v_info_per_layer.json |
| 4 Benchmark + ablation | 1-2 | ~12h single GPU | final_table.csv |
| **总计** | **8-10 周** | **~54h GPU** | 论文 ready |

**注意**: Task #87 占据 cuda:0 至 ETA ~02:30 (R5 R7 约束不能抢), Phase 0/1 可立即在 cuda:3 启动 CPU-only 或轻 GPU 任务

---

## 5. 风险与缓解

**风险 1**: 三分量学习冗余信息 (Kendall τ 全 > 0.9)
- 原因: embedding 信息量有限, 三分割可能没有足够独立信号
- 缓解: D1.5 检测到后立即停止, 重新设计分割 (e.g. PCA 三分割, 或不同语义子空间)

**风险 2**: Riemannian 优化不稳定 (码本 collapse)
- 原因: geoopt 在 batch 内 RSGD 可能与 Adam encoder 优化器不兼容
- 缓解: 加 L2 正则 / 降低码本 lr / 用 RSGDAdam 混合优化器

**风险 3**: 乘积流形搜索 K³ 复杂度
- 球面 K=256, 欧氏 K=256, 双曲 K=256 → 256³ = 16.7M 组合, 实际不是全笛卡尔积而是各自 argmin + 加权 (本设计规避)
- 缓解: 各自独立 argmin + 乘积度量合成, 复杂度 O(K) 而非 O(K³)

**风险 4**: Toys dataset 信息不足
- 11924 items 可能不足以让乘积流形学到有意义的分割
- 缓解: Phase 2 全 Toys + 后续若需更大 dataset 再考虑 Amazon Beauty/Sports (Task 文件 R5 禁止, 需用户确认)

**风险 5**: 与 Task #87/Task #85 抢 GPU
- 缓解: Phase 0/1 可立即在 cuda:3 启动 (Task #87 占 cuda:0, Task #85 占 cuda:1/2)
- Phase 2-4 需 cuda:0/3 之一空闲, 等待 Task #87 Stage 3 启动后 cuda:0 释放

---

## 6. 完成度跟踪

### Phase 0: 几何诊断

- [ ] D0.1 计算 Toys sentence-t5-base embedding 的 δ-hyperbolicity
- [ ] D0.2 计算 sectional curvature 分布
- [ ] D0.3 UMAP 投影到 Poincaré ball / sphere / Euclidean 三方向可视化
- [ ] D0.4 写 D0 报告 `verdicts/task22_phase0_d0_report.md`
- [ ] D0.5 Go/No-Go 决策：proceed Phase 1 或停止

### Phase 1: Toy Implementation

- [ ] P1.1 实现 `ProductManifoldCodebook` (sphere K=64 + euclidean K=64 + hyperbolic K=64)
- [ ] P1.2 实现 component-wise log/exp residual maps
- [ ] P1.3 Toy 训练：100 epochs, K=64, 10K items subset
- [ ] P1.4 诊断：三码本利用率 + Kendall τ 矩阵 + 搜索时间
- [ ] P1.5 Go/No-Go 决策

### Phase 2: Full-Scale Single-Layer

- [ ] P2.1 全 Toys K=256 PM-RQ 单层训练
- [ ] P2.2 与 HRQ/RQ-VAE/AQ 单层 baseline 对比
- [ ] P2.3 写 phase2 metrics + verdict

### Phase 3: Multi-Layer Cascade

- [ ] P3.1 ProductManifoldRQ 三层 cascade 实现
- [ ] P3.2 V-info per layer per segment 诊断
- [ ] P3.3 三层 PM-RQ vs 单层 baseline 对比

### Phase 4: Benchmarking & Ablation

- [ ] P4.1 完整对标 RQ-VAE/HRQ/AQ/PM-RQ (Recall@1/10/100, MRR)
- [ ] P4.2 消融实验 A/B/C
- [ ] P4.3 最终 verdict + 论文表 + 准备投稿

---

## 7. 完成度判据

任务完成 = Phase 0-4 全跑完 + 最终 verdict 落盘 `verdicts/task22_result.md`

**完成定义** (per loop.md §9):
- D0-D3 决策表全有结论
- 每个 Phase 的 checkbox 全绿或显式标注 skip
- 最终表格 + verdict 文件存在且包含 `result:` 行

**提前结束条件**:
- D0 否证 → 写 verdict "embedding 不需要混合曲率, 任务提前终止"
- D1.5 否证 → 写 verdict "K³ 搜索/三分量不稳定, 任务提前终止"
- D3 否证 → 写 verdict "PM-RQ 未超过 RQ-VAE baseline, 多几何 RQ-VAE 收益有限"

---

## 8. 资源 & 依赖

**新增依赖** (需 pip install):
- `geoopt` (Riemannian 优化)
- `hyperbolic_diagnostic` 或自实现 δ-hyperbolicity

**已有依赖**:
- V-info 诊断工具 (Task #82 验证过)
- codebook collapse 诊断
- cos θ 度量

**GPU 资源**:
- cuda:3 当前空闲 (per nvidia-smi 2026-07-19 01:18 check)
- cuda:0 被 Task #87 占 (~02:30 释放)
- cuda:1/2 被 Task #85 占 (~25:00 patience 触发)

**禁止**: 修改 `src/` `data/` `task` 上游源码 (CLAUDE.md R6)
**禁止**: 改 datasets 至 Toys 之外 (CLAUDE.md R5)

---

## 9. 备注

本任务是 Task #85 的延伸但采用完全不同的架构路线：
- Task #85: 三几何独立 RQ-VAE (各自 cascade), 最后取一种 κ
- Task #22 (本): 乘积流形 (同一 cascade 内三种 κ 共存)

如 Task #22 验证通过，将为推荐系统的几何选择提供"无需选 κ"的方案——这是 PM-RQ 论文投稿的核心卖点。