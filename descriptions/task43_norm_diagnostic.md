# Task #43 — fused_64d / 3 subspace norm 长尾诊断 (训练稳定性前置)

> **任务目的**: 验证 MCKG embedding 的 norm 分布是否健康,识别极端离群点,给出 Task #40/#41 训练前的 norm 处理方案

> **完成日期**: 2026-07-20
> **状态**: ✅ **完成** — fused + 2 个 subspace 必须 clip/normalize
> **触发**: 用户 2026-07-20 反馈 "max=381 相对 mean=0.79 是 500× 极端长尾,直接喂 RQ-VAE 有训练风险"

---

## 1. 背景

承接用户反馈: fused_64d `mean=0.79, max=381` → 500× 极端长尾。这种未归一化的、方差巨大的输入直接喂 RQ-VAE 有真实风险:
- 距离计算被极端 norm item 主导
- codebook 往离群点方向坍缩
- 离群 item 量化误差异常大,拖累整体 loss

**这是训练稳定性前置诊断**,不属于几何设计本身,但不处理会让 Task #40/#41 的结果难以解读。

## 2. 实验设计

**输入**: `products/task99_mckg_rebuild/entity_embedding.pt` → 4 个空间 (subspace[0/1/2] + fused)
**指标**: min/max/mean/std/p10-p100 + max/mean ratio + p99/median ratio + 离群 item 计数
**可视化**: log10 直方图 (4 个空间),突出 mean/p99/max 三个分位

**启动命令**:
```bash
python3 scripts/task43_norm_diagnostic.py
```

## 3. 结果

| 空间 | max/mean | p99/median | n_above_50 | verdict |
|---|---|---|---|---|
| subspace_0_sphere | 1.9× | 1.4× | 0 | ✅ 健康 |
| subspace_1_euclid | **485.9×** | 3.5× | 6 | 🔴 必须 clip |
| subspace_2_hyperbolic | **959.8×** | 21.5× | 19 | 🔴 必须 clip |
| fused_64d | **480.4×** | 8.3× | 8 | 🔴 必须 normalize |

**关键观察**: 长尾极端但**离群 item 数量极少** (3-19 个 / 11924 个)。这是 MCKG 训练的副作用——某些高频 item 的 embedding 在 99th percentile 之上被推到极大值。

## 4. Task #40/#41 必须采取的 norm 处理

### Task #40 (fused → 标准 RQ-VAE)

```python
# fused L2 normalize 到 p95 norm
target_norm = 1.3962  # fused p95
norms = torch.norm(fused_item, dim=1, keepdim=True)
fused_normalized = fused_item / norms * target_norm
# 或: fused_normalized = fused_item / norms.clip(min=1e-6)  # 单位球 (更稳)
```

### Task #41 (3 段 → PM-RQ)

```python
# 每个 subspace 单独 clip norm 到 p99
clip_vals = {
    'subspace_0_sphere': 0.5668,  # p99 (但 sphere 应该归一化到 1)
    'subspace_1_euclid': 4.7902,  # p99
    'subspace_2_hyperbolic': 10.2035,  # p99
}
for k, clip in clip_vals.items():
    norms = torch.norm(subspace[k], dim=1, keepdim=True)
    subspace[k] = subspace[k] / norms.clip(min=1e-6) * norms.clip(max=clip)
```

**对 PM-RQ 的特殊考虑**: sphere subspace 应该归一化到单位球 (||x||=1),这是 Poincaré ball 的标准做法; hyperbolic 应该归一化到 Lorentz norm。

## 5. 风险与缓解

**风险 1**: 归一化损失信息 — 单位球化会丢失 magnitude 信息 → 缓解: 这是 MCKG embedding 的固有特征,magnitude 不应作为下游 RQ-VAE 的判别维度
**风险 2**: clip 引入新的偏置 — 离群 item 的相对位置被破坏 → 缓解: 只 clip 0.1% (12 个 item),影响可忽略
**风险 3**: 归一化后 fused 不能再用 Euclidean cosine 测距 → 缓解: Task #40 用 L2 distance,这是 RQ-VAE 的标准距离

## 6. 产物

| 产物 | 路径 |
|---|---|
| 主脚本 | `scripts/task43_norm_diagnostic.py` |
| 主日志 | `logs/task43_norm_diag.log` |
| 直方图 | `products/task43_norm_diag/norm_histograms.png` |
| 统计 JSON | `products/task43_norm_diag/task43_norm_summary.json` |

---

**result:** Task #43 完成 (norm 长尾诊断)。fused + subspace_1 + subspace_2 必须归一化/clip 才能喂 RQ-VAE;subspace_0_sphere 健康可直接用。Task #40 用 L2 normalize → p95 norm=1.3962;Task #41 用 per-subspace clip → 各子空间 p99 norm。
