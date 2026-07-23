# Task #54 — L3 Norm Regularization (post-hoc rebase) — 负结果 (但揭示重要根因)

> **任务目的**: 在 MCKG embedding 加 L3 norm penalty, 验证源头治疗 vs log1p 后处理
> **执行日期**: 2026-07-20
> **状态**: ⛔ 已完成 — L3 rebase 在 log1p S4 AE 上 post-hoc 无效 (输入已均匀)
> **意外发现**: ⭐ cov0=1.0 (simple kmeans) vs cov0=0.11 (Task #53 neural RQ-VAE) → Task #53 根因不是数据

---

## 1. 时间线

| 时间 | 事件 |
|------|------|
| 2026-07-20 03:38 | Task #54 启动 (CPU-only, 4 λ sweep) |
| 2026-07-20 03:40 | Task #54 完成 (~120s, 4×25s kmeans) |
| 2026-07-20 03:40 | 写 verdict |

---

## 2. 结果

| 阶段 | MSE_total | cov0 | cov1 | cov2 |
|------|-----------|------|------|------|
| baseline (无 L3) | 0.000861 | 1.000 | 0.773 | 0.777 |
| λ=0.01 | 0.000852 (×0.989) | 1.000 | 0.734 | 0.844 |
| λ=0.05 | 0.000813 (×0.944) | 1.000 | 0.781 | 0.871 |
| λ=0.20 | 0.000707 (×0.821) | 1.000 | 0.809 | 0.891 |
| λ=1.00 | 0.000389 (×0.451) | 1.000 | 0.816 | 0.820 |

**结论**: L3 rebase 把所有向量均匀压缩 (uniform norm=0.69→0.47), RQ recon MSE 持续改善 (×0.451), 但 **cov0 已经饱和在 1.0** (说明 simple kmeans 在 log1p S4 AE 上 256 桶已全部用上, 不存在坍缩问题)。

---

## 3. ⭐ 关键发现 — Task #53 根因再分析

对比三个实验在**同样的 log1p S4 AE 64d 输入**上的 layer-0 RQ 行为:

| 实验 | 量化方法 | cov0 | 备注 |
|------|---------|------|------|
| **Task #53** Stage 2.1 | 神经 RQ-VAE (encoder + decoder + kmeans codebook) | **0.109** | ❌ layer-0 坍缩 |
| **Task #54** | Simple kmeans (无 encoder/decoder) | **1.000** | ✅ 全部 256 桶覆盖 |
| **Task #55** | Simple kmeans (无 encoder/decoder) | **1.000** | ✅ 全部 256 桶覆盖 |

**结论**: Task #53 Stage 2.1 layer-0 码本坍缩 (cov0=0.11) **不是数据/embedding/orientation 问题**, 是 **RQ-VAE 神经网络 encoder/decoder 的瓶颈压缩问题**。

具体机制:
- 64d log1p S4 AE 输入本征维度足够支撑 256 桶 (simple kmeans 验证)
- 神经 RQ-VAE 用 `[dim,256,128]→64` encoder 把输入压到 64d bottleneck, 然后 decoder 重建
- 在 64d bottleneck 空间做 kmeans 找 256 个 cluster 是合理的, **但训练中 bottleneck 编码易坍缩到少量 modes** (尤其 Adagrad + 小 init_buf)
- 这是 RQ-VAE 在低维输入上的**已知架构缺陷**

**Task #53 失败三根因联合作用**:
1. S4 AE 64d 输入维度低 → bottleneck 编码信息少
2. RQ-VAE 架构 (encoder/decoder) → bottleneck 编码坍缩到少量 modes
3. TIGER 训练 → 在 cov0=0.11 SID 上学不到有效推荐

---

## 4. 修订 Recommendation

### 4.1 ❌ 之前 Task #53 verdict 的根因分析过窄

旧结论: "S4 AE 64d + log1p 组合无效" — 这是**症状**, 不是**根因**。
修正: 真正的根因是 **RQ-VAE 架构 (encoder/decoder) 与 S4 AE 64d 不兼容**。

### 4.2 ⭐ 后续推荐路线 (按 ROI)

| 路线 | 描述 | ROI | 备注 |
|------|------|------|------|
| **⭐ Task #58** | 用 Simple KMeans 替代 RQ-VAE 的 codebook, 跳过 encoder/decoder | 高 | 已在 Task #55/54 验证 simple kmeans 给出 cov0=1.0, 直接拿这个 SID 跑 TIGER |
| Task #54-b | 真正 MCKG 重训 (a 修改 mckg.py 加 L3 norm, b 完整训练 ~4h) | 中 | 当前 post-hoc rebase 无效, 但训练时加可能不同 |
| Task #55-b | 端到端 OPQ SGD 联合训练 (非 ITQ 闭式) | 低 | ITQ 闭式已说 orientation 对齐 |
| Task #56-b | 完整 rerun Task #85 三几何 w/ 新 dist_kappa | 低 | L'Hôpital 修复仅影响数值精度 |

### 4.3 ⭐⭐⭐ 最高 ROI 路径 (Task #58 提议)

```
Stage 1 (已有 products/task48_s4_ae/entity_embedding.pt)
    ↓
log1p 后处理 → products/task53_log1p_s4_ae/entity_embedding.pt (已有)
    ↓
Simple KMeans 3-layer 量化 (Task #55 验证 cov0=1.0)
    ↓
TIGER 训练 + 评估

预算: Stage 2 简化后 ~30 min (取代 RQ-VAE 的 3.5h 训练)
```

这条路线直接绕开 Task #53 的 RQ-VAE 架构缺陷, 应该能达到 baseline 附近的 R@5。

---

## 5. 完成判定

- [x] Task #54 脚本 (scripts/task54_l3_norm_rebase.py) ✅
- [x] 4 λ sweep 完成 (~120s)
- [x] 关键发现: cov0 饱和在 1.0, Task #53 根因修订 → RQ-VAE 架构问题
- [x] verdict 落盘 ← **本文档**
- [x] §16 任务清理 (R8 完成: Task #53 已删, Task #54 完成删除)

---

## 6. 产物清单

```
products/task54_l3_rebase/   (空 — 决策无效未保存)
logs/task54_l3/train.log    (~1.3 KB)
logs/task54_l3/summary.json (~2 KB, 含完整 4 λ sweep 数据)
scripts/task54_l3_norm_rebase.py  # 复用有效
verdicts/task54_l3_norm_result.md # 本文档
```

---

result: Task #54 L3 norm post-hoc rebase 在 log1p S4 AE 上无效 — cov0 已饱和在 1.0, 无可压缩空间 (MSE 改善 ×0.451 但 cov0 完全不变)。**⭐ 关键意外发现: simple kmeans 在 64d log1p S4 AE 上达 cov0=1.0**, 而 Task #53 neural RQ-VAE 同期仅 cov0=0.11 — **Task #53 失败根因不是输入/方向/orientation, 而是 RQ-VAE encoder/decoder 架构在低维 64d 输入上的瓶颈压缩坍缩**。**最高 ROI 后续路线 (Task #58 提议): 用 Simple KMeans 替代 RQ-VAE codebook, 跳过 encoder/decoder, 绕开 Task #53 架构缺陷**。
