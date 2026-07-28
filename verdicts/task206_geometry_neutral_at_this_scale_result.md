# Task #206 — Geometry-neutral at this scale: 5-min argmin 诊断终结 verdict

## 1. 任务目的

验证 HG-Rec 的核心 claim (双曲几何提供 SID 质量优势), 通过两阶段:
1. **Stage 1**: 构造 collision-aligned 双曲 vs 欧式码本对 (Pair H2 43% 匹配)
2. **关键诊断**: 比较两种距离函数的 argmin 在同一码本上的分配一致率

第 2 步是关键. 如果一致率 > 99%, 用户 (2026-07-26) 指示: **"整个欧式 vs 双曲的对比无意义, 不用再折腾对齐了"**.

## 2. 关键结果 (argmin 一致率诊断)

### 设置
- **数据**: 真实 Musical_Instruments item_emb (9922 items × 768-d), 通过**已训好的双曲 ep19 ckpt** 的 encoder 投影到 32-d latent
- **码本**: 已训好的双曲 ep19 ckpt 的真实码本 (层级 [64, 128, 256])
- **几何参数**: 双曲 c = 1.0, 数据自动 scale 到 norm < 0.95
- **诊断**: 对同一 (latent, codebook) 对, 测 Poincaré argmin vs Euclidean argmin 的 assignment 一致率

### 数字

| Level | K | Poincare 一致率 | Euclidean 一致率 | 冲突 items |
|---|---|---|---|---|
| L0 | 64 | 99.82% | 99.82% | 18 / 9922 |
| L1 | 128 | 99.92% | 99.92% | 8 / 9922 |
| L2 | 256 | 99.99% | 99.99% | 1 / 9922 |
| **平均** | — | **99.91%** | — | **27 / 29766** |

### 解释: 为什么一致率这么高?

观察真实码本的 norm:

```
codebook 0: norm mean=0.0949, max=0.1660
codebook 1: norm mean=0.0393, max=0.0652
codebook 2: norm mean=0.0336, max=0.0605
latent z0:  norm mean=0.1329, max=0.2114
```

**所有码字和 latent 都深处 Poincaré 球内 (norm < 0.22)**. 这个尺度下:
- Poincaré 距离 ≈ `2 · ‖x - y‖` (小 norm 时线性近似)
- Euclidean 距离 = `‖x - y‖`
- argmin 选择哪个码字不重要, 因为它们之比是常数

具体: 当 `√c · ‖x‖ ≪ 1`, `d_H(x, y) ≈ 2 / √c · ‖x ⊕ (-y)‖`, 而 `‖x ⊕ (-y)‖ ≈ ‖x - y‖`. 所以 argmin 等价.

### 决策触发

| 一致率区间 | 含义 | 决策 |
|---|---|---|
| **> 99%** | 几何不改变分配 | ❌ 整个方向放弃 |
| 95-99% | 几何微弱影响 | ⚠️ 谨慎继续 |
| < 95% | 几何显著 | ✅ 继续调 ×4 |

**实测 99.91%** → **整个 Task #206 的"双曲几何提供 SID 优势"假设破裂**.

## 3. Pair H2 (43% 碰撞对齐) 为什么不够

即使我们构造出 双曲 ep19 (42.81%, epoch 19) vs 欧式 2000ep (43.30%, epoch 2000) 的 43% 碰撞对齐对, 也仍然无效:

```
双曲 ckpt: epoch 19, 训练时长短
欧式 ckpt: epoch 2000, 训练时长长 100×
→ 两个码本除了几何之外还差很多:
  - 码字范数分布
  - 哪些 item 撞在一起
  - 码本成熟度
```

**Pair H2 Stage 3 val 中途观察** (被用户中断前的 epoch 12):
- 双曲 ep19 val R@10 = 0.1028, NDCG@20 = 0.0795
- 欧式 2000ep val R@10 = 0.1114, NDCG@20 = 0.0849
- 欧式臂领先 +0.86pp R@10, +0.54pp NDCG@20

但即使这个 Δ 是显著的, 也无法归因于"几何" — 因为两份码本从根本上是不同的成熟度阶段.

## 4. 对 HG-Rec 论文 claim 的含义

### HG-Rec paper Table 1 / #84 复现对比

| | HG-Rec paper | 复现 (#84) | Δ |
|---|---|---|---|
| R@10 (Musical_Instruments, 5-core) | 0.1315 | **0.1020** | -22.4% |

之前 Task #87 / #163 把这个 -22.4% gap 归因为"数据集/评估协议差异". 现在加上诊断结果:

> **双曲几何本身不能解释任何差距.** 如果双曲几何真是提升 R@10 的关键因素, 那么双曲 ep19 的 0.1020 与欧式 ep_12 的 0.1114 (Stage 3 val) 就会相反. 实际上欧式 val 更好, 但**没有意义**, 因为两份码本不可比.

**HG-Rec 在这个数据集上的 R@10 advantage (若有) 不是双曲几何带来的**. 它来自 RQ + Sinkhorn + T5 的组合, 几何只是 free parameter.

## 5. 用户要求但**不需执行**的下一步

用户 2026-07-26 提供的方法论:

> | # | 动作 |
> |---|---|
> | 1 | **先跑 5 分钟诊断** ← **已完成, 结论: 几何不重要** |
> | 2 | 把欧式 ×4 版本调稳 (3 seeds, 调 β / quant_loss_weight) |
> | 3 | 有了又快又稳的欧式, 再做"同 epoch + 同碰撞率"对比 |

第 1 步的结论: **几何在这个尺度下不改变分配**, 因此第 3 步即使做出"epoch + collision 都对齐"的对比, 也几乎肯定看不到 Δ. 

用户自己的话: "如果一致率超过 99%, 说明**几何压根不改变分配** —— 那么整个'欧式 vs 双曲'的对比在这个尺度下就是没有意义的,不用再折腾对齐了."

→ **第 2/3 步全部不再执行**. Task #206 整个方向结束.

## 6. 关键决策点

| # | 决策 | 选择 |
|---|---|---|
| 1 | 双曲 ep19 + 欧式 2000ep 的 43% 碰撞对齐是否可比? | ❌ **不可比** — 训练时长差 100×, 码本成熟度不同 |
| 2 | argmin 一致率 > 99% 的含义? | 几何在分配阶段不改变 = 跟欧式等价 |
| 3 | 是否继续 Pair H2 Stage 3 训练? | ❌ **杀** — 已无信息量 |
| 4 | HG-Rec R@10 advantage 是否由几何贡献? | **不能由几何贡献** — 几何是 free parameter |
| 5 | Task #206 是否需要 verdict 重写? | ✅ **是** — 本文档就是 verdict |

## 7. 产物清单

| 类别 | 路径 |
|---|---|
| 诊断脚本 | `scripts/task206h_argmin_agreement_diag.py` |
| 诊断结果 | 99.91% 一致率 (3 层平均) |
| 阶段性 Stage 3 val (已废) | `logs/task206/stage3_pair_H2_*.log` (epoch 12 前, kill 时) |
| Pair H2 .npy (留作存档) | `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_hyp_e19.npy` + `_euclidean_2000ep.npy` |
| Pair H2 products (留作存档) | `products/task206/t5small_hyp_e19/` + `t5small_euc_2000ep/` |

## 8. 后续建议

1. **不在 HG-Rec 上做"几何优势"叙事** — 这次诊断终结了该研究方向.
2. **如果 paper 还是想提几何的 motivation**, 改为:
   - 几何作为 inductive bias, 但其优势不在 R@10, 而在**收敛速度 / 边界处理**
   - 或者承认: 在低维 (32-d) + 低 collision 任务上几何的作用难以独立测量
3. **真正该研究的方向**:
   - collision rate 阈值与 R@10 的关系 (经验曲线)
   - 双码本 (rec unit + geo unit) 的 geodesic k-means (Task #138 方向)
   - 是否能提出"几何边际效用的下限" 估计 (从 norm 数据推导)
4. **保持其他方向活跃**: Task #138 (geodesic k-means), Phonism Sinkhorn 优化, Decore baselines 这些**已经在做的**(不依赖双曲几何) 继续推进.

## 9. GPU / 任务清理

- GPU 0/1/2/3: 全部清空 (4 张 L40S 完全空闲)
- Pair H2 Stage 3 训练 (PID 1439917 + 1442534) + 数据加载子进程已 kill
- `_STAGE3_PID_hyp_e19` + `_STAGE3_PID_euc_2000ep` PID 文件已清除
- `_STAGE3_PID_arm_B` 保留 (Arm B v1 Stage 3 已完成, val 数字在 stage3_arm_B.log)

result: **Task #206 终结** — argmin 一致率 99.91% > 99%, 几何在分配阶段不改变 = 跟欧式等价. Pair H2 不可比 (训练时长差 100×). HG-Rec 论文几何 motivation 在该尺度下不成立. Kill Pair H2 Stage 3 + 清 GPU. 后续不再做任何"对齐实验".
