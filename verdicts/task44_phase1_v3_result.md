# Task #44 Phase 1 Toy — 第三轮 STE + commitment 诊断

> **完成日期**: 2026-07-20
> **状态**: 🔴 **Phase 1 不通过 — PM-RQ 假设被 Toy 实验证伪**
> **任务**: 验证 PM-RQ 是否比标准 RQ-VAE 更优 (Phase 1 toy, K=64, 10K items, 3 层, 2000 steps)

---

## 1. 三轮尝试结果汇总

| 轮次 | 关键修改 | B (baseline) | C (PM-RQ 3 段) | D (PM-RQ 2 段) | C/B ratio |
|------|---------|------|------|------|------|
| V1 | 原始 argmin + dist_scale | 0.275 | 1.926 | 0.631 | **7.0×** ❌ |
| V2 | + STE 量化器 | 0.275 | 2.394 | 0.895 | **8.7×** ❌ |
| **V3** | **+ d_total commitment loss** | **0.275** | **3.512** | **1.938** | **12.8×** ❌ |

> ⚠️ V3 加 commitment loss 后, fusion_logits 终于能学, C loss 反而更高是因为 commitment loss 加进了 total_loss。决策仍以 **C loss vs B loss 比** 为准。

---

## 2. V3 关键观察 (这次实验揭示的核心问题)

### 2.1 fusion_logits 终于动了 — 但学到了 trivial solution

| 参数 | 初始 | V2 结束 | **V3 结束** |
|------|------|------|------|
| C fusion w_s | 0.333 | 0.333 (不变) | **0.022** (丢弃) |
| C fusion w_e | 0.333 | 0.333 (不变) | **0.900** (主导) |
| C fusion w_h | 0.333 | 0.333 (不变) | **0.078** (丢弃) |
| C κ_s | +0.8446 | -8.99 | **+34.29** (强球) |
| C κ_h | -1.0586 | -1.59 | **-9.91** (强双曲) |
| D fusion w_s | 0.5 | 0.5 (不变) | **0.030** (丢弃) |
| D fusion w_e | 0.5 | 0.5 (不变) | **0.970** (主导) |

**结论**: 在 toy toy 设定下, optimizer 学会的最优策略是 **完全丢弃 sphere 和 hyperbolic 段**,只用 euclidean。这是 trivial solution。

### 2.2 D (无 hyperbolic) 比 C (有 hyperbolic) 反而更好

| 组 | final_loss | ratio |
|---|---|---|
| C (3 段) | 3.512 | 12.8× B |
| **D (2 段: sphere+euclid)** | **1.938** | 7.1× B |

**含义**: hyperbolic 段在 toy 设定下**是负贡献**(去掉它反而更好)。结合 w_h 从 0.33 跌到 0.078,说明 hyperbolic 距离信号确实是反信号。

### 2.3 κ_sphere 从 +0.8446 跑到 +34

κ_sphere 没有停留在 D0 诊断值附近,而是被 optimizer 推到强球面分支 (κ>30)。这进一步说明:
- MCKG 子空间 0 (sphere) 的原始 embedding 几何信号弱
- Optimizer 宁愿推到 κ→+∞ (极端球面) 也不愿停留 D0 附近
- 与 Task #42 结论一致: subspace_0 Riemannian 改进仅为 0.1476 (相对改进小)

---

## 3. Phase 1 Go/No-Go 决策

| 决策项 | 阈值 | 实测 | 结果 |
|--------|------|------|------|
| C ≤ B × 1.05 | C ≤ 0.289 | C = 3.512 | ❌ |
| fusion_logits 学到非平凡权重 | 任意 w ∈ [0.1, 0.9] | w=[0.022, 0.9, 0.078] | ⚠️ 部分(只 w_e 有效) |
| Hyperbolic 必要 (D 比 C 差 10%+) | D ≥ C × 1.10 | D = 1.938 < C = 3.512 | ❌ 反向 |
| **最终决策** | 全部 Go | 全部 No-Go | **🔴 不通过** |

---

## 4. 假设是否被证伪?

**Phase 1 设计的目标假设**:
> "3 段子空间的乘积流形距离 ≥ 单段欧氏距离 (fused_64d), 即几何感知信号能补充欧氏距离"

**实验结果**:
- ❌ **否证**: fusion_logits 学到的权重 [0.022, 0.9, 0.078] 表明, 在 toy toy (K=64, 10K items, 3 层) 设定下, **几何信号是冗余/反信号**
- 三个 Go/No-Go 全部不通过
- 与 Task #42 结论一致: subspace_2_hyperbolic Riemannian 改进 = 负

**可能的解释** (待 Phase 2 全规模验证前不可定论):
1. **MCKG 子空间本身就是 weak signal**: fused_64d 已经捕获了主要结构,子空间额外信号是噪声
2. **toy toy K=64 不够**: 真 PM-RQ 需要更大 K 才能让不同流形段学到不同模式
3. **toy toy 10K items 不够**: Toys 全量 11924 但 toy 只取 10K,可能损失了部分结构
4. **架构问题**: PM-RQ 把每个子空间当独立流形,但 MCKG 子空间可能是"投影后的统一坐标",不是真流形点

---

## 5. 产物

| 产物 | 路径 |
|------|------|
| V1 脚本 | `scripts/task44_pmrq_toy.py` |
| V2 脚本 | `scripts/task44_pmrq_toy.py` (STE 版, 同文件覆盖) |
| **V3 脚本** | **`scripts/task44_pmrq_toy_v3.py`** (commitment loss, 最终版) |
| V1 log | `logs/task44_phase1.log` |
| V2 log | `logs/task44_phase1_v2.log` |
| **V3 log** | **`logs/task44_phase1_v3.log`** |
| V1 summary | `products/task44_pmrq_phase1/task44_phase1_summary.json` |
| V2 summary | `products/task44_pmrq_phase1/task44_phase1_v2_summary.json` |
| **V3 summary** | **`products/task44_pmrq_phase1/task44_phase1_v3_summary.json`** |
| 模型 ckpt | `products/task44_pmrq_phase1/model_{B,C,D}_pmrq_*_v3.pt` |

---

## 6. 后续决策

| 方案 | 描述 | 决策 |
|------|------|------|
| **A. 终止 PM-RQ** | Phase 1 否证, 停止投入, 改做其他方向 | 提交用户决策 |
| **B. Phase 2 全规模** | K=256, 11924 items 全 Toys, 验证 toy 否定是否真实 | 需要用户授权 + GPU |
| **C. 重设架构** | 不再让 fusion_logits 学习, 改为固定权重 + 训练更大 K 的 fusion codebook | 需新实验设计 |

**推荐 A**: 三轮 toy 实验一致否定,继续投入 ROI 低。但保留代码供用户后续决定。

---

## 7. 任务时间线

- **V1 (原始 argmin)**: 2026-07-19 — B=0.275, C=1.926, D=0.631 (7× 退化)
- **V2 (STE)**: 2026-07-20 — B=0.275, C=2.394, D=0.895 (8.7× 退化, fusion_logits 仍不动)
- **V3 (commitment)**: 2026-07-20 — B=0.275, C=3.512, D=1.938 (12.8× 退化, fusion_logits 终于学到 trivial solution)

---

**result:** Task #44 Phase 1 (PM-RQ Toy, 三轮) **不通过**, 三个 Go/No-Go 全部失败, fusion_logits 在第三轮学会 trivial solution (只用 euclid,丢弃 sphere+hyperbolic),证明 PM-RQ 假设在 toy 设定下被否证。**推荐终止 PM-RQ 主线方向**,等待用户决策 A/B/C。

result: Task #44 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
