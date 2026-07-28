# Task #207 — 欧式 vs 双曲 collision 对齐比较 (paper 基调实验)

> **任务目的**: 在 collision 率对齐的条件下, 比较欧式全距离 vs 双曲全距离 (c=1) 的 Recall@10, 确定 HG-Rec 的几何贡献是本质性的还是工程级别的.
>
> 结果决定 paper 基调:
> - 双曲 ≈ 欧式 → "HG-Rec κ 是工程优化" (c=1 未激活几何)
> - 双曲 > 欧式 → "几何结构有本质贡献, c∈[1,10] + 层级 margin loss"

> **完成日期**: (in progress)
> **状态**: 🔵 设计中

---

## 1. 背景

**来源**: 用户 2026-07-26 在 Task #205 结论中明确要求 (见 memory `task205-curvature-leverage-refuted.md`):

> "欧式 vs 双曲 (collision 对齐) 实验: HG-Rec 全程, inner product → L2. 手动控制 collision 对齐, 比较 Recall. 答: 双曲 = 欧式 → paper 写 'HG-Rec κ 是工程优化'. 答: 双曲 > 欧式 → paper 写 '几何结构本质, 需 c∈[1,10] + L_hier margin'. 这个决定 paper 基调, **必须做**."

**前置 7 阶段 κ 实验结论**:
- #84 baseline: HG-Rec c=1, collision~13%, R@10=0.1020
- #199: c 固定扫描 {1,10,30,100} — c=30 退化, c=100 坍缩
- #199+#201+#203: exp(θ) κ 参数化 3 次独立证伪
- #204: c=1 vs c=10 Δ 0.86% (第三种结果, 统计等价)
- #206: 几何激活路线 (c≥93 不可训练) — **终止**
- **累积结论**: c=1 是次优但接近最优, c∈[1,10] 是安全窗口

**本任务要回答**: 即使在 c=1 这个"安全但几乎欧式等价"的窗口, 双曲距离是否仍然提供可测量的优势?

### 1.1 codebase 就绪状态

| Patch | 文件 | 状态 |
|-------|------|------|
| `--euclidean_qloss True` (MSE codebook loss) | `utils.py` HVectorQuantization | ✅ 已有 (#182) |
| `--loss_type mse` (MSE recon loss) | `hrqvae.py` HRQVAE.compute_loss | ✅ 已有 |
| `--loss_type poincare` (Poincaré recon loss) | `hrqvae.py` HRQVAE.compute_loss | ✅ 已有 (baseline) |

**只需要启动命令改 flag, 不需要新代码 patch.**

---

## 2. 实验设计

### Phase 0: Stage 1 训练 (2 臂 × 1 GPU, ~2h)

| 臂 | 量化 loss | 重构 loss | 曲率 | 预期 collision |
|----|-----------|-----------|------|---------------|
| **H** (双曲, baseline) | Poincaré distance | Poincaré MSE | c=1 | ~13% |
| **E** (欧式) | MSE (`--euclidean_qloss`) | MSE (`--loss_type mse`) | c=1 (无效) | 待定 |

**其余全对齐**: 500 epoch, batch_size=1024, K0=64, num_emb_list=[64,128,256], e_dim=32, β=0.5, sk_eps=[0,0,0], kmeans_init=True, lr=1e-3

### Phase 1: collision 对齐 (if needed)

若 Phase 0 两臂 collision 相差 > 3 个百分点, 调整欧式臂的 β 或 K0 使 collision 匹配到 ~13%.

| 旋钮 | 效果 | 方向 |
|------|------|------|
| β (commitment weight) | β↑ → collision↓ | 调高 β 使欧式 arm collision 追上双曲 |
| K0 (L0 codebook size) | K0↑ → collision↓ (见 #194) | 增大 K0 |

### Phase 2: Stage 2 SID + Stage 3+4 评估 (1 GPU, ~4h)

若 Phase 0 两臂 collision 已接近 (<3% 差), 直接推进:
1. Stage 2 SID 推断 (Sinkhorn-Knopp + 4th-digit dedup)
2. Stage 3 T5-mini 9.18M 训练 (200 epoch, early_stop=20)
3. Stage 4 test R@10 评估

### Phase 3: 层级 margin loss 探针 (方向 D 预览, 可选)

若 Phase 2 显示双曲 > 欧式, 在欧式 arm 上追加:
```python
# 在欧式训练中加层级 margin loss
L_hier = Σ max(0, d_E(a,b) − d_E(a,e) + margin)
# 其中 a,b 同顶层码, a,e 不同顶层码
```
看 margin loss 能否补上欧式跟双曲的 gap. 若补上 → 几何贡献完全可被 margin loss 替代 → paper 写 "几何通过层级分离起作用".

---

## 3. 决策触发 (vs HG-Rec baseline R@10=0.1020)

| 条件 | 含义 | 决策 |
|------|------|------|
| **R@10_E ≥ 0.1020** | 欧式≥双曲, κ 无本质贡献 | paper: "κ 是工程优化, 几何未激活" |
| **R@10_E ∈ [0.095, 0.102)** | 欧式略低于双曲 (在 ±5% 噪声范围) | paper: "c=1 窗口下几何优势微小" |
| **R@10_E < 0.095** | 欧式明显低于双曲 (>5%) | paper: "几何结构有本质贡献" |
| **Phase 3: margin 补上 gap** | 几何贡献 = 层级分离 | paper: "曲率应复用为层级 margin" |

---

## 4. 预算

| Phase | 估算时间 |
|-------|---------|
| Phase 0 Stage 1 (2 臂并发 2 GPU) | ~2h (500 epoch) |
| Phase 1 collision 对齐 (if needed) | +1h |
| Phase 2 Stage 2 + 3 + 4 | ~4h (1 GPU) |
| Phase 3 (可选) | +2h |
| 总计 | ~4-8h (并发) |

---

## 5. 风险与缓解

**风险 1: 欧式 arm 直接坍缩** (类似 #182 collision 83%) → 缓解: 增大 β 到 1.0 或启用 Sinkhorn 稳定化
**风险 2: collision 差异太大无法对齐** → 缓解: 改变 β 或 K0 重训欧式 (额外 ~2h)
**风险 3: 两臂 R@10 差异太小 (<1%), 统计不显著** → 缓解: 报告 Δ, 不做显著性测试, 承认噪声等价
**风险 4: 欧式 arm 需要新代码 patch** → 排查: `--euclidean_qloss True --loss_type mse` 是否已涵盖全欧式训练

---

## 6. 完成度跟踪

- [ ] Phase 0: 启动 2 臂
- [ ] Phase 0: Stage 1 收敛确认 (collision/recon_loss)
- [ ] Phase 1: collision 对齐 (if needed)
- [ ] Phase 2: Stage 2 SID + Stage 3 T5 训练
- [ ] Phase 2: Stage 4 Recall 评估
- [ ] Phase 3: 层级 margin loss 探针 (可选)
- [ ] 写 verdict + 更新 paper narrative
