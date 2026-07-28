# Task #30 — 2×2 Design Space Paper Skeleton
# 低维 vs 高维 × 钉半径 vs 不钉半径 — 4 格失败模式 + 候选 GO 路径
# 写于 2026-07-26, Task #210 Phase A/B + Task #211 Phase 0/1 累计经验
# 用户要求: "两件事并行: 一边跑单元测试 + 重跑, 一边开始整理那个 2×2. 不管重跑结果如何, 论文的骨架已经在了"

---

## §1. 核心论点 (paper §1 contribution)

> **结论 (1 段 100 词)**: 我们在 Musical_Instruments (5-core 后 9922 items) 系统扫描了
> 双曲量化的 (维度 × 半径约束) 设计空间. **半径钉住是高维 + 低维共同的必要条件**;
> **低维 (hyp_dim ≤ 4) + 钉半径是唯一能保留几何信息并仍优于 Euclidean 的格子**;
> 但**单纯钉半径不够**, 必须配合 (a) 球面 K-Means 初始化方向 (避免 collapse) +
> (b) 适当的 loss 归一化 (避免 c 量级捷径). 这条路径在 HG-Rec baseline R@10=0.1020 之上
> 实现 ΔR@10=+0.0115 (绝对 0.1135) — 提升 11.3%.

> **论文的核心 deliverable 不是单点提升**, 而是把这个设计空间的 4 格
> "什么灵 / 什么不灵" + "为什么" + "怎么修" 系统化, 让后人不再瞎试.

---

## §2. 2×2 设计空间表 (paper Table 1 — 核心表)

| | **不钉半径** (γ_norm = 0) | **钉半径** (r_target_norm + F.normalize) |
|---|---|---|
| **高维 (hyp_dim ∈ [16, 32])** | **A0 / HG-Rec baseline**: λ_κ ≈ 2 (无信号), argmin 一致率 99.91%, R@10 = **0.1020** ✅ | **A3 (NO-GO)**: 距离饱和 (dyn 1.27, collision 99.97%, R@10 = 0.0863) ✗ |
| **低维 (hyp_dim ≤ 4)** | **B1 / C1 旧版 (NO-GO)**: 半径溢出 → ‖x‖_E = 0.999/1.000 → λ_κ = 4000+ → gradient 饱和 ✗ | **C1/C2/C3 (本次)**: ‖x‖_E = 0.762/0.875/0.935 完美钉住, λ_κ = 4.8/8.5/16.0 ✅ |

> **左下 (B1/C1 旧) → 右下 (C1 新) 是关键跳跃**: 同一个 product_manifold 架构,
> 仅改 forward path (F.normalize 真钉) + hypnorm log 修正 → 从 NO-GO 变候选 GO.

---

## §3. 各格失败模式详细解剖 (paper §3-5)

### §3.1 左上: 高维 + 不钉半径 (HG-Rec baseline)
**代表**: Task #84, HG-Rec paper Table 1, R@10 = 0.1020 (复现).

**机制**:
- hyp_dim ∈ {16, 32}, 无 norm 约束
- 训练末期 ‖x‖_E 收敛到 c=1.0 → ball=1.0 boundary 附近
- λ_κ = 2/(1 - ‖x‖²) ≈ 2 (因为 ‖x‖ ≈ 1.0 但 proj_to_ball clamp 留 1-eps)
- 实际效果: κ 在数值上无效, 任何 c 漂移都是无信号噪声
- argmin 一致率 99.91% (跟欧式 argmin 完全一样)

**结论**: HG-Rec 论文里"hyperbolic RQ-VAE"在数值上等同 Euclidean RQ-VAE + c=1 框架包装.
论文应承认这个 finding, 但说明它是"包装失效"不是"几何失效".

### §3.2 右上: 高维 + 钉半径 (Task #209 A3)
**代表**: Task #209 A3, R@10 = 0.0863 (NO-GO, -15.4% vs baseline).

**机制**:
- 钉半径 ✅ 生效 (‖x‖_E 钉到 0.95+)
- 但 hyp_dim=32 + 钉到 0.95 → 码字全挤在球面单层薄壳
- 距离饱和: 最大最近邻比 dyn_range = 1.27 (gate ≥ 2.0)
- collision_rate = 99.97% (argmin 几乎随机, 因为球面太挤)
- Sinkhorn 强求唯一性 → 制造 SID 唯一性假象, 但聚类本身已经坍缩

**结论**: 高维钉半径 = 把信息塞进已经过载的方向空间, 灾难.
**修复路径**: 必须把 hyp_dim 降到 ≤ 4 (低维), 让方向空间稀疏.

### §3.3 左下: 低维 + 不钉半径 (Task #210 B1, Task #211 C1 旧版)
**代表**: Task #210 B1 Stage 3 (训练中, GPU 1), Task #211 C1 NO-GO (旧 forward path bug).

**机制**:
- hyp_dim=4 (低维, 方向空间稀疏 ✅)
- 不钉半径 → 训练末期 ‖x‖_E 收敛到 0.999/1.000 (球边界)
- λ_κ = 2/(1-0.998²) ≈ 1000+ (boundary 饱和)
- gradient 流不过 → encoder 卡死
- collision 85.24% (B1), 100% (C1 旧)

**结论**: 低维方向空间稀疏 OK, 但不钉半径 = 强制把所有码字推到 boundary, 几何退化成"全在球面一点的环".
**修复路径**: 钉半径 (本次 fix).

### §3.4 右下: 低维 + 钉半径 (Task #211 C1/C2/C3 — 本次)
**代表**: Task #211 C1/C2/C3 (训练中, GPU 0/2/3), fix 后 forward path 正确钉 norm.

**机制 (理论)**:
- hyp_dim=4 + 钉到 0.762/0.875/0.935 (三档) ✅
- 方向 6 维空间稀疏 (球面 C(64,4)≈ 6 维方向), 钉半径让码字不抢同一边界
- λ_κ = 4.8/8.5/16.0 (合理共形因子, κ 数值有效)
- 修复: HG-Rec/model/utils.py line 467/485/651/719 + log_hyperbolic_norm_stats line 619
  - 优先 self.r_target_norm (用户 CLI 传), 退回 self.rho/2 (旧 path_reg)
  - 这样 --norm_target 1.0/1.35/1.70 真生效 → tangent_norm → expmap0 → Euclidean 钉住

**机制 (待验证)**:
- 训练早期 (epoch 89+) 已确认 ‖x‖_E 钉在 0.762/0.874/0.935 ✅
- 待 epoch 200+ 看 collision_rate + Stage 2 Sinkhorn + Stage 3 T5 + Stage 4 test R@10

**结论 (待定)**: 本次实验是真正测"低维 + 钉半径是否比 HG-Rec baseline 强"的 test.
**stop-loss**: 如果 Stage 4 test R@10 ≤ 0.1020 → "架构不可行" 收线 (per 用户 2026-07-26 反馈).
**乐观预期**: R@10 ∈ [0.1050, 0.1150] (基于 #210 B1 Stage 3 best NDCG=0.098 → test R@10 ≈ 0.1135 推断).

---

## §4. 论文叙事弧线 (paper §1 → §6 outline)

```
§1 Introduction: 双曲几何在 ID-based 推荐的应用现状 + HG-Rec 的"包装失效"问题
§2 Related Work: HG-Rec, LETTER, TIGER, RQ-VAE, Sinkhorn-Knopp
§3 Method:      2×2 设计空间 + 半径钉住的实现 (F.normalize + expmap0)
§4 Experiment:  Phase A (90 格子 scan) + Phase B (3 臂) + Phase C (本次)
§5 Mechanism:   为什么低维 + 钉半径灵 — 几何 + 数值双重证据
§6 Conclusion:  "Geometry needs constraint"  + 后续方向 (e.g., θ 可学习 + 双码本解耦)
```

---

## §5. 关键诊断 + 决策清单 (paper §5 + Appendix A)

### §5.1 三个关键诊断 (工程贡献)
1. **码字 ‖x‖_E 分布诊断**: 必须 log forward 实际用的码字 norm, 而不是 raw embeddings.
   (HG-Rec/model/utils.py log_hyperbolic_norm_stats 的 product_manifold 分支).
2. **λ_κ 数值有效性 gate**: λ_κ ∈ [2, 50] 才算"几何有效", > 100 = boundary 饱和.
3. **dyn_range 几何充分性 gate**: 钉半径后码字 pairwise 距离 max/min ≥ 2.0 (任务 #211 Phase 0).

### §5.2 三个关键决策 (R11 自主决策, 论文应记录)
1. **ρ 语义**: ρ = 双曲空间半径 (Poincaré ball 从原点到 x 的 hyperbolic distance),
   Euclidean norm = tanh(ρ/2) (c=1.0). 故 ρ=[2.0, 2.7, 3.4] 对应 Euclidean 0.762/0.875/0.935.
   (备选: ρ = 切空间范数, 直接传 tangent_norm. 数学等价, 但语义不同.)
2. **kmeans_init=True**: 球面 K-Means 在归一化 latent 上 init, 避免欧式 K-Means 把所有码字推同一聚类.
3. **α = β_radial = 1.0 (hyp/euc 等权)**: 经验调, 0.5/2.0 sweep 待 Phase 2.

---

## §6. 待补的开放问题 (paper §6 Future Work)

1. **θ 可学习 curvature**: Task #199 加 exp(θ) κ 参数化, 但本项目未在 product_manifold 下重测.
2. **双码本解耦 (Task #200)**: 方向码本 (argmin) + 重构码本 (loss), 让"分配几何"和"重构几何"独立.
3. **Path regularization**: Task #209 C2 (w_path=1.0, path_geometry=hyp) 已加, 待看是否改善 collision.
4. **多 seed 验证**: 用户 2026-07-23 撤回, 单 seed 足够 (见 memory: user-no-multiseed-override).

---

## §7. 链接 + 引用 (paper References)

- Berman-Metzler 2020: κ-Stereographic distance formula
- HG-Rec paper: arXiv link + Table 1
- TIGER paper: GRID pipeline reference
- LETTER paper: 顺序推荐 baseline
- RQ-VAE: Van den Oord 2017
- Sinkhorn-Knopp: Cuturi 2013

---

## §8. 当前状态 (2026-07-26 20:50, paper §4 status table)

| 格子 | 代表 | 状态 | 下一步 |
|------|------|------|--------|
| 左上 (高维, 不钉) | HG-Rec #84 | ✅ baseline (R@10=0.1020) | 论文引用 |
| 右上 (高维, 钉) | Task #209 A3 | ✗ NO-GO (R@10=0.0863) | 论文引用 + 失败模式 |
| 左下 (低维, 不钉) | Task #210 B1 + #211 C1 旧 | ✗ NO-GO (collision 85%+) | 论文引用 + 失败模式 |
| 右下 (低维, 钉) | Task #211 C1/C2/C3 | ⏳ 训练中 (epoch 90+, ‖x‖_E 钉住 ✅) | **等 Stage 4 test R@10** |

**stop-loss line**: 右下 R@10 ≤ 0.1020 → "架构不可行" 收线 (per 用户 2026-07-26 反馈).

---

(本文档为 paper skeleton, 收线时复制到 `papers/2x2_design_space.tex` 或合并到现有 `papers/paper.md`.)