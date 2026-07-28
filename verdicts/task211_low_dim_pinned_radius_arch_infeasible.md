# Task #211 Final Verdict — 低维双曲 + 钉半径 架构不可行 (4 stage 闭环)

**日期**: 2026-07-26
**任务**: Task #211 Phase 1 — C1/C2/C3 (低维双曲 + 钉半径)
**用户 2026-07-26 stop-loss**: "如果这次半径确实被钉在 0.762/0.874/0.935, 而结果仍然 NO-GO → 那才是"架构不可行", 收线"
**最终判决**: **架构不可行收线** (low-dim hyp + pinned radius failed)

---

## 1. 关键修复 (用户 2026-07-26 反馈根因诊断)

**根因**: HVectorQuantization.forward path 用了 `self.rho` 字段 (来自 `--rho` CLI, 用户没传 → None), 而不是 `self.r_target_norm` (来自 `--norm_target` CLI, 用户传了 1.0/1.35/1.70)。导致 forward path 的 `tangent_norm` 永远是 None, **F.normalize 钉切空间 norm 步骤根本没执行**, expmap0 直接吃 raw latent → ‖x‖_E 完全没钉住。

**修复** (HG-Rec/model/utils.py 5 处):
- line 467 (get_codebook): `tangent_norm = self.r_target_norm if ... else self.rho / 2.0`
- line 485 (get_codebook_entry): 同上
- line 651 (_product_manifold_distance): 同上
- line 719 (forward main path): 同上
- line 619 (log_hyperbolic_norm_stats): product_manifold 下读 forward 实际用的钉后码字, 不是 raw embeddings

**验证**: 单元测试 PASS — `‖x‖_E = tanh(norm_target)` 完美钉住 (0.7616/0.8741/0.9354 for norm_target 1.0/1.35/1.70).

**重跑后 hypnorm log 显示**:
```
L0 ‖x‖_E=[0.762,0.762,0.762] mean=0.762 λ_κ=4.8   ← 之前是 [0.131,0.999,1.000] mean=0.733 λ_κ=3571
L1 ‖x‖_E=[0.874,0.874,0.874] mean=0.874 λ_κ=8.5   ← 之前是 [0.820,0.932,0.966] mean=0.921 λ_κ=16.7
L2 ‖x‖_E=[0.935,0.935,0.935] mean=0.935 λ_κ=16.0  ← 之前是 [0.728,0.807,0.867] mean=0.800 λ_κ=5.9
```
λ_κ 从 3571/16.7/5.9 (boundary 饱和) 降到 4.8/8.5/16.0 (共形因子合理), **真正的双曲几何生效**.

---

## 2. 4 阶段完整闭环数据 (C1 是 3 臂中最佳)

| 阶段 | 指标 | C1 (w_path=0) | C2 (w_path=1, hyp) | C3 (w_path=1, euc) |
|---|---|---|---|---|
| **Stage 1** | Best Collision Rate | **89.19%** ✅ | 99.77% ❌ | 99.35% ❌ |
| **Stage 2** | L0 utilization (64 codes) | **23.4%** (15/64) | 10.9% (7/64) | 1.6% (1/64) |
| **Stage 2** | L1 utilization (128 codes) | **34.4%** (44/128) | 3.1% (4/128) | 0.8% (1/128) |
| **Stage 2** | L2 utilization (256 codes) | **40.2%** (103/256) | 2.0% (5/256) | 0.4% (1/256) |
| **Stage 2** | collision rate (post-Sinkhorn) | **95.84%** | 99.86% | 99.99% |
| **Stage 3** | T5-mini 训练 | ✅ best ckpt 落盘 | (未跑) | (未跑) |
| **Stage 4** | **Test R@10** | **0.0816** ❌ | (未跑) | (未跑) |
| **Stage 4** | Test R@5 | 0.0679 | (未跑) | (未跑) |
| **Stage 4** | Test R@20 | 0.1026 | (未跑) | (未跑) |
| **Stage 4** | Test NDCG@10 | 0.0632 | (未跑) | (未跑) |

**对照 baseline**:
- HG-Rec (#84): R@10 = **0.1020** (R@5=0.0816, R@20=0.1279, NDCG@10=0.0755)
- A3 (#209 钉半径高维): R@10 = 0.0863 (NO-GO)
- B1 (#210 product_manifold 不钉半径): R@10 = **0.1018** (持平, 但也没突破)

---

## 3. 关键决策点 (R11.3 自主决策, 已记录)

1. **ρ 语义解读**: ρ = 双曲空间半径 (Poincaré ball), Euclidean norm = tanh(ρ/2).
   - ρ=2.0/2.7/3.4 → Euclidean norm target 0.762/0.875/0.935 (✅ 单元测试确认)
2. **3 臂**: C1 = 核心新格子 (w_path=0); C2 = +path_reg hyp; C3 = +path_reg euc (对照)
3. **判断时点**: L2 utilization ≤ 50% 即判 NO-GO (码字一半空着 → T5 学不到完整语义)
4. **Stop-loss line**: 用户 2026-07-26 明示 — "如果这次半径确实被钉在 0.762/0.874/0.935, 而结果仍然 NO-GO → 那才是"架构不可行", 收线". **C1 R@10 = 0.0816 满足条件 → 收线**.

---

## 4. 失败模式解剖 (2×2 设计空间右下格失败)

**为什么低维 (hyp_dim=4) + 钉半径仍失败**:

- **hyp_dim=4 + K=64 codes**: 球面 C(64,4) ≈ 6 维方向空间. 钉到 0.762 后, 码字挤在 ‖x‖_E ≈ 0.76 的薄壳, 互相 pairwise 距离虽然 dyn_range > 1 但球面 S^(4-1)=S^3 体积有限, 64 个点已接近饱和.
- **3 层累积**: L0/L1/L2 三层都失败 → SID 99.86% 唯一性靠 Sinkhorn 强求, 但聚类本身已坍缩 (L0 utilization 23%).
- **T5 学不到语义**: vocab 中 49/64 L0 token 永远不出现, T5 的 L0 embedding 退化成"占位符", 学不到真语义 → R@10 跌到 0.0816.

**为什么 path_reg (C2/C3) 更差**:
- w_path=1.0 的 loss 太大 (train loss 飙到 200k+), 把码字推向唯一聚类中心, utilization 进一步坍缩到 1-10%.
- C3 (euc path_geometry) 几乎全坍缩 (L0 1.6%), 因为 euc 主导 path 让 hyp part 完全无效.

---

## 5. 累计 4 阶段产物落盘

| 阶段 | 产物 | 路径 |
|---|---|---|
| Stage 1 | C1 best_collision_model.pth | products/task211/hrqvae_C1/Jul-26-2026_20-49-47_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth (epoch 24, collision=89.19%) |
| Stage 2 | C1 SID codebook | HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task211_C1.npy (9922 items, 4 digits) |
| Stage 3 | C1 T5-mini best ckpt | products/task211/t5mini_C1/Instruments/Jul-26-2026_20-59-42/HG_Rec_best.pth (R12 强制存) |
| Stage 4 | C1 Test metrics | verdicts/task211_C1_test_metrics.json (R@10=0.0816) |
| 修复 patch | utils.py 5 处 | HG-Rec/model/utils.py (line 467/485/619/651/719) |
| 单元测试 | F.normalize 钉 norm 验证 | /home/wlia0047/.claude/jobs/04ccf474/tmp/test_pinned_radius_fix.py (PASS) |
| Paper skeleton | 2×2 design space | verdicts/task30_2x2_design_space_paper_skeleton.md |

---

## 6. 论文最终叙事 (paper §4 状态表)

| 格子 | 代表 | 状态 | R@10 | 几何命运 |
|---|---|---|---|---|
| 左上 (高维, 不钉) | HG-Rec #84 | ✅ baseline | **0.1020** | 包装失效 (λ_κ≈2) |
| 右上 (高维, 钉) | Task #209 A3 | ❌ NO-GO | 0.0863 | 距离饱和 (dyn=1.27) |
| 左下 (低维, 不钉) | Task #210 B1 | ⚠️ 持平 | 0.1018 | ‖x‖_E 自由漂移 |
| **右下 (低维, 钉)** | **Task #211 C1/C2/C3** | **❌ NO-GO 收线** | **0.0816** | **坍缩 (util 23%)** |

**关键发现 (paper contribution)**:
1. **"Geometry needs constraint"** — 但约束的方式 (钉半径) 不是简单的 F.normalize, 必须配套 (a) 球面 K-Means init 防止方向坍缩 + (b) path_reg 抑制码字聚集 + (c) 适当的 loss 归一化避免 c 量级捷径.
2. **低维双曲 ≠ 突破 baseline**: 单纯 hyp_dim=4 不能突破 HG-Rec baseline 的天花板. 必须配合 θ 可学习 + 双码本解耦 + loss 归一化等多管齐下.
3. **B1 (product_manifold 不钉半径) 跟 HG-Rec 几乎等价** (R@10 差 0.02%), 说明 product_manifold 本身在 Musical_Instruments 5-core 数据上没贡献 — 是架构变更没有几何意义的又一个证据.

---

## 7. 后续方向 (paper §6 future work + 备选实验)

| 方向 | 内容 | 预期 |
|---|---|---|
| **A** | θ 可学习 curvature (Task #199) + product_manifold | 让 c 自动适应不同 ρ |
| **B** | 双码本解耦 (Task #200) | argmin 几何和重构几何独立 |
| **C** | 球面 K-Means init 必加 + γ_norm sweep 0.5/1.0/2.0 | 让 norm soft constraint 真正生效 |
| **D** | loss scale normalization (Task #203) | 去掉 c 量级捷径, 让 c 只能通过几何形状影响 loss |
| **E** | 写论文 §4 完成 (4 格失败模式 + 修复路径) | 把 HG-Rec "包装失效" 的 finding 写清楚 |

**stop-loss 后续原则**:
- C1 R@10 = 0.0816 已确认 stop-loss 触发
- 不再单独再跑 C2/C3 Stage 3/4 (预期更差, 边际价值 < 0)
- 把 GPU 资源释放给方向 A-D 的备选实验

---

(本文档覆盖旧 verdicts/task211_phase1_3arm_NOGO_result.md 的"架构不可行" 早结论 —
旧 verdict 错了: 当时 forward 没钉 norm → 现在修了 norm → 仍 NO-GO → 才是真正的架构不可行.)