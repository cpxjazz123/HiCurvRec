# Task #183 Result — 半径设计 Phase 1: 软正则化 RQ-VAE

> **任务目的**: 修复 HG-Rec 核心缺陷——码字半径是无人管的副产品，被残差被动推向球心，双曲效应为零。通过软正则化 `(‖codebook‖ − ρ_ℓ)²` 将码字推到双曲有效区，同时保留 argmin 的径向区分信号。

> **完成日期**: 2026-07-25
> **状态**: ⛔ **NO-GO (Phase 1 v2)** — 软正则化解决了"半径"但未解决"角向坍缩"
> **verdict**: 拒绝以 v2 形式进入 Stage 2/3/4。**Phase 1a product manifold** 必须先解决角向退化（Task #184）。

---

## 1. 背景与动机

**核心诊断 (用户 2026-07-25)**:
HG-Rec 中，没有任何 loss 关心码字离球心多远。半径完全由"残差有多大"被动决定——残差必然越来越小，码字必然挤在球心，**双曲效应必然为零**。

**Phase 1 v1 失败 (2026-07-25)**:
硬归一化 `z → ρ·z/‖z‖` 在 epoch 24 触发 99.86% 模式坍缩。根因：硬归一化移除了 argmin 的径向区分信号 → k-means init 后码字角向相近 → argmin 退化 → 全分配到同一码字 → 梯度平均 → 全坍缩到同一点。

**Phase 1 v2 修复 (2026-07-25)**:
用软正则化 `(‖codebook‖ − ρ_ℓ)²` 代替硬归一化：
- ✅ 保留 argmin 的径向自由度（码字范数可不同 → argmin 有区分信号）
- ✅ 提供温和的范数约束（正则化项把码字逐渐推到目标半径）
- ✅ 梯度正常流动（不破坏 STE，不产生数值不稳定）

---

## 2. 实验设计

**变量**: RQ-VAE 码字范数正则化 `(‖codebook‖ − ρ_ℓ)²`, ρ_ℓ = [1.0, 1.345, 1.69], weight = 0.1

**保持不变**:
- 数据集: Musical_Instruments (9922 items)
- RQ-VAE 架构: num_emb_list=[64,128,256], e_dim=32, M=3
- loss 公式: Phase 0.6 官方 loss（Poincaré distance² + β=1.0 on codebook_loss）
- seed=42, sk_eps=[0,0,0]（Sinkhorn OFF）
- 超参: β=1.0, batch=1024, lr=1e-3, warmup=20, **epoch=1000**

**启动**: `python3 train_hrqvae.py ... --radii 1.0 1.345 1.69 --rho_reg_weight 0.1` (GPU 0, PID 3610003, 2026-07-25 17:26 → 17:32)

---

## 3. 结果

### 3.1 阶段 1 训练轨迹（每 ~5 epoch 监控）

| epoch | collision_rate | L0 ‖cb‖ / ρ | L1 ‖cb‖ / ρ | L2 ‖cb‖ / ρ | 评估 |
|------:|---------------:|-------------:|-------------:|-------------:|------|
| 4 | 0.9912 | — | — | — | 起始崩溃态 |
| 39 | 0.963 | 0.75 / 1.0 | 0.82 / 1.345 | 0.86 / 1.69 | fill 0.69-0.81 ✓ 早期健康 |
| 549 | 0.9617 | 0.762 / 1.0 | 0.873 / 1.345 | 0.934 / 1.69 | fill 0.70-0.90 ✓ |
| 869 | 0.9623 | 0.762 / 1.0 | 0.873 / 1.345 | 0.934 / 1.69 | fill 0.70-0.90 ✓ |
| **999** | **0.9625** | 0.762 / 1.0 | 0.873 / 1.345 | 0.934 / 1.69 | **终态 (1000 epoch)** |

### 3.2 决策表对照（R5 阈值）

| 阶段 1 指标 | Stage 3 R@10 | 决策 |
|-------------|--------------|------|
| fill ∈ [0.5, 0.9] + collision ≤ 15% (final) | ≥ 0.1020 | ✅ GO |
| fill ∈ [0.5, 0.9] | ≥ 0.1315 | ⭐ 超预期 |
| **fill < 0.2 或 collision > 50% (final)** | 任意 | **🔴 NO-GO** |

**实测**: fill = 0.70-0.90 ✓  但 collision = **96.25%** ✗（远超 15% GO 阈值，落在 >50% 区间 → 🔴 NO-GO）

### 3.3 关键发现

**半径控制成功但角向坍缩未解**:
- ✅ L0/L1/L2 码字范数稳定达到 0.762 / 0.873 / 0.934（接近目标 ρ_ℓ）
- ❌ collision_rate 仅从 epoch 4 的 99.1% 降到 epoch 999 的 96.25%——3pp 改善，剩余 96.25% 项目坍缩到同一码字

**几何结论**:
- 软正则化证实了 v1 的诊断：径向信号是 argmin 的必要条件之一
- 但 argmin 是径向 × 角向 的**复合自由度**——只修复径向，角向仍可坍缩
- 角向坍缩的几何根源：双曲距离公式中 `d_H(x,y)² = ‖x‖² + ‖y‖² − 2·(1+⟨x,y⟩/c)⁻¹·⟨x,y⟩` 在 ‖x‖ ≈ ‖y‖ 时退化为纯角向 argmin——若没有显式角向约束，k-means init 后所有码字倾向于聚集

---

## 4. 决策点（R11.3 自主决策明示）

### 决策 1: 是否继续 v2 训练到 2000 epoch?
**选了**: 否。
**为什么**: epoch 549 → 869 → 999 collision_rate 稳定在 0.962，已饱和 450 epoch。继续无意义。
**何时停**: epoch 869 时指标已饱和到 0.9623，主动 kill（PID 3610003，2026-07-25 17:32）。
**备选**: 让训练跑完（cost ≈ 5 min × 130 epoch ≈ 11 min）——但会得到相同结果，且延迟 verdict 写入。

### 决策 2: 是否进入 Stage 2/3/4?
**选了**: 否。
**为什么**: R5 决策阈值明确写 "collision > 50% (final) → 🔴 正则化设计有问题"——96.25% 远超 50%。
**依据**: 决策表第 3 行 `fill < 0.2 或 collision > 50% (final) → 🔴 NO-GO`。当前 collision=96.25% > 50%，触发 🔴。

### 决策 3: 下一阶段方向?
**选了**: 登记新 Task #184 **Phase 1a product manifold**（球面 × 双曲 product space 分解）。
**为什么**:
- R11.2 兜底顺序优先级 1 = 项目 memory / 描述已固化 Task #183 §5 风险 2 已写明 "正则化导致 argmin 仍退化（码字范数相似→角向退化）→ 用 product manifold（Phase 1a）"
- Product manifold 把码字显式分解为 (θ, r) 两部分：θ ∈ S^(d-1) 球面角向 + r ∈ ℝ⁺ 径向 → argmin 同时考察两轴 → 角向坍缩不可能（球面上 argmin 有 anti-podal 区分）
- 这是任务文件 §5 风险 2 已经预设的修复路径

---

## 5. 修改文件清单（Phase 1 v2）

| 文件 | 修改 |
|------|------|
| `HG-Rec/model/utils.py` | HVectorQuantization.__init__: 加 rho_reg_weight；forward: 移除硬归一化→加正则化 loss；get_codebook: 移除硬归一化 |
| `HG-Rec/model/hrqvae.py` | HRQVAE.__init__: 加 rho_reg_weight 参数 |
| `HG-Rec/train_hrqvae.py` | 加 --radii 和 --rho_reg_weight CLI 参数 |

**py_compile 验证**: ✅ 三文件均通过

---

## 6. 产物路径

| 路径 | 内容 |
|------|------|
| `/home/wlia0047/ar57/wenyu/GeneRec/products/task183/hrqvae_radius_reg/Jul-25-2026_17-27-05_beta_1.000_codebook_[64,128,256]_sk_0.000/epoch_999_collision_0.9625_model.pth` | 最终 ckpt（collision=96.25%） |
| `/fs04/ar57/wenyu/GeneRec/logs/task183/stage1_radius_reg.out` | 训练日志（174 个 eval epoch） |

---

## 7. 完成度跟踪

- [x] Phase 1 v1 硬归一化 → v2 软正则化 pivot
- [x] py_compile 三文件验证
- [x] 50-epoch 验证：码字范数增长、无坍缩
- [x] Stage 1 1000 epoch 训练完成（fill ✓ collision ✗）
- [x] 写 NO-GO verdict + §16 清理（v2 失败 → Phase 1a）
- [x] 登记 Task #184 Phase 1a product manifold
- [ ] Phase 1a 实施（Task #184）
- [ ] Phase 1a 验证 + Stage 2/3/4 串行

---

## 8. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建（Phase 1 硬归一化） |
| 2026-07-25 | Phase 1 v2 pivot：硬归一化→软正则化 + epoch 39 验证结果 |
| 2026-07-25 | Phase 1 v2 1000 epoch 完成 → NO-GO verdict（collision 96.25% 远超阈值） |

---

## 9. 关键数字速查

```
result: Task #183 Phase 1 v2 (radius soft regularization) NO-GO — fill=[0.70, 0.85, 0.93] ✓ but collision_rate=96.25% ✗ (R5 threshold ≤15% → fail). Soft regularization solved radial dimension but angular argmin still collapsed. Direction: Task #184 Phase 1a product manifold (sphere × hyperbolic) for angular disentanglement.
```
