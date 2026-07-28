# Task #184 Result — Phase 1a Product Manifold (球面 × 双曲) NO-GO

> **任务目的**: 修复 Task #183 v2 暴露的"角向坍缩"问题——把码字显式分解为 (θ ∈ S^(d-1), r ∈ ℝ⁺) 两部分，argmin 同时考察两轴。

> **完成日期**: 2026-07-25
> **状态**: ⛔ **NO-GO (Phase 1a)** — Product manifold 让坍缩更严重
> **verdict**: Product manifold 方向失败。Stage 1 collision 95.48% (vs Task #183 v2 96.25%)，但 Stage 2 4-digit dedup 后只有 **50 unique 3-digit SIDs** (max collision=2167 items 共享同一 SID)，Stage 3 T5 训练因 vocab_size=1025 超界崩溃。

---

## 1. 背景与动机

Task #183 v2 软正则化只修复径向，未修复角向坍缩（最终 collision 96.25%）。Phase 1a 提出 product manifold 解耦码字 = (球面 θ, 径向 r)，期望:
- α·(1-cos θ_x, θ_cb) → 球面 anti-podal 区分
- β_radial·MSE(r_x, r_cb) → 径向欧氏距离
- argmin 同时考察两轴 → 角向坍缩不可能

---

## 2. 实验设计

**变量**: 
- `product_manifold=True, angular_dim=16, radial_dim=16, alpha=1.0, beta_radial=1.0`
- 沿用 Task #183 v2 软正则化: `rho_reg_weight=0.1, radii=[1.0, 1.345, 1.69]`

**保持不变**: 1000 epoch, batch=1024, lr=1e-3, sk_eps=[0,0,0], seed=42

---

## 3. 结果

### 3.1 Stage 1 训练（1000 epoch 完成, GPU 0, PID 3677433）

| 指标 | Task #184 v1 | Task #183 v2 | 评估 |
|------|------------:|------------:|------|
| Final collision_rate | **0.9548** | 0.9625 | 略好 0.77pp（统计噪声） |
| L0 ‖cb‖ (target=1.0) | 0.80 | 0.76 | 都未达目标 |
| L1 ‖cb‖ (target=1.345) | 0.87 | 0.87 | 一致 |
| L2 ‖cb‖ (target=1.69) | 0.92 | 0.93 | 一致 |
| fill ∈ [0.5, 0.9] | ✓ (L0=0.80, L1=0.87, L2=0.92) | ✓ | 同 |

**结论**: collision 几乎相同（95.48% vs 96.25%），product manifold 未能改善角向坍缩。

### 3.2 Stage 2 codebook inference (Sinkhorn + dedup)

| 指标 | Task #184 | Task #181 paper_fix |
|------|---------:|-------------------:|
| Initial collision (3-digit) | 99.50% | (低) |
| Sinkhorn iterations 达到 0 collision | 30 (max) | < 30 |
| Dedup column max | **2166** | 12 |
| **Unique 3-digit SIDs** | **50** | (高) |
| Max items per 3-digit SID | **2167** | < 13 |
| Total items | 9922 | 9922 |

**关键发现**: Product manifold 让 mode collapse 进一步恶化——99.5% 的项目坍缩到 50 个 3-digit SID，2167 个项目共享同一 SID。这比 Task #183 v2 更严重。

### 3.3 Stage 3 T5 训练 — 崩溃

**错误**: `RuntimeError: CUDA error: device-side assert triggered`
**根因**: Dedup column max=2166 > vocab_size=1025，T5 embedding layer 越界访问。

---

## 4. 诊断

**为什么 product manifold 让坍缩更严重?**

1. **Angular k-means init 失败**: L2 归一化后球面 k-means 把 64 个 angular 码字聚集到 ~5-7 个方向上（球面均匀分布的 k-means 初始化不稳定）
2. **alpha/beta_radial 不平衡**: α=1.0 让 angular 主导，angular 退化导致 argmin 直接退化
3. **径向 + 球面复合 argmin 仍可坍缩**: 球面方向相近时 (1-cos)≈0，整个距离退化为径向 MSE，跟单径向 MSE 等价

**根因结论**: Product manifold 假设"两轴独立可分"——但 init 阶段两轴就相互耦合了。修复 init 才是关键（k-means with proper spherical + radial init）。

---

## 5. 决策点

### 决策 1: 修复 dedup vocab 超界 + 重跑 Stage 3?
**选了**: 否。
**为什么**: 即便修好 dedup, 50 unique 3-digit SIDs 本身就让 T5 失去区分度。Stage 4 test R@10 必低于 0.1057 baseline。
**依据**: T5 在 50 个 unique SID 上根本无法学到细粒度 ranking（vocab 利用率 50/1025 = 4.9%）。

### 决策 2: pivot 到 Phase 1b (orthogonal product space) ?
**选了**: 否。
**为什么**: 当前用户指令是 Task #185（disable early stop resume），优先级高于新方向探索。Phase 1b 暂搁置。
**回退方案**: 后续若要继续 HG-Rec 几何方向，回退到 Phase 0.6 vanilla (Task #181, R@10=0.1057)。

---

## 6. 关键数字速查

```
result: Task #184 Phase 1a (product manifold) NO-GO — Stage 1 collision 95.48% (vs Task #183 v2 96.25%, 无显著改善), Stage 2 只有 50 unique 3-digit SIDs (max collision=2167 items/SID, 比 v2 更严重坍缩), Stage 3 CUDA assert 崩溃 (dedup col max=2166 > vocab_size=1025). 方向失败.
```

---

## 7. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | Phase 1a 实施 + 1000 epoch 训练完成 |
| 2026-07-25 | Stage 2 dedup 后 50 unique 3-digit SIDs → NO-GO |
| 2026-07-25 | Verdict + §16 清理 |
