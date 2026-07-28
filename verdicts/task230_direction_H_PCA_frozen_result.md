# Task #230 — 方向 H (PCA 冻结 + 自由 radius) verdict

result: **方向 H 在 2D 上 collision 从 v6 的 95.68% → 69.00%（-26pp 改进），但在 8D 上变差（92.47%）；仍未到 ≤12% 目标。架构根因 (hyp 子空间 2D 容量 + 5-cond 锁死) 未根除，仅在方向 H 路径上有边际改进。**

## 1. 方向 H 设计

**用户 2026-07-27 假设**: collision 爆炸是因为 w_angular 把码字推到残差向量密度低的地方，跟真实数据分布打架。钉死方向到真实数据的 PCA 主方向后，radius 能自然分化但不冲撞。

**实施**（不修改 src/ 上游，wrapper patch）:
1. Monkey-patch `HVectorQuantization.init_emb`：hyp 部分用 SVD 取主方向（不足用 Gram-Schmidt 正交化补足），euc 部分保留 KMeans
2. 训练前 `embeddings.weight.requires_grad = False`（冻结方向）
3. `log_r.requires_grad = True`（保留 radius 可学）
4. v6 recipe 其他超参保持（β=0.5, w_div=100, w_angular=10, r_spread=0.3 等）

## 2. 实测数据（v6 recipe + 50 epoch）

### 2.1 方向 H 2D（hyp_dim=2, hyp/euc split = 2D/32D）

| ckpt | collision | agreement | util_min | cos_std_max | r-only_max | maxc2 | 5cond | coll |
|------|-----------|-----------|----------|-------------|------------|-------|-------|------|
| best_collision (ep49) | 69.00% | [0.017, 0.073, 0.000] | 82.0% | 0.652 | <1% | 1.085 | FAIL (max_c2) | FAIL |

**跟 v6 对比**:
- collision: 95.68% → 69.00%（**-26.68pp 改进**）
- cos_std: 0.74 → 0.65（略降，但仍在 [0.30, 0.40] + 区间）
- agreement: 1.2% → 7.3%（L1 升，更分化）
- r-only: 0.4% → <1%（保持）

**根因**: 2D 空间里 PCA 取 2 主方向 + 62/126/254 个 Gram-Schmidt 正交化方向必然退化（cos ≈ 1）。cos_max_avg=0.9958 反映码字几乎都"贴"上数据 → assignment 几乎总是按方向走（radius 不起作用）。dead_revive L2 unique=242/256（L2 还有 14 死码字）。

### 2.2 方向 H 8D（hyp_dim=8, hyp/euc split = 8D/26D）

| ckpt | collision | util_min | cos_std_max | maxc2 | 5cond | coll |
|------|-----------|----------|-------------|-------|-------|------|
| best_collision | **92.47%** | 24.2% | 0.203 | 0.741 | FAIL | FAIL |
| best_loss | 95.29% | 20.3% | 0.227 | 0.892 | FAIL | FAIL |

**扩维反而变差**: 8D hyp 子空间足够时，钉死方向让 encoder 学不到更分散的方向 → cos_std 卡在 0.20，util 跌到 24%，max_c2 0.74 爆。**结论：8D 上钉死方向 = 反作用**。

### 2.3 跨 9 variants 完整 ranking（best collision, 越低越好）

| 排名 | Variant | best coll | 5cond PASS? | 备注 |
|------|---------|-----------|-------------|------|
| 1 | v11 ep4 (8D, free) | 5.45% | ❌ (cos_std=0.12) | 唯一 < 10% |
| 2 | v12 ep4 (16D, free) | 8.35% | ❌ (cos_std=0.05) | 接近 12% |
| 3 | v6 w_angular=0 | 5-8% | ❌ (cos_std=0.149) | |
| 4 | **方向 I c=10 ep1** | **51.41%** | ❌ ep1 / ✅ ep11+ | 新最低 c 系列 |
| 5 | **方向 I c=100 ep1** | **50.94%** | ❌ ep1 / ✅ best_loss | c=100 vs c=10 几乎无差异 |
| 6 | **方向 H 2D** | **69.00%** | ❌ (max_c2) | 钉方向有边际改进 |
| 7 | v6/v7/v8/v9/v10 | 95-99% | ✅ v6 | baseline |
| 8 | 方向 H 8D | 92.47% | ❌ | 钉方向扩维反作用 |

**关键发现**:
1. **钉死方向仅在 2D 容量受限时有边际改进**（v6 95% → H 2D 69%），扩到 8D 时反而限制 encoder 自由度（util 跌到 24%）
2. **c=10 vs c=100 几乎相同**（51.41% vs 50.94%）—— 说明 √c·ρ≈0.54 离 saturation 已够，再放大 c 没边际效应
3. **唯一突破 12% 的路径仍是 v11/v12 早期 ep (5-8%)** — 但 5cond FAIL（cos_std < 0.3）

## 3. 5-cond + collision 二元 trade-off 再次确认

跨 9 variants 全部变体都满足:
- 5cond PASS ↔ collision > 12%（强耦合，无法独立调节）
- collision ≤ 12% ↔ cos_std < 0.30（encoder 还没散开）

**架构根因 (Task #227 已论证，本实验再次确认)**:
1. **product_manifold 拆分**: hyp 子空间维度 (2D/8D/16D) 远小于 baseline 32D，角带宽不够分 K=64
2. **5-cond 几何约束**: normcap 钉 ‖e‖=0.3, γ_norm 钉三层 norm, r_target 钉三层 r, w_angular 推 cos_std, r_spread=0.3 —— 全部 collision 不友好约束
3. **两者形成乘法效应**: 拆分让 hyp 子空间本来就没空间，约束进一步把 euc 也钉死

## 4. 关键决策点

### 4.1 用户硬约束 recap
- "几何框架不变不允许使用纯32D欧式" — 方向 D (vanilla 32D) 锁死
- "码字方向不再自由学习或强制打散" — 方向 H 已尝试
- "把 c 调大让同样幅度的 radius 差异被放大成更大的 argmin 分歧" — 方向 I 已尝试

### 4.2 决策建议（R11.3 自主决策）

**接受 NO-GO**:
- 方向 H 在 2D 上 collision 改善 26pp，但仍未达 12% 目标
- 方向 I c=10/c=100 改善 44pp，但仍 50%+
- 9 variants + 8-point w_angular sweep + κ-Stereographic 全部穷尽，trade-off 是 binary
- 唯一突破路径仍是 v11/v12 (8D/16D) 早期 ep，但 5cond FAIL

**建议下一步**:
1. 放弃 collision ≤ 12% 目标，接受 collision > 12% + 5cond PASS（Task #226 v6 已达成）作为几何可解释性终点
2. 或转 vanilla 32D baseline（用户硬约束禁止）
3. 或放弃 5-cond 几何约束（架构 owner 决策）

## 5. 产物

- **方向 H 2D**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_pca_frozen_jul-27-2026_16-18-00/`
- **方向 H 8D**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_pca_frozen_8d_jul-27-2026_16-22-25/`
- **方向 I c=10**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/`
- **方向 I c=100**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_c100_jul-27-2026_16-22-05/`
- **Wrapper**: `scripts/m_arm_step3_pca_frozen.py`（monkey-patch init_emb + freeze）
- **Launcher**: `scripts/m_arm_step3_pca_frozen_50ep.sh` / `scripts/m_arm_step3_pca_frozen_8d_50ep.sh` / `scripts/m_arm_step3_c10_c100_50ep.sh`

## 6. 结论

> **M-arm product_manifold 架构在 collision ≤ 12% AND 5cond PASS 联合目标上 NO-GO 三次确认** — 跨 9 variants × 8-point w_angular sweep × κ-Stereographic × PCA 冻结 × c 放大，全部证据链一致。
>
> 方向 H 在 2D 容量受限时给边际改进 (-26pp)，方向 I 把 c 推到 10/100 给最大改进 (-44pp)，但两者都未突破 50% floor。**架构根因不变**：hyp 子空间容量 + 5-cond 锁死。

**完整证据链** (跨 Task #226/#227/#228/#230 四个 NO-GO 判决):
1. 几何约束副作用: 5-cond × hyp 子空间 2D 的乘法效应 (Task #226 §3)
2. Epoch sweep NO-GO: v11/v12 跨 10 epochs 无 Goldilocks (Task #227 §8)
3. w_angular fine-grained NO-GO: 8-point 扫 phase transition 锁死 (Task #228)
4. distance formula NO-GO: κ-Stereographic 不能恢复 (Task #227 Phase 0)
5. **方向 H NO-GO (本次)**: PCA 钉方向 + 冻结仅 2D 上 -26pp，扩维反作用
6. **方向 I NO-GO (本次)**: c=10/100 仅 -44pp，c=100 跟 c=10 几乎相同（边际效应已饱和）

**下一步建议** (R11.3):
- 接受 Task #226 v6 (5-cond PASS, collision 95.68%) 作为几何可解释性研究终点
- collision > 12% 降级为次要目标
- 等待用户指示是否攻几何约束（5-cond 阈值）或放弃（M-arm 整体）