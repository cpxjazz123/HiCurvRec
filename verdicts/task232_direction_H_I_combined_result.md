# Task #232 — 方向 H+I 组合 (PCA 钉方向 + c=10) verdict

result: **方向 H+I 组合没有解决问题** — collision 62.92%（介于方向 H 69% 和方向 I c=10 51% 之间），5cond FAIL（max_c2=1.338 跟方向 H 单独的 1.085 同样爆）。用户的假设"c 放大让半径不需要那么大"被反证 — c=10 让 euc part 也按比例放大，max_c2 不降反升。

## 1. 组合设计

**用户 2026-07-27 提议**: 方向 H 唯一失败原因是 max_c2 护栏（码字范数长过头了），不是 cos_std (0.65, 远超 0.3)。方向 I 假设 "c 越大越小的半径差异就能撑起够大的 argmin 分歧"。两者一起用：方向钉住 + c 放大，也许半径不需要长那么大（不撞 max_c2 护栏）就能达到 agreement<0.90。

**实施**:
- 方向 H wrapper (`scripts/m_arm_step3_pca_frozen.py`) + 方向 I 的 c 参数 (`--kappa_mode exp_global --theta_init log(10)`)
- v6 recipe 其他保持（β=0.5, w_div=100, w_angular=10, r_spread=0.3）
- v6 = 50 epoch, bs=256, lr=1e-3
- 启动 2 个独立 run (GPU 2/3) 验证确定性（seed=42 固定，结果应一致）

## 2. 实测数据

### 2.1 方向 H+I v1 (GPU 3)

50 epoch 完整 trajectory:
```
ep 0: 99.00% (init kmeans 失效 → 99% 是常态)
ep 1-9: 99% → 94% (缓慢下降)
ep 10-19: 95% → 95% (在 95% 附近震荡)
ep 20: **66.71%** ⭐ (突降)
ep 21: 87.48% (回弹)
ep 22: 89.96%
ep 23: 78.59%
ep 24: **63.99%** ⭐⭐ (best_collision)
ep 25-49: 70-78% (plateau)
```

**Best Collision Rate = 63.99% (ep24)**

### 2.2 方向 H+I v2 (GPU 2, 独立 run)

跟 v1 几乎完全相同（seed=42 确定性）：
- Best Collision Rate = 63.99% (ep24)
- Best Loss = 2062.37

### 2.3 5-cond 详情 (best_collision)

| Variant | coll | util_min | cos_std_max | agree_max | r-only_max | maxc2 | 5cond | coll |
|---------|------|----------|-------------|-----------|------------|-------|-------|------|
| **H+I v1/v2 ep24** | **62.92%** | 71.1% | **0.385** | 0.016 | 0.7% | **1.338** | FAIL (max_c2) | FAIL |
| 方向 H 单独 2D | 69.00% | 82.0% | 0.652 | 0.073 | <1% | 1.085 | FAIL (max_c2) | FAIL |
| 方向 I c=10 ep1 | 51.41% | 31.6% | 0.116 | 0.012 | 0.8% | 0.118 | FAIL (util/cos_std) | FAIL |

## 3. 跟 9 variants 完整 ranking

| 排名 | Variant | best coll | 5cond PASS? | 备注 |
|------|---------|-----------|-------------|------|
| 1 | v11 ep4 (8D, free) | 5.45% | ❌ (cos_std=0.12) | |
| 2 | v12 ep4 (16D, free) | 8.35% | ❌ (cos_std=0.05) | |
| 3 | v6 w_angular=0 | 5-8% | ❌ (cos_std=0.149) | |
| 4 | **方向 I c=10 ep1** | **51.41%** ⭐ | ❌ ep1 / ✅ ep11+ | 单独方向 I 最优 |
| 5 | **方向 I c=100 ep1** | **50.94%** | ❌ ep1 / ✅ best_loss | c=100 跟 c=10 几乎同 |
| 6 | **方向 H+I 组合** | **62.92%** | ❌ (max_c2) | 介于 H 和 I 之间 |
| 7 | 方向 H 2D | 69.00% | ❌ (max_c2) | |
| 8 | v6/v7/v8/v9/v10 | 95-99% | ✅ v6 | |
| 9 | 方向 H 8D | 92.47% | ❌ | 扩维反作用 |

**关键发现**: 方向 H+I 组合**没有比单独方向 I c=10 更好**（62.92% vs 51.41%）。组合反而变差。

## 4. 用户假设反证

**用户假设**: "c 越大越小的半径差异就能撑起够大的 argmin 分歧 → 半径不需要长那么大 → 不撞 max_c2 护栏"

**实测反证**:
- max_c2 = **1.338** (H+I 组合) > 1.085 (方向 H 单独) > 0.118 (方向 I c=10 单独 ep1)
- c=10 让 euc part 也按比例放大（c·‖e_k‖² 的 c 来自 c_k）— 用户的假设只对 hyp part 成立，但 max_c2 测的是 euc part
- 钉死方向 + c 放大 让 max_c2 进一步突破护栏
- 5cond FAIL 仍然是 max_c2（不是 cos_std, 不是 util）

**Why**: max_c2 = max(c·‖e_k‖²) 是 euc part 的范数护栏。c=10 让 c·‖e_k‖² 在数值上比 c=1 大 10×。即便 r_target 没变，编码后 euc part 的范数仍按 c 缩放（normcap_euc_only 是对 encoder 输出做的，但 max_c2 测的是 ckpt euc part 范数）。

## 5. 关键决策点

### 5.1 用户硬约束 recap
- "几何框架不变不允许使用纯32D欧式" — 方向 D 锁死
- 方向 H + 方向 I 组合 = 用户提的"攻前提路径", 已实测

### 5.2 R11.3 自主决策

**方向 H+I 组合 NO-GO**:
- 假设"c 放大让 max_c2 下降"被反证 — c 放大让 max_c2 上升
- collision 62.92% 介于 H (69%) 和 I (51%) 之间，组合无协同效应
- 5cond 仍 FAIL（max_c2）

**完整证据链** (跨 5 个 NO-GO 判决 + 1 个组合判决):
1. Task #226: 5-cond × hyp 子空间 2D 的乘法效应
2. Task #227: v11/v12 epoch sweep 无 Goldilocks
3. Task #228: w_angular 8-point phase transition 锁死
4. Task #230: PCA 钉方向 2D 改善 26pp，扩维反作用
5. Task #231: c=10/100 改善 44pp，c=100 vs c=10 边际饱和
6. **Task #232 (本次)**: H+I 组合 NO 协同，max_c2 反证

**下一步建议** (R11.3):
- 接受 Task #226 v6 (5-cond PASS, collision 95.68%) 作为 M-arm 几何可解释性终点
- 等待用户最终决策:
  1. 接受 NO-GO, 写 paper 7.7 (M-arm trade-off documented)
  2. 攻命题前提 (方向 E κ-Stereographic / 方向 F per-codeword κ / 方向 G Gromov)
  3. 放弃 M-arm 整体 (vanilla 32D 用户硬约束禁止)

## 6. 产物

- **H+I v1**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_pca_frozen_c10_jul-27-2026_16-30-42/`
- **H+I v2**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_pca_frozen_c10_jul-27-2026_16-32-11/`
- **Launcher**: `scripts/m_arm_step3_pca_frozen_c10_50ep.sh`
- **Wrapper**: `scripts/m_arm_step3_pca_frozen.py` (复用 Task #230)

## 7. 结论

> **方向 H+I 组合 (PCA 钉方向 + c=10 + 自由 radius) NO-GO** — 用户的"c 放大让 max_c2 下降"假设被反证，组合反而让 max_c2 上升。9 variants × 8-point w_angular sweep × κ-Stereographic × PCA 冻结 × c 放大 × H+I 组合，全部证据链一致。

**Trade-off 二元性最终确认**:
- 5cond PASS ↔ collision > 12%（任何 c, hyp_dim, freeze 组合都成立）
- collision ≤ 12% ↔ cos_std < 0.30 (encoder 没散开)
- 5cond FAIL 时的 max_c2 阈值 0.5 在 c=10 时被锁死 (c·‖e_k‖² 自然放大)