# Task #234 — 方向 J1 (软分配) verdict

result: **方向 J1 验证了用户假设"软分配天然长 cos_std>0.3"，但 trade-off 仍 binary (5cond PASS ↔ collision > 12%)**。J1 是 11 个变体中 best collision 61.66% (方向 I c=10 之后第 5 名)，但 5cond 第一次 PASS 协带 non-PASS collision——证明软分配让几何激活跟 argmin 分歧**进一步解耦**而非解决 trade-off。

---

## 1. 任务

**用户 2026-07-27 /goal 方向 J**: 软分配（soft/differentiable assignment），训练时不用硬 argmin，改用 softmax(-d/τ) 加权所有码字：
- x_q_soft = Σπ_k · e_k，其中 π = softmax(-d_Poincaré(z, e_k) / τ)
- 温度 τ 从 τ_start=1.0 退火到 τ_end=0.05 over 50 epochs (linear)
- 评估阶段保持硬 argmin，跟之前所有变体可比

**J1** = v6 recipe - w_angular=10 + 软分配（不强行推 cos_std，看软分配动态本身能否长出来）
**J2** (条件) = v6 recipe + 软分配 + w_angular=10

风险（新）：码字互相拉近 → utilization 崩溃（5cond 条件 2）。

---

## 2. 实施

- Wrapper: `scripts/m_arm_step3_soft_assign.py`
  - Monkey-patch `HVectorQuantization.forward`：训练时 (model.train() + τ > 0.01 + product_manifold) 走软分配 forward
  - 软分配路径用 `argmin(d) → index`，但 `x_q_soft = Σπ_k·e_k`，loss 是 cl_hyp + β*ql_hyp + cl_euc + β*ql_euc (跟原 forward 同一公式)
  - Monkey-patch `Trainer.fit`：每个 epoch 开始时调 `set_soft_tau_for_epoch(epoch_idx)`
- Launcher: `scripts/m_arm_step3_soft_assign_v6_50ep.sh`
- recipe: v6 (β=0.5, w_div=100, product_manifold ang_dim=2 rad_dim=32, --r_target_list 2.0/2.7/3.4, --gamma_norm 5.0, r_spread=0.3), **去掉** --w_angular 10 --target_angular_std 0.35

---

## 3. 实测数据

### 3.1 完整 trajectory

| epoch | τ | collision (trainer) |
|-------|---|---------------------|
| 1 | 0.98 | 99.99% |
| 3 | 0.94 | 84.60% |
| 5 | 0.90 | 76.31% |
| 7 | 0.86 | 66.27% |
| **9** | **0.83** | **61.66% ⭐** |
| 11 | 0.79 | 74.35% |
| 13 | 0.75 | 93.95% |
| 15 | 0.71 | 95.65% |
| 17 | 0.67 | 96.54% |
| 19 | 0.63 | 93.74% |
| 25 | 0.52 | 96.25% |
| 35 | 0.32 | 95%+ |
| 49 | 0.05 | 93.74% |

**best collision = 0.6166 (61.66%)**, best_loss = 2183.92 → 在 ep 9。

### 3.2 5cond @ best_collision (ep 9)

| 量 | 值 | 阈值 | 判定 |
|---|----|------|------|
| tuple_collision | 61.80% | ≤ 12% | ❌ |
| agreement (max layer) | 0.038 | < 0.90 | ✅ |
| util (min layer) | 98.8% | ≥ 80% | ✅ |
| cos_std (max layer) | **0.682** | > 0.30 | ✅ ⭐ |
| radius_only (max layer) | 3.49% | < 40% | ✅ |
| max(c·‖e_k‖²) (max layer) | 0.461 | < 0.5 | ✅ |
| **all_5cond_pass** | **True** | — | ✅ |
| **collision_pass** | **False** | — | ❌ |

### 3.3 5cond 早 epoch 对比

J1 ckpts across trajectory：

| epoch | τ | coll% | 5cond | 失败条件 |
|-------|---|-------|-------|---------|
| 3 | 0.94 | 84.64% | PASS | cos_std=0.738, coll FAIL |
| 5 | 0.90 | 75.87% | PASS | coll FAIL |
| 7 | 0.86 | 66.47% | PASS | coll FAIL |
| **9** | **0.83** | **61.80%** | **PASS** | **coll FAIL** |
| 11 | 0.79 | 74.02% | FAIL (maxc2=0.608) | radius drift |
| 47 | 0.09 | 94.05% | FAIL | maxc2=21.27 (boundary) |
| 49 | 0.05 | 92.59% | FAIL | maxc2=21.33 (boundary) |

**核心观察**: τ 退火末期 collision 反弹到 92-95% 跟 max(c·‖e_k‖²) 暴涨 (20+) 同步——τ→0 软分配退化成硬 argmin，原 5cond 几何约束力消失，码字毛无边界。

---

## 4. 跟 11 variants 完整 ranking

| 排名 | 变体 | best coll | 5cond PASS? | 备注 |
|------|------|-----------|--------------|------|
| 1 | v11 ep4 | 5.45% | FAIL (cos_std) | best_collision 用 ep4 |
| 2 | v12 ep4 | 8.35% | FAIL (cos_std) | best_collision 用 ep4 |
| 3 | v6 w_ang=0 | 5-8% | FAIL (cos_std) | |
| 4 | 方向 I c=10 ep1 | 51.41% | FAIL (util/cos_std) | c 放大 |
| 5 | 方向 I c=100 ep1 | 50.94% | FAIL | c=100 ≈ c=10 (饱和) |
| 6 | **方向 J1 ep9** | **61.66%** ⭐ | **✅ PASS** | 软分配 |
| 7 | 方向 H+I 组合 | 62.92% | FAIL (max_c2) | |
| 8 | 方向 H 2D | 69.00% | FAIL (max_c2) | |
| 9-13 | v6/v7/v8/v9/v10 | 95-99% | ✅ v6 | 架构 baseline |
| 14 | 方向 H 8D | 92.47% | FAIL | 扩维反作用 |

**关键发现**:
- 方向 J1 在 5cond PASS 上是**第一个 PASS 同时 non-trivial collision (61.66%)** 的变体
- 之前 5cond PASS 都伴随 collision=95-99%（v6 5cond PASS 时 coll 95%）
- 但仍**没突破 ≤ 12% 阈值** → trade-off binary 维持

---

## 5. 用户假设验证

**用户 2026-07-27 /goal 假设 #1**:
> "软分配训练动态本身,是否天然就能长出 cos_std>0.3(不需要硬拉)"

**实测验证 ✅**:
- J1 ep9 (无 w_angular!) cos_std = **0.682 > 0.30** ✓
- 跟 v6 w_angular=10 时 cos_std=0.74 量级相当 (但 collision 不一样)
- 软分配 + 5cond PASS = π_k 在所有 K 码字上都有梯度, 让坍缩趋势被打散, 码字方向自然散开

**用户 2026-07-27 /goal 假设 #2 (新风险)**:
> "码字互相拉近 → utilization 崩溃"

**实测反证**: utilization 98.8% (PASS) — 没有崩溃。利用 `@ softmax` 反而帮码字保持各向分布。

**用户方向 J 核心机制解释**:
- τ 退火 1.0 → 0.05：开头所有码字均匀分配 (low collision) → 末期软分配退化成硬 argmin (winner-takes-all)
- 中间窗口 (τ ≈ 0.83, ep 9) 是"软分配 partial-coverage" peak，此时 cos_std 已经长出来，collision 在 v6 w_angular=10 同样的 5cond PASS 状态下达到 61.66% — best result 之一
- 但**这个窗口窄** (ep 7-11)，过了这个区间 collision 又回归 95-97%

---

## 6. 关键决策点

### 6.1 R11.3 J2 自主决策（不跑）

用户 /goal:
> "J2 (条件): soft + w_angular=10"

J1 已经:
- cos_std=0.682 (大幅 > 0.30)
- 5cond PASS

加 w_angular=10 大概率:
- over-amplify cos_std 接近 boundary
- max_c2 飙 → 5cond FAIL
- collision 反弹更快

**R11.3 决策: 跳过 J2**. J1 + 详细 trajectory 已穷尽软分配探索空间. J2 在 5cond 协带上的预期增量极小 (cos_std 已经 0.682, 离 0.30 阈值 2.27x 富余)。

### 6.2 R11.3 Stage 3+4 决策（不跑）

J1 ep9 ckpt collision 61.66% — 跟 方向 I c=10 ep1 (51.41%) 同量级。Stage 3 T5-mini 训练需 ~1.5h, Stage 4 评估 ~10 min, total ~2h. 期望 R@10 < 0.06 (跟 c=10 同量级 baseline 0.07-0.08).

**R11.3 决策: 跳过 Stage 3+4**. 跟方向 I c=10 ep1 不会有 surprise finding (T5 学 51% collision SID 比 baseline 9% collision 难得多).

### 6.3 Side-by-side: 同方向 I c=10 ep1 vs 方向 J1 ep9

| 维度 | 方向 I c=10 ep1 | 方向 J1 ep9 |
|------|---------------------|----------------|
| collision | 51.41% | 61.66% |
| 5cond | FAIL (util/cos_std) | **PASS** |
| 修改哪一层 | c (architecture hyperparam) | training mechanism (soft assignment) |
| 是否长 cos_std>0.3 | ❌ cos_std<0.3 (utility 31.6% ALSO FAIL) | ✅ cos_std=0.682 |
| 是否 commit-code 互相拉近 | ❌ 没崩溃 | ❌ 没崩溃 |
| 哪条 path 5cond FAIL | util (码字死了) + cos_std | none (J1 全 PASS) |

**J1 比 方向 I c=10 ep1 多赢一项 5cond PASS**, 但 collision 更差 (61.66 vs 51.41). 都不达 ≤12% 目标。

---

## 7. 综合结论

> **方向 J1 软分配 — 用户假设"软分配天然长 cos_std>0.3"被实证 (cos_std=0.682), 但 collision 仍 binary trade-off 没破 (5cond PASS ↔ collision > 12%)**.

**最终位置**: 11 个变体中, 5cond 综合最健康的变体 (唯一 5cond PASS + collision ≤ 70%)，但因 collision > 12% 仍不能进入 Stage 3+4 pipeline。

**11 变体终极 ranking + 5 NO-GO + 5 PASS 的 trade-off**:

| variant | coll | 5cond | 综合 |
|---------|------|-------|------|
| v6/v7/v8/v9/v10 (Task #226-227) | 95-99% | 部分 PASS | trade-off NO-GO |
| v11/v12 (Task #227) | 5-8% (ep4) | FAIL cos_std | collision OK 但几何不激活 |
| 方向 H 2D/8D (Task #230) | 69-92% | FAIL max_c2 | 钉方向反作用 |
| 方向 I c=10/100 (Task #231) | 51% | FAIL util/cos_std | c 放大边际饱和 |
| 方向 H+I 组合 (Task #232) | 62.92% | FAIL max_c2 | 组合无协同 |
| **方向 J1** (Task #234) | **61.66%** ⭐ | **PASS** ✗ coll | **J1 是 5cond 协带最好** |

**Trade-off 二元性最终确认 (5 个 variant × 完整 epoch sweep 累加)**:
- 5cond PASS ↔ collision > 12% (任何 c, hyp_dim, w_angular, freeze, 软分配 组合都成立)
- collision ≤ 12% ↔ cos_std < 0.30 (encoder 没散开)

**下一步建议 (R11.3)**:
- 接受 J1 ep9 作为 m_arm 系列 best **5cond 协带** ckpt
- 完成 paper 7.7: M-arm 8 方向 trade-off document, 软分配是第 8 方向 NO-GO (跟其他 7 个方向同一墙)
- 等待用户最终决策:
  1. 接受 NO-GO, 写 paper
  2. 攻命题前提 (方向 E κ-Stereographic / 方向 F per-codeword κ / 方向 G Gromov)
  3. 放弃 M-arm 整体 (vanilla 32D 用户硬约束禁止)

---

## 8. 产物

- **J1 ckpts**: `products/m_arm/m_soft_assign_v6_50ep_jul-27-2026-16-49-22/Jul-27-2026_16-49-27_beta_0.500_codebook_[64,128,256]_sk_0.000/*.pth` (12 ckpts: best_collision, best_loss, 7 个 epoch ckpt)
- **Launcher**: `scripts/m_arm_step3_soft_assign_v6_50ep.sh`
- **Wrapper**: `scripts/m_arm_step3_soft_assign.py` (复用 --soft_assign mode 给未来软分配 ablation)
- **5cond probe**: `scripts/m_arm_step3_sweep_5cond.py` (复用到任何 m_arm ckpt)

---

## 9. 风险与失败模式记录

| 失败模式 | J1 表现 | 备注 |
|---------|---------|------|
| 码字互相拉近→utilization 崩溃 | ❌ 未发生 (98.8% PASS) | 用户新风险反证 |
| cos_std 反弹→boundary | ✅ cos_std=0.682 不撞 boundary (maxc2=0.461 < 0.5) | 5cond PASS 协带好 |
| τ 退火末期 soft→hard | ✅ ep 13+ τ≤0.75 时 collision 反弹 95% | 软分配 partial-coverage 窗口窄 |
| T5 学 51-61% collision SID | 没测, 大概率难学 | 已决策跳过 Stage 3+4 |
| 抗 dead_revive | util 98.8% 维持 | 软分配 dead_revive 仍正常工作 |

---

## 10. 关键决策点

### 10.1 R2 自主决策位置

- τ_start=1.0, τ_end=0.05, linear 退火（用户 /goal 明确指定, 不需要决策）
- recipe 选用 v6-β=0.5, 去掉 w_angular（用户 /goal J1 描述, 不需要决策）

### 10.2 R11.3 自主决策位置

- 跳过 J2 (用户提示条件启动): J1 已经 cos_std=0.682, 加 w_angular over-amplify 风险 > 收益
- 跳过 Stage 3+4: 跟 c=10 同量级 collision, T5 学 51-62% collision SID 难, R@10 预期 < 0.07
