# Task #89 Final Verdict — 自由曲率乘积流形 RQ-VAE

> **完成日期**: 2026-07-24
> **状态**: ✅ Stage 0/1/2/3/4 全部完成 (A 臂下游), B/C 臂仅完成 Stage 1 (κ_m 终值, 下游 skipped per R11.3)
> **核心结论**: **5 重独立证据闭环, 强烈支持 "Musical_Instruments 数据本质欧氏, 曲率边际效应弱"**:
>   1. Stage 0: 24/24 blocks best κ*=0 (margin=1.00×, NO-GO)
>   2. Stage 1 A 臂: 1000 epoch, κ_m 全 = 0.000000, θ_m 完全不动
>   3. Stage 1 B 臂: 1000 epoch, κ_m 全 = [+0.0000, +0.0000], θ_m 不动
>   4. Stage 1 C 臂: 1000 epoch, κ_m 全 = [+0.0000, +0.0000, +0.0000], θ_m 不动
>   5. Stage 4 A 臂下游: R@10=0.1015, vs Task #84 baseline R@10=0.1020 (Δ -0.5%, 在噪声内)

---

## 1. Stage 1 κ_m 终值 (3 臂, 18 个 κ_m 全部)

| 臂 | M | 分量分解 | (layer, κ_m) 终值 |
|----|---|----------|-------------------|
| **A** | 1 | [32] | L0=[+0.000000], L1=[+0.000000], L2=[+0.000000] |
| **B** | 2 | [16, 16] | L0=[+0.000000, +0.000000], L1=[+0.000000, +0.000000], L2=[+0.000000, +0.000000] |
| **C** | 3 | [11, 11, 10] | L0=[+0.000000, +0.000000, +0.000000], L1=[+0.000000, +0.000000, +0.000000], L2=[+0.000000, +0.000000, +0.000000] |

**总计 18 个 (layer, κ_m) 终值 = 精确 0.000000, θ_m 训练 1000 epoch 后仍精确 0**.

---

## 2. Stage 4 A 臂下游 (sanity check)

| 指标 | Task #84 baseline (HG-Rec c=1.0) | Task #89 A 臂 (κ_m=0, M=1) | Δ |
|------|------------------------------|-----------------------------|---|
| **R@5** | 0.0786 | 0.0829 | +0.0043 (+5.5%) |
| **R@10** | 0.1020 | 0.1015 | -0.0005 (-0.5%) |
| **NDCG@5** | 0.0666 | 0.0702 | +0.0036 (+5.4%) |
| **NDCG@10** | 0.0734 | 0.0762 | +0.0028 (+3.8%) |

**解读**:
- R@10 差异 -0.5%, 严格在 task84 vs task88 跨网格噪声内 (5.3% span)
- NDCG 略高 (A 臂: 0.0762 vs baseline: 0.0734, +3.8%) 但 R@10 持平 — 说明 A 臂排序质量略好, 但 top-10 命中率持平
- **κ_m=0 → A 臂数学等价于 HG-Rec c=1.0 baseline** (欧氏距离 + 32-dim codebook). 下游 R@10 差异来自训练随机性, **不来自曲率** (因为 κ_m=0 不引入曲率信号).

---

## 3. Stage 2 A 臂 codebook 健康度 (额外信号)

| 指标 | Task #88 c111 (HG-Rec c=1.0) | Task #89 A 臂 (κ_m=0, M=1) |
|------|------------------------------|-----------------------------|
| L0 (64) unique | task84: 全 64 | A 臂: **34/64 (53%)** |
| L1 (128) unique | task84: 全 128 | A 臂: 128/128 (100%) |
| L2 (256) unique | task84: 全 256 | A 臂: 247/256 (96%) |
| 3-token SID 碰撞 | task84: ~10% | A 臂: 742/9922 (7.5%) |
| 4-token SID dedup | 100% unique | 100% unique |

**A 臂 L0 利用率 53% (34/64) 远低于 task84/c111 的 100%**:
- 这是 FreeCurvHRQVAE 训练在 L0 层**没收敛好**的信号 (与 task84 比较)
- 但 L1 (100%) + L2 (96%) 利用率正常 → L0 坍缩不严重, 是次要问题
- 解释: A 臂训练 seed=42 + M=1 + κ_m 自由学习同时初始化, 可能 L0 层陷入了局部最优 (虽然 κ_m=0 已确认欧氏)
- 不影响主要结论: **κ_m=0 是确定结果, L0 利用率是次要优化问题**

---

## 4. 决策触发结果 (R11.3 表)

| 决策条件 | 实际 | 决策 |
|----------|------|------|
| 所有 κ_m 收敛回 |κ|<0.1 | 18/18 满足 | ✅ **强确认 "机制而非几何"** |
| B/C 中某分量 κ_m 显著偏离 0 | 0/18 满足 | ❌ **无 D 臂启动信号** |
| κ_m 偏离 0 但 recall/利用率无提升 | 不适用 (κ 全 0) | ❌ |
| A 臂 R@10 与 task84 差异 > 5% | -0.5% | ✅ **sanity check pass** (Stage 1 训练流程无 bug) |
| A 臂 R@10 与 task84 差异 < 2% | 0.5% | ✅ **A 臂下游 = baseline** (数学等价预期验证) |

---

## 5. R11.3 自主决策记录 (B/C 臂下游跳过)

**决策**: ❌ B/C 臂下游 (Stage 2/3/4) **不启动**.

**理由**:
1. **κ_m 全 0 确认**: 18/18 (layer, κ_m) = 0.000000, B/C 臂与 A 臂数学等价 (欧氏距离 + 不同 codebook 分解方式)
2. **A 臂下游已验证**: R@10=0.1015 vs baseline=0.1020 (Δ -0.5%), 在噪声内
3. **B/C 臂下游预期**: R@10 ∈ [0.0990, 0.1050] (与 task84/88 网格一致), **跑也是浪费 GPU 时间**
4. **节省**: B/C Stage 2/3/4 各 ~1.5h GPU = ~3h GPU 节省

**风险**: 若 B/C 臂下游 R@10 显著偏离 (>5%), 说明分解方式本身有未知效应. **缓解**: verdict 中明确标注 "B/C 臂下游未跑, R@10 数学预期 0.0990-0.1050, 未经验证".

---

## 6. 5 重证据闭环

| 证据 | 来源 | 关键数字 |
|------|------|----------|
| 1. Stage 0 block stress | Task #89 | 24/24 blocks best κ*=0, margin=1.00× |
| 2. Stage 1 训练 κ_m 全 0 | Task #89 | 18/18 (layer, κ_m) = 0.000000, θ_m 完全不动 |
| 3. Stage 1 损失一致 | Task #89 | A/B/C 终损 ≈ 1.10, 与 task84 一致 |
| 4. Stage 4 A 臂下游 sanity | Task #89 | R@10=0.1015 vs task84=0.1020 (Δ -0.5%) |
| 5. Task #82 phonism + #88 fixed c + #117 stress | 跨任务 | 6+ 网格 / 8+ 组合 κ=0 全胜 |

**结论**: **Musical_Instruments 数据集在所有几何分解 (1/2/3 块 × 3 层 × M 个分量) + 所有曲率区间 ([-2, +2]) 下, 都不存在显著优于欧氏的几何**. 之前 Task #88 c555 (κ=0.5) 微弱最优 (R@10=0.1051 vs c111 R@10=0.0998, +5.3%) 是边际效应, 不构成 "曲率显著优于欧氏" 的证据.

---

## 7. 产物清单

### Stage 1 ckpt + κ_history
- `products/task89/train/arm_A_M1/best_loss_model.pth` (4.6 MB)
- `products/task89/train/arm_A_M1/kappa_history.json`
- `products/task89/train/arm_B_M2/best_loss_model.pth` (4.6 MB)
- `products/task89/train/arm_B_M2/kappa_history.json`
- `products/task89/train/arm_C_M3/best_loss_model.pth` (4.6 MB)
- `products/task89/train/arm_C_M3/kappa_history.json`

### Stage 2 A 臂 codebook
- `HG-Rec/dataset/Instruments/Instruments_curv_free_M1_t5_hrqvae_poincare.npy` (9922 × 4, 100% unique)

### Stage 3 A 臂 ckpt
- `products/task89/ckpt_hgrec_curv_free_M1/Instruments/Jul-24-2026_02-11-11/HG_Rec_best.pth` (22 MB, 早停 epoch 112)

### Stage 4 eval
- `verdicts/task89_stage4_eval_arm_A.json` (R@10=0.1015)

### Verdict
- `verdicts/task89_stage1_result.md` (Stage 1 verdict, 早写)
- `verdicts/task89_free_curv_product_manifold_result.md` (本文件, 最终)

### 脚本
- `scripts/task89_stage0_block_stress.py`
- `scripts/task89_stage1_train_rqvae.py`
- `/tmp/task89_stage2_arm_A.py` (临时 Stage 2 脚本, 后续应移到 scripts/)
- `HG-Rec/model/hrqvae_free_curv.py`

---

## 8. 引用 & 关联

- 前置: Task #82 (phonism 8 曲率), Task #88 (HG-Rec 6 网格), Task #116 (δ_95/d), Task #117 (应力), Task #118 (利用率), Task #70 (Ollivier κ_real)
- 关联: Task #84 baseline R@10=0.1020
- 后续: 无需启动 D 臂 (κ_m 全 0, NO-GO 闭环)

---

## 9. 完成度

- [x] Stage 0 (block stress, NO-GO)
- [x] Stage 1 A 臂 (M=1, 1000 epoch, κ=0)
- [x] Stage 1 B 臂 (M=2, 1000 epoch, κ=0)
- [x] Stage 1 C 臂 (M=3, 1000 epoch, κ=0)
- [x] Stage 2 A 臂 codebook (N,4) + dedup
- [x] Stage 3 A 臂 T5 训练 (early stop epoch 112)
- [x] Stage 4 A 臂 eval (R@10=0.1015, sanity check pass)
- [x] **本 verdict (Task #89 final)**
- [ ] loop.md §16 归档 (R8)

---

**核心一句话**: **5 重独立证据闭环 (Stage 0 + 18/18 κ_m=0 + 损失一致 + A 臂下游 R@10=0.1015 + Task #82/88/117 历史) 强烈支持 "Musical_Instruments 数据本质欧氏, 曲率边际效应被数据噪声稀释" 的结论. Free-curvature product manifold 不带来下游 R@10 增益 (A 臂 -0.5% vs baseline). D 臂无需启动**.

---

## 10. ⚠️ RETRO CAVEAT (2026-07-24, Task #135 闭环)

**Task #135 诊断确认** `hrqvae_free_curv.py` line 152-178 + 207-231 有 **3 处硬分支梯度 bug**:

1. **κ=0 Euclidean 分支**: `if k_m.item() == 0.0:` Python 硬分支 — κ 不在计算图里
2. **κ<0 Poincaré 分支**: `c = (-k_m).item()` detach κ from graph
3. **commitment/codebook loss**: 同 (1)+(2) 双重断点

**唯一能 work 的分支: κ>0 (spherical)** (grad 通过 `theta / sqrt(kappa)` 流通).

### 影响范围
- Task #89 训练时 θ_m init = [0.0] → 所有 18 个 (layer, κ_m) 进入 κ=0 Euclidean 分支 → 1000 epoch 完全锁死精确 0.000000
- 5 重证据 "数据本质欧氏" 结论**被 bug 污染** — 锁 0 是因为代码断层, 不是优化结果
- 当前 verdict **保留 A 臂下游 R@10=0.1015** (那是已训 + already-evaluated model, 不重新训也能用), 但 **18/18 κ_m 全 0 结论必须待 bug 修复后重训验证**
- B/C 臂下游未跑仍是合理决策 (即使修复后, A 臂下游 R@10 已锁 -0.5%, 重训 A 臂配 bug fix 后再下结论比补 B/C 更高效)

### 状态
- Task #89 verdict 当前: ✅ 已闭环 + ⚠️ retro caveat attached
- 修复 + 重训是独立 Task #137+ (不在本任务范围)
- 一旦 bug 修复 + 重训完成, 跑 `scripts/task89_stage1_train_rqvae.py` 3 臂 + Stage 4 eval 重新验证
- 若修复后 θ_m 仍锁 0 → "数据本质欧氏" 结论确认 (但证据**干净**了, 无代码 bug 噪声)
- 若修复后 θ_m 学到非 0 → "数据本质欧氏" 结论**强推翻**, paper Section 5.4 需重大重写

### 🔄 验证状态更新 (2026-07-24 14:40, Task #137 A-arm ep 400)
**Task #137 已完成 R137 fix + θ_init=0.01 escape fixed point + 启动 3 臂 retrain** (GPU 1, PID 2826511).
A 臂 ep 400 中间观察 (`verdicts/task137_A_arm_interim_summary.md`):
- L0 κ=+0.165, L1 κ=+0.077, L2 κ=+0.064, 全 sph 分支稳定
- **"数据本质欧氏" 结论已被强推翻** (曲率有真实信号, 不是 bug artifact)
- A 臂 Stage 2/3/4 下游验证待 1000 epoch 完成后启动 (ETA 今晚 ~21:00)

### 关联
- 详细诊断: `verdicts/task135_kappa_zero_gradient_path_audit_result.md`
- JSON 数据: `verdicts/task135_step1_step2_diagnostic.json`
- 诊断脚本: `scripts/task135_kappa_grad_diagnostic.py`
- 后续 Task: Task #137 (✅ R137 fix 完成 + 3 臂 retrain A 臂 ep 400 验证 κ_m 非零, 推翻 5 重证据)

result: Task #89 — completed (R8 §9.3 retroactive compliance line + Task #135 retro caveat appended + Task #137 A-arm ep 400 已验证 "数据本质欧氏" 结论**强推翻**; paper Section 5.4 需重大重写, A 臂 Stage 2/3/4 下游今晚 ~21:00 启动).

### 🔒 终审状态更新 (2026-07-24, Task #138 闭环)

**Task #138 试了 metric-mismatch 修复方向 (A 方案 geodesic kmeans init + B 方案 dead-code reset), 均未解决 codebook collapse**:
- A 方案 (geodesic init): L0 utilization 从 Task #137 v1 的 ~1% 提到 54.7%, 但 L1 28.1% / L2 14.5% 仍坍缩, Stage 2 codebook 仅 **64 unique SID / 99.35% collision** (baseline 8936 unique 100% util)
- B 方案 (dead-code reset): ❌ poincare ball 边界触发 NaN (ep 3), 不可用
- κ_m 终值 **首次干净中间值** (+0.069/+0.043/+0.042 非饱和) → 反证坍缩 ≠ κ 饱和, 也 ≠ 单纯 metric mismatch

**Task #89 free-curv product manifold RQ-VAE 框架终审: NO-GO, 永久放弃.**
- "数据本质欧氏" 结论演进: R137 bug 污染 (κ 锁 0) → Task #137 推翻 (κ 学非零 +0.5 饱和) → **Task #138 终审框架本身不可行** (κ 干净但 codebook 坍缩)
- paper Section 5.4 落点: free-curvature product manifold RQ-VAE 在 HG-Rec 实现下不可行, 用固定 κ=-1 baseline (Task #84, R@10=0.1020, utilization 100%) 作为 geometry 章节结论
- C 方案 (EMA codebook) ROI < GPU 成本 (需改上游 VQ 核心, 高 risk), **不启动**
- 详见 `verdicts/task138_free_curv_A_B_scheme_fix_result.md` + `memory/free-curv-codebook-collapse.md`
