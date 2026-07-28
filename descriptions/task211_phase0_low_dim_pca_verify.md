# Task #211 — Phase 0 前置验证:低维双曲 + 钉半径可行性

> **创建日期**: 2026-07-26
> **状态**: ⏳ 进行中 (CPU 推理, 不占 GPU)
> **上游依据**: 用户 2026-07-26 提出新实验方案 (替代 task210 Phase B 的 product_manifold 路线)

---

## §1 目标

**在动训练之前,先确认"低维双曲 + 钉半径"在数值上站得住.** Phase 0 通过后才进入 Phase 1 训练.

## §2 Phase 0 做法 (纯 CPU 推理)

输入: `HG-Rec/dataset/Instruments/item_emb.parquet` (9922 items × 768 dim, sentence-t5-base embedding)

流程:
1. **PCA → d 维**: 把 latent 降到 d ∈ {2, 4, 8}
2. **球面 k-means**: init 方向为单位向量 (sklearn k-means on unit-normalized data), 钉到半径 ρ_ℓ
3. **分配**: 用真实 latent 做 argmin 距离分配

**三层各跑一遍** (ℓ ∈ {0, 1, 2}):
- ℓ=0: K=64, ρ=2.0
- ℓ=1: K=128, ρ=2.7
- ℓ=2: K=256, ρ=3.4

## §3 四个测度

| 指标 | 通过标准 | 对应哪次失败 |
|---|---|---|
| **动态范围**(最远/最近邻) | **≥ 2.0** | A3 栽在 1.27 |
| **max ‖x‖**(球内) | **≤ 0.95** | B1 栽在 1.000 |
| **用到的码字数 / K** | **≥ 90%** | A3 只用了 3 个 |
| 最小两两夹角 | 记录, 解释用 | — |

## §4 输出

写到 `verdicts/task211_phase0_verify.md` + `verdicts/task211_phase0_metrics.json`:
- 18 格子表 (3 d × 3 layer × 6 指标)
- 推荐 d_hyp (按 dyn ≥ 2.0 + util ≥ 90% 综合筛选)
- 关键 decision tree: 哪个 d 通过?

## §5 约束 (R11.3 + 用户)

- **不动训练代码** (Phase 0 不改 `train_hrqvae.py` / `hrqvae.py`)
- **不占 GPU** (CPU 推理, sklearn k-means + numpy)
- Phase 0 不过 → **不进 Phase 1** (用户明确)

## §6 Phase 1 预告 (Phase 0 通过才跑)

4 臂 × 1000 epoch × 4 GPU:
- C0: A0 baseline (#181, 复用)
- **C1: 低维双曲 + 钉半径** (核心: 没试过的格子)
- C2: C1 + path_reg hyp
- C3: C1 + path_reg euc control

**提前中止条件** (epoch 20 + 50 检查):
- `hyp_norm_max > 0.98` → abort (B1 失效)
- `dyn_range < 1.5` → abort (A3 失效)

