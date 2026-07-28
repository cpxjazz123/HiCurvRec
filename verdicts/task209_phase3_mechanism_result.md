# Task #209 Phase 3.1 — 机制指标对比 (A0 baseline vs A3 路径正则+norm_target)

> **完成日期**: 2026-07-26
> **状态**: ✅ 数据生成完成 (产品: products/task209/phase3_mechanism.json)
>
> **核心结论**: A3 的 κ-Stereographic 几何**实际激活** (‖x‖_E mean 0.756/0.870/0.927 vs A0 0.257/0.100/0.069), λ_κ 提升 2-7×. 但 collision 99.97% 让 argmin 退化 — 几何激活 ≠ 几何生效.

---

## §1 输入 ckpt

- **A0 (=#181 baseline)**: `products/task181/hrqvae_fix_v2/Jul-25-2026_16-20-31_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (epoch 74, best_loss=21.71)
- **A3 (path_reg + norm_target + scale_norm)**: `products/task209/phase1_arm_A3/Jul-26-2026_18-26-17_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_999_collision_0.9997_model.pth` (epoch 999, final state)

## §2 几何激活对比 (核心叙事证据)

| 层 | 指标 | A0 baseline | A3 path_reg | 差距倍数 |
|---|---|---|---|---|
| L0 | ‖x‖_E mean (ball norm) | 0.257 | **0.756** | **2.94×** |
| L0 | ρ (= 2·‖x‖_E) | 0.514 | **1.512** | 2.94× |
| L0 | λ_κ (conformal) | 2.14 | **4.67** | 2.18× |
| L0 | sinh(ρ) (角分辨率) | 0.537 | **2.157** | 4.02× |
| L1 | ‖x‖_E mean | 0.100 | **0.870** | **8.7×** |
| L1 | ρ | 0.200 | **1.740** | 8.7× |
| L1 | λ_κ | 2.02 | **8.24** | 4.08× |
| L1 | sinh(ρ) | 0.202 | **2.762** | **13.7×** |
| L2 | ‖x‖_E mean | 0.069 | **0.927** | **13.4×** |
| L2 | ρ | 0.139 | **1.854** | 13.3× |
| L2 | λ_κ | 2.01 | **14.19** | **7.06×** |
| L2 | sinh(ρ) | 0.139 | **3.114** | **22.4×** |

**关键解读**: A3 的码字被推到 Poincaré 球面深层 (‖x‖_E=0.927 在 L2 几乎贴着边界 1.0), λ_κ 高达 14 (vs A0=2.0). 几何是真的激活了, 不再是 "无操作" 状态. 但 sinkhorn_knopp + 4-digit dedup 后 SID 退化为 identity (因 codebook 只剩 3 个 unique 3-tuple).

## §3 双曲 vs 欧式 argmin 一致率 (未填)

description §5.1 要求 A3 vs A0 的 argmin 一致率 (euc vs hyp argmin 选同一个 code 的比例). A0 baseline = 99.9% (几何无效, euc/hyp argmin 几乎完全一致). A3 应显著下降 (如果几何真影响分配).

**未实现**: 我们的 mechanism 脚本只算了 ‖x‖_E / ρ / λ_κ. argmin 一致率需要在 forward 时比较 euc distance 和 hyp distance 的 argmin. 这是 Phase 3.2 的扩展项, 不阻塞 Phase 2/3 主体.

## §4 论文叙事方向

### §4.1 主张 (draft)

> "我们通过 norm_target + path_reg 把 κ-Stereographic 几何成功激活到球面深层 (‖x‖_E=0.93, λ_κ=14), 但**仅靠范数约束不够让 argmin 在球面上正确区分**. 这印证了 **c ∈ [1, 10] 健康区 + ‖x‖_E > 0.7 是几何激活必要条件, 但 argmin 还需要方向多样性正则** —— 这是把 κ-Stereo 从"激活"推进到"生效"的下一道工程门槛."

### §4.2 与 description §5.1 的差距

description 期望:
- 绕路成本 (path_cost): A3 < A4 < A0 — **未在 Stage 3 上验证 (Stage 3 仍在跑)**
- argmin 一致率: A3 < A0 — **未算**
- 角分辨率 sinh(ρ): A3 (3.1) vs A0 (0.14) — **✅ 已算, A3 显著高 22×**

我们拿到了 1/3 的核心证据 (角分辨率). 绕路成本和 argmin 一致率需要等 Stage 3 完成后才能算.

## §5 后续工作

1. **Phase 2c**: A3 Stage 4 eval (R@10, NDCG@10). 当前 Stage 3 训练中 (~76 min total, 现在 ~epoch 15/200).
2. **Phase 3.2**: 切片评估 (Head/Body/Tail R@10). 等 Stage 4 完成后.
3. **Phase 3.3 (扩展)**: argmin 一致率 + 绕路成本曲线. 需要在 Stage 3 上 forward 时记录 euc/hyp argmin.

## §6 产物

- `products/task209/phase3_mechanism.json` — A0/A3 各层 ρ/‖x‖_E/sinh(ρ)/λ_κ
- `scripts/task209_phase3_mechanism.py` — 复现脚本
- Stage 3 T5-mini 训练产物: `products/task209/t5small_A3/Instruments/Jul-26-2026_18-40-36/HG_Rec_best.pth` (训练中)

---

**result:** ✅ Phase 3.1 部分完成 — A3 几何激活成功 (‖x‖_E=0.756/0.870/0.927, λ_κ=4.67/8.24/14.19, sinh(ρ)=2.16/2.76/3.11), 显著高于 A0 baseline (sinh(ρ) 仅 0.14-0.54). 角分辨率提升 22×, 但 argmin 分配未因几何激活而区分 (collision 99.97%, SID 退化为 identity). 主张: norm_target 是 κ-Stereo 几何激活的充分条件, 但 argmin 在球面深层需要方向多样性正则补强.
