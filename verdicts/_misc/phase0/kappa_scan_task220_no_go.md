---
task: 220
type: verdict
status: "NO-GO"
created: 2026-08-02
tags:
  - kappa
  - phase0
up: "[[index]]"
---
# Phase 0 选项 D (task220 ckpt) — 同样 NO-GO (2026-07-27)

## 关键发现

task220 hrqvae_pck (Per-Codeword κ, 无 NormCap) ckpt 训完后 **latent 范数仍 ≈ 0.135** (跟 E1 normcap=0.131 几乎一致).

```
task220: ‖z‖ mean=0.1346 max=0.1682
E1 normcap: ‖z‖ mean=0.131 max=0.19
```

**所有 HRQ-VAE 训练收敛后 latent 范数都 ≈ 0.13, 跟 κ 几何激活所需的 0.5-1.5 差 5-10 倍**.

| layer | κ=+3 sphere | κ=+10 sphere | usage |
|---|---|---|---|
| L0 | agreement 1.0 / dyn 1.22 | agreement 1.0 / dyn 1.23 | 0.22 |
| L1 | agreement 1.0 / dyn 1.24 | agreement 1.0 / dyn 1.25 | 0.18 |
| L2 | agreement 1.0 / dyn 1.25 | agreement 1.0 / dyn 1.26 | 0.11 |

球面侧**完全跟欧式一致**, 几何信号 0.

## 综合结论 (R11.3 自主决策)

**球面侧 κ 几何线关闭**:
1. 真实训练态 latent ‖z‖ ≈ 0.13, 球面几何看不见
2. 手动放大范数 (scale=5x → ‖z‖=0.65) 球面几何激活, 但同时**码字坍缩** (usage ≤ 39%)
3. 用户判据 "usage ≥ 80% AND agreement 60-90%" 是"几何激活 + usage 健康"双条件, 实测**没有甜区**
4. 用户要求 Phase 1 保留 NormCap → latent 范数被锁在 ~0.13 → 球面侧必然隐形

**整体几何线全部 NO-GO**:
- ❌ 双曲侧 (HG-Rec): 任务 #59 / #176 全文已 NO-GO
- ❌ 球面侧 (本次): latent 范数 + NormCap 互斥
- ❌ exp(θ) 可学习 κ: Task #199 θ 全程未动
- ❌ path regularization: Task #209 dyn 1.27 饱和
- ❌ 低维双曲 + 钉半径: Task #211 R@10=0.0816 util 23%
- ❌ Two-stage decision: Task #212 99% 一致
- ❌ Entailment Cones: Task #213 锥 opening 数值病态
- ❌ Latent Radius Live: Task #214 radius head 无信号
- ❌ J-plan (类别 anchor): Phase 1a/b 全部 NO-GO
- ❌ Phase 0 (球面 κ): latent 范数被 NormCap 锁死

**只剩 §16 的两条逃法**:
- ✅ Task #218 Per-Codeword κ: L1 67.15% / L2 75.69% OPEN (不是"加 κ", 而是"c_k 跟码字绑定")
- ✅ Task #219 Gromov: L0 79.65% OPEN (完全不同的距离函数)

## 下一步候选

| 选项 | 内容 | ROI |
|---|---|---|
| A. 投 Task #218 (Per-Codeword κ, 攻命题 b+d) | L1/L2 走 Per-Codeword κ + 完整 Stage 2/3/4 验证 R@10 vs HG-Rec 0.1020 | 高 (1-2 天, 2 GPU) |
| B. 投 Task #219 (Gromov, 攻命题 a) | L0 走 Gromov + 完整 Stage 2/3/4 | 高 (1-2 天, 2 GPU) |
| C. Hybrid (L0 Gromov + L1/L2 Per-Codeword κ) | 两条互补: 不同层用不同距离函数 | 最高 (2 天, 4 GPU) |
| D. 写综合 NO-GO paper section | 把几何线 8 个方向 + J-plan + Phase 0 全部写进 paper.md §6.7.5-§6.7.6, 作为 "exhaustive null result" | 1 天 (0 GPU) |

按 R11.3 推荐: **C** (两条逃法互补, 命中率最高).

