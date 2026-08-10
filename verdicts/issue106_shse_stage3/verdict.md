# Issue #106 — Stage3 SHSE (Hyperbolic SID Embedding) Verdict (R37 NO-GO: test_R@10 regress)

## 状态: ❌ NO-GO (R37: v29 test_R@10 < v18, -6.6%)

## R37 决策行

v29 SHSE 比 v18 差: test_R@10 0.0944 < v18 0.1011 (-0.0067, -6.63%), 回退至 v18 重新创新 (SHSE line 终止)。

## 4 Gate 验证

| Gate | 期望 | 实测 | 状态 |
|------|------|------|------|
| Gate 1: Stage3 训练健康 | loss ↓ 无 NaN/Inf | loss 5.56→2.62 ↓, 无 NaN/Inf | ✅ PASS |
| Gate 2: SID 对齐 | 沿用 v15 baseline sha | SHA256=5f8331cc462c867f... (v15 baseline 一致) | ✅ PASS |
| Gate 3: Stage3 HAB 兼容 | λ_eff 与 v18 baseline 相近 | λ_eff init=0.0924 (v18 同样 0.0924) | ✅ PASS |
| Gate 4: test_R@10 > v18=0.1011 | > 0.1011 | test_R@10=0.0944 (-6.6%) | ❌ **FAIL** |

## 关键指标轨迹 (Stage3 DDP 4-card, ep1-107)

| Epoch | loss | valid_R@10 | NDCG@20 | 状态 |
|-------|------|------------|---------|------|
| 1     | 5.5629 | -- | -- | init |
| 5     | 3.3846 | 0.1001 | 0.0827 | rising |
| 20    | 2.7954 | 0.1129 | 0.0946 | rising |
| 25    | 2.7436 | 0.1193 | 0.0954 | rising |
| 30    | 2.6971 | 0.1204 | 0.0967 | rising |
| 35    | 2.6637 | 0.1197 | 0.0971 | saturating |
| 40    | (no eval) | -- | -- | -- |
| **55** | **2.6251** | **0.1206** ⭐ | 0.0980 | **BEST** |
| **60** | **2.6261** | **0.1206** ⭐ | 0.0984 | **BEST (tie)** |
| 70-107 | 2.62 | 0.119-0.120 | 0.099 | 饱和 (13/30 no improvement) |

**R38 mid-training trigger**: best_valid 0.1206 < v18 0.1267 (-0.0061, -4.8%) + 53 epoch 饱和, 立即 kill 节省 GPU。

## Stage4 v29 SHSE (单 ckpt + beam=20, R35 强约束)

| 指标 | v29 SHSE | v18 baseline | Δ |
|------|----------|--------------|---|
| test_R@5  | 0.0772 | 0.0819 (v18) | -0.0047 (-5.7%) |
| test_R@10 | **0.0944** | **0.1011 (v18)** | **-0.0067 (-6.6%)** ❌ |
| test_R@20 | 0.1152 | 0.1265 (v18) | -0.0113 (-8.9%) |
| NDCG@10   | 0.0699 | 0.0718 (v18) | -0.0019 (-2.6%) |

## R18 4 维度对比 (v29 SHSE vs v18 baseline)

| 维度 | v18 baseline | v29 SHSE | 差异 |
|------|--------------|----------|------|
| D1 spec | T5 shared(input_ids) Euclidean | T5 shared(input_ids) → SHSE encoder (W_hyp·emb+b_hyp + exp_map_0) → hyperbolic input | **Stage3 input embedding 双曲化** |
| D2 实施核心 | embed 输入直接 attention/CE | W_hyp init=I b_hyp init=0 c=1.0 + exp_map_0 + proj_to_ball | Stage3 forward patch |
| D3 Gate 1 失败机制 | N/A (baseline healthy) | loss 5.56→2.62 健康下降, 无 collapse | NEW: healthy |
| D4 引用文献 | Nickel & Kiela (2017) hyperbolic embedding | 同 + "SHSE — Hyperbolic SID Embedding" 概念 | 与 v18 引用重合 |

**R18 判定**: D1 + D2 都不同 → 满足 R18 实验触发, 实测已跑。

## 根因分析

v29 SHSE 把 T5 shared embedding 通过 exp_map_0 推到 Poincaré ball,但下游 Stage3 HAB / attention / CE loss 都期望 Euclidean 输入:

1. **T5 attention 期望 Euclidean**: self-attention q·k^T 在 Euclidean 假设下, 双曲输入会让 q·k 内积几何失真, attn 权重偏离真实语义
2. **Stage3 HAB Dbar 计算**: HAB Dbar 是基于 Stage2 codebook 几何, 但 Stage3 input embed 变 hyperbolic → HAB 与 input geometry mismatch, λ_eff 信号错位
3. **proj_to_ball 强截断**: 当 input embed norm > 1/sqrt(c)=1.0 时强制缩放 → 数值上稳定但 magnitude 失真

**对比 v28 SHIE (norm mismatch 30%)**:
- v28: Stage1 输出推到 norm=0.76, Stage3 收到 Euclidean 假装 → 7.9% test_R@10 下降
- v29: Stage1 输出 norm=1.0 (Euclidean), Stage3 embed 后插 projection → 6.6% test_R@10 下降 (略好)

SHSE 比 SHIE 影响小(因为 projection 在 Euclidean embedding 局部稳定区工作),但仍破坏 Stage3 内部 Euclidean 假设 → 失败。

## 产物 (留作记录, 不作为下一版本起点)

- Stage2 v15 capmatch ckpt: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt`
- Stage2 v15 capmatch SID: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (SHA256=5f8331cc...2274f07, 9922 unique)
- Stage3 v29 SHSE ckpt: `taskA/_history/v29_shse_stage3/HG_Rec_best.pth` (ep60, valid_R@10=0.1206, 22.5MB)
- Stage4 v29 SHSE eval: `taskA/_history/v29_shse_stage4_beam20_eval/eval_test.json` (test_R@10=0.0944)
- Verdict: `verdicts/issue106_shse_stage3/verdict.md`

## R37 后续 (回退至 v18)

v29 SHSE line **终止**。Stage3 embedding projection path 与 Stage1 SHIE 都证伪 — Stage3 内部的 Euclidean 假设是 T5 base architecture hard constraint,**不能通过 exp_map_0 局部修改**。下一版本必须以 v18 为唯一 base,且需找**完全不破坏 T5 内部 Euclidean 流形**的新机制方向。

可能的 v30 方向 (R36 曲率机制 + 严守 T5 Euclidean):
- Stage3 **外部**引入曲率信号 (例如: 把 Stage2 codebook curvature 作为 soft prior 加到 T5 logits, 不改 embed 流形)
- Stage3 **attention bias**而非 input embed (已在 #100 v24 试过, NO-GO; 但可在更精细的 bias 设计上重试)
- Stage4 **rerank** (已在 #238 v25 PASS 但仅 +0.0001,边际收益已耗尽)

## 修改文件

- `common/stage3/stage3_train_pure_t5_v85p_repro.py` (SHSE flags + SHSEEncoder class + install_shse function + main install call + find_unused 标志)
- `tasks/v29_shse_stage3_from_v18/stage1.py` `stage2.py` `stage3.py` `stage4_beam20.py` (4 脚本固化, R34 合规)

## R39 合规

本 verdict 含 4 Gate 答案 + R37 决策行, Issue #106 commit + push + comment + close 闭环。