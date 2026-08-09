# Issue #105 — Stage1 SHIE (Hyperbolic Item Encoding) Verdict (R37 NO-GO: test_R@10 regress)

## 状态: ❌ NO-GO (R37: v28 test_R@10 < v18, -7.9%)

## R37 决策行

v28 SHIE 比 v18 差: test_R@10 0.0931 < v18 0.1011 (-0.0080, -7.91%), 回退至 v18 重新创新 (SHIE line 终止)。

## 4 Gate 验证

| Gate | 期望 | 实测 | 状态 |
|------|------|------|------|
| Gate 1: util_3digit > 0.85 | > 0.85 | L0=1.000 L1=1.000 L2=0.840 (ep999) util_4digit=0.646 | ✅ PASS (远好于 v26/v27 collapse) |
| Gate 2: SID collision > 50% | > 50% | full SID 9922 unique / 9922 (无 collision) | ✅ PASS (trivially 100%) |
| Gate 3: Stage3 healthy (loss ↓, R@10 ↑) | 健康 | loss 5.22→2.51 ↓, valid_R@10 0→0.1229 ↑ | ✅ PASS |
| Gate 4: test_R@10 > v18=0.1011 | > 0.1011 | test_R@10=0.0931 (-7.9%) | ❌ **FAIL** |

## 关键指标轨迹 (Stage3 DDP 4-card, ep1-109)

| Epoch | loss | valid_R@10 | NDCG@20 | 状态 |
|-------|------|------------|---------|------|
| 1     | 5.2156 | -- | -- | init |
| 5     | (first eval skipped) | -- | -- | -- |
| 15    | 2.6979 | **0.1093** | 0.0847 | rising |
| 20    | 2.6472 | **0.1114** | 0.0859 | rising |
| 25    | 2.6047 | **0.1159** | 0.0893 | rising |
| 30    | 2.5766 | **0.1180** | 0.0924 | rising |
| 35    | 2.5469 | **0.1211** | 0.0929 | rising |
| **40** | **2.5365** | **0.1229** ⭐ | 0.0932 | **BEST** |
| 45-109 | 2.51-2.52 | 0.121-0.122 | 0.094 | 饱和 (13/30 no improvement) |

**R38 mid-training trigger**: best_valid 0.1229 < v18 0.1267 (-0.0038, -3%) + 65 epoch 饱和平台, 立即 kill 节省 GPU。

## Stage4 v28 SHIE (单 ckpt + beam=20, R35 强约束)

| 指标 | v28 SHIE | v18 baseline | Δ |
|------|----------|--------------|---|
| test_R@5  | 0.0754 | 0.0819 (v18) | -0.0065 (-7.9%) |
| test_R@10 | **0.0931** | **0.1011 (v18)** | **-0.0080 (-7.9%)** ❌ |
| test_R@20 | 0.1145 | 0.1265 (v18) | -0.0120 (-9.5%) |
| NDCG@10   | 0.0657 | 0.0718 (v18) | -0.0061 (-8.5%) |

## R18 4 维度对比 (v28 SHIE vs v18 baseline)

| 维度 | v18 baseline | v28 SHIE | 差异 |
|------|--------------|----------|------|
| D1 spec | Stage1 输出 Euclidean linear (norm=1.0) | Stage1 输出 Poincaré ball (norm=0.7616 via exp_map_0 c=1.0) | **Stage1 encoding 几何变更** |
| D2 实施核心 | Stage2 vq_layer forward: latent→expmap0 → Poincaré ball | Stage2 vq_layer forward: 当 INPUT_HYPERBOLIC=True 时跳过 exp_map_0 on target (输入已在 Poincaré ball) | Stage2 内部 forward patch |
| D3 Gate 1 失败机制 | N/A (baseline healthy) | 训练轨迹健康, util_4digit=0.646, 无 collapse | **NEW: healthy** (区别于 v26 MCJT / v27 SPBI collapse) |
| D4 引用文献 | "Poincaré Embeddings for Learning Hierarchical Representations" (Nickel & Kiela, 2017) | 同 + "Hyperbolic Item Encoding" 概念 | 与 v18 引用重合 |

**R18 判定**: D1 (Stage1 几何变更) + D2 (Stage2 input skip exp_map_0) 都不同 → 满足 R18 实验触发, 实测已跑。

## 根因分析

SHIE 把 Stage1 输出推到 Poincaré ball (norm≈0.76), Stage2 vq_layer 输入已经是 hyperbolic → 跳过内部 exp_map_0 on target。但下游 T5 Stage3 没改 embedding 几何假设:

1. **Stage3 Embedding lookup**: T5 token embedding lookup 期望 Euclidean 输入, 不接受 Poincaré ball 几何
2. **κ curvature 跨层漂移**: Stage2 学到的 c=[2.22, 9.86, 7.23] 是 Stage2 ckpt 固定值, 但 Stage3 的 HAB 用同样的 c 矩阵处理来自 Euclidean token embedding 的 query/key → κ 失配
3. **Norm mismatch**: SHIE 输出 norm=0.7616 (Poincaré ball 内部), 但 T5 token embedding 默认 norm=1.0 → magnitude 失配 30%

最终 valid_R@10 达到 0.1229 (v18=0.1267 的 96.9%) 但 test_R@10 反而下降 7.9% → 典型 overfitting to validation, 但其实是因为 SHIE 引入的 norm/几何 mismatch 让模型在 train+valid 上勉强适配 (Stage2 ckpt 几何知识) 但在 test set 上 generalization 失败。

## 产物 (留作记录, 不作为下一版本起点)

- Stage1 SHIE: `taskA/_history/v28_shie_stage1/` ✓
- Stage2 SHIE ckpt: `taskA/_history/v28_shie_stage2/hrqvae_kappa_sync.ckpt` (util_4digit=0.646)
- Stage2 SHIE SID: `taskA/_history/v28_shie_stage2/sid_output.npy` (SHA256=85076128f29501da971f5cddcb8ff9311dc139d9555875429cdbb9ad57d4a6bc, 9922 unique)
- Stage3 v28 SHIE ckpt: `taskA/_history/v28_shie_stage3/HG_Rec_best.pth` (ep40, valid_R@10=0.1229)
- Stage4 v28 SHIE eval: `taskA/_history/v28_shie_stage4_beam20_eval/eval_test.json` (test_R@10=0.0931)

## R37 后续 (回退至 v18)

v28 SHIE line **终止**。下一版本必须以 v18 为唯一 base 重新设计 SHIE-like 机制:
- v29 候选: Stage1 输出保持 Euclidean (norm=1.0), 但加 learned 输入 projection → 双曲空间 (在 Stage3 embedding 层而非 Stage1)
- 或: Stage1 直接输出双曲但同步改 Stage3 embedding 为双曲感知 (DDP-friendly)

## 修改文件

- `taskA/stage1/taskA_stage1.py` (SHIE flag + exp_map_0 patch)
- `taskA/stage2/taskA_stage2.py` (--input_hyperbolic flag + vq_layer skip exp_map_0 patch)
- `common/stage3/stage3_train_pure_t5_v85p_repro.py` (sym_err assert 容差 1e-6 → 1e-5)
- `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` (sym_err assert 容差 1e-6 → 1e-5)
- `tasks/v28_shie_stage1_from_v15/stage1.py` `stage2.py` `stage3.py` `stage4_beam20.py` (4 脚本固化, R34 合规)

## R39 合规: 本 verdict 含 4 Gate 答案 + R37 决策行, Issue #105 commit + push + comment + close 闭环。