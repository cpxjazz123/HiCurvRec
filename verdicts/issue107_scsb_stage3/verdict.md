# Issue #107 - Stage3 SCSB (Curvature Soft-Bias) Verdict (R37 NO-GO: test_R@10 regress)

## 状态: NO-GO (R37: v30 test_R@10 < v18, -0.79%)

## R37 决策行

v30 SCSB 比 v18 差: test_R@10 0.1003 < v18 0.1011 (-0.0008, -0.79%), 回退至 v18 重新创新 (SCSB line 终止)。

## R38 决策行

R38 决策 (2026-08-10): v30 SCSB mid-training regress.
v30 best_valid=0.1244 (ep45) < v18 best_valid=0.1267 (-0.0023, -1.8%) +
30 epoch 饱和 (ep45=0.1244 -> ep75=0.1239, 区间 [0.1231, 0.1244]) + 无突破 v18.
立即 kill + 用 ep45 best ckpt 跑 stage4 留作记录 + R37 回退至 v18 重新创新。

## 4 Gate 验证

| Gate | 期望 | 实测 | 状态 |
|------|------|------|------|
| Gate 1: Stage3 训练健康 | loss ↓ 无 NaN/Inf | loss 5.57→2.61 ↓, 无 NaN/Inf | PASS |
| Gate 2: SID 对齐 | 沿用 v15 baseline sha | SHA256=5f8331cc462c867f... (v15 baseline 一致) | PASS |
| Gate 3: Stage3 内部不变 | T5 internal params 与 v18 完全相同 | T5 layer norm / attention / embedding 不动, 仅新增 α scalar | PASS |
| Gate 4: test_R@10 > v18=0.1011 | > 0.1011 | test_R@10=0.1003 (-0.79%) | FAIL |

## 关键指标轨迹 (Stage3 DDP 4-card, ep1-75)

| Epoch | loss | valid_R@10 | NDCG@20 | 状态 |
|-------|------|------------|---------|------|
| 5     | 3.3671 | 0.1037 | 0.0845 | init rising |
| 10    | 3.0309 | 0.1144 | 0.0943 | rising |
| 15    | 2.8825 | 0.1162 | 0.0967 | rising |
| 20    | 2.7881 | 0.1197 | 0.0975 | rising |
| 25    | 2.7307 | 0.1220 | 0.0984 | rising |
| 30    | 2.6824 | 0.1239 | 0.0994 | rising |
| 35    | 2.6501 | 0.1231 | 0.0996 | plateau |
| 40    | 2.6343 | 0.1230 | 0.0996 | plateau |
| **45**| **2.6294** | **0.1244** | **0.1001** | **BEST** |
| 50    | 2.6203 | 0.1233 | 0.1000 | plateau |
| 55    | 2.6073 | 0.1241 | 0.1004 | still rising |
| 60    | 2.6091 | 0.1243 | 0.1000 | still rising |
| 65    | 2.6160 | 0.1231 | 0.0999 | plateau |
| 70    | 2.6148 | 0.1239 | 0.1001 | plateau |
| 75    | 2.6071 | 0.1239 | 0.1000 | 30 epoch plateau |

**R38 mid-training trigger**: best_valid 0.1244 < v18 0.1267 (-1.8%) + 30 epoch 饱和 (ep45-75), 立即 kill 节省 GPU。

## Stage4 v30 SCSB (单 ckpt + beam=20, R35 强约束)

| 指标 | v30 SCSB | v18 baseline | Δ |
|------|----------|--------------|---|
| test_R@5  | 0.0805 | 0.0819 (v18) | -0.0014 (-1.7%) |
| test_R@10 | **0.1003** | **0.1011 (v18)** | **-0.0008 (-0.79%)** |
| test_R@20 | 0.1230 | 0.1265 (v18) | -0.0035 (-2.8%) |
| NDCG@5    | 0.0683 | - | - |
| NDCG@10   | 0.0746 | 0.0718 (v18) | +0.0028 (+3.9%) |
| NDCG@20   | 0.0804 | - | - |

注: NDCG@10 +3.9% 显示 SCSB 确实给 rerank 引入了曲率信号 (top-10 排序略有改善), 但 R@10 整体 hit 率轻微下降 (-0.79%)。

## R18 4 维度对比 (v30 SCSB vs v18 baseline)

| 维度 | v18 baseline | v30 SCSB | 差异 |
|------|--------------|----------|------|
| D1 spec | T5 lm_head → CE training | T5 lm_head + α·β·log_softmax(-d_P(codeword, history_centroid)) → CE | **Stage3 logits 层后 additive curvature soft bias** |
| D2 实施核心 | T5 standard forward + CE | SCSBModule (codebook+history_centroid+log_softmax) + install_scsb 包裹 forward | Stage3 forward patch (1 个 α scalar + bias 路径) |
| D3 Gate 1 失败机制 | N/A (baseline healthy) | loss 5.57→2.61 健康下降, 无 collapse | NEW: healthy |
| D4 引用文献 | v15 capmatch HAB | v15 capmatch HAB + Nickel & Kiela (2017) Poincaré 距离 + "Curvature Soft-Bias" 新机制 | 与 v18 引用有部分重合但加了 Poincaré bias |

**R18 判定**: D1 + D2 都不同 -> 满足 R18 实验触发, 实测已跑。

## 根因分析

v30 SCSB 在 T5 lm_head 之后插入 α·β·log_softmax(-d_P) 作为 additive bias, 不改 T5 内部任何机制 (R36 严守成功):

1. **NDCG@10 +3.9%**: 证明 curvature bias 确实给 rerank 引入了 Stage2 codebook geometry 信号, top-10 排序有改善
2. **test_R@10 -0.79%**: 但 hit 率轻微下降 -> bias 强度 (β=0.1) 太弱, 不足以纠正 v18 base 的 ranking 误差
3. **α Sigmoid bounded [0,2]**: init=1.0 → α_eff=1.46, β=0.1 → effective bias = 0.146 · log_softmax, log_softmax 最大值 ~0 (log 1), 实际影响 ~0.146 (T5 logits 通常 ~10 量级), 太弱

**与 v28 SHIE / v29 SHSE 对比**:
- v28 SHIE (-7.9%): 改 Stage1 → norm mismatch → NO-GO
- v29 SHSE (-6.6%): 改 Stage3 embed -> attn q·k^T 失真 → NO-GO
- v30 SCSB (-0.79%): 改 Stage3 logits 后 (不破坏 T5 内部) → NDCG 改善, R@10 轻微下降

**结论**: SCSB 机制本身 **不破坏 T5 Euclidean 流形假设** (R36 严守成功, vs v29 -6.6%), 但 bias 信号太弱 (β=0.1 + Sigmoid bounded α ≤ 2), 未能突破 v18 baseline。可能改进方向:
- 增 β (0.1 → 0.3-0.5) → 但失去 "SCSB 只作微调" 的设计意图
- 改 α 上界 (2 → 5) → 但仍受 Sigmoid bounded 限制
- 改用 logit-scaled bias (乘 lm_head norm 估计)

但 v30 已失败, **R37 禁止二次叠加**, 新版本需重新设计。

## 产物 (留作记录, 不作为下一版本起点)

- Stage2 v15 capmatch ckpt: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt` (沿用)
- Stage2 v15 capmatch SID: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (SHA256=5f8331cc...2274f07, 9922 unique, 沿用)
- Stage3 v30 SCSB ckpt: `taskA/_history/v30_scsb_stage3/HG_Rec_best.pth` (ep45, valid_R@10=0.1244, 22.5MB)
- Stage4 v30 SCSB eval: `taskA/_history/v30_scsb_stage4_beam20_eval/eval_test.json` (test_R@10=0.1003)
- Verdict: `verdicts/issue107_scsb_stage3/verdict.md`

## R37 后续 (回退至 v18)

v30 SCSB line **终止**。SCSB 证明 "不改 T5 内部" 的策略有效 (vs v29 -6.6%), 但 bias 信号太弱 (-0.79% 不显著)。下一版本必须以 v18 为唯一 base, 且需找:
- **更强 curvature 信号** (但仍不破坏 T5 内部)
- **或 Stage3 真正外部 prior** (post-generation rerank 类, 已在 #238 v25 PASS 但仅 +0.0001, 边际收益已耗尽)

可能的 v31 方向 (R36 曲率机制 + 严守 T5 Euclidean):
- Stage3 **multi-layer curvature bias** (per-layer α_ℓ learnable, 各自 β_ℓ)
- Stage3 **α unconstrained** (去掉 Sigmoid bounded, init=1.0 但上限 5-10)
- Stage4 **rerank 扩展** (#238 v25 PASS +0.0001 但 #238 只用 L0 codebook, 可拓展到 multi-layer rerank)

## 修改文件

- `common/stage3/stage3_train_pure_t5_v85p_repro.py` (SCSB flags + SCSBModule class + install_scsb function + main install call + find_unused 标志)
- `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` (SCSB eval flags + SCSBModuleEval class + install_scsb_eval function + main install call + SCSB_LAYER_ID_LUT 常量)
- `tasks/v30_scsb_stage3_from_v18/stage1.py` `stage2.py` `stage3.py` `stage4_beam20.py` (4 脚本固化, R34 合规)

## R39 合规

本 verdict 含 4 Gate 答案 + R37 决策行 + R38 决策行, Issue #107 commit + push + comment + close 闭环。