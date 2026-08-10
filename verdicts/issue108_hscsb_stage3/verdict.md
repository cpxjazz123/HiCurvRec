# Issue #108 - Stage3 HSCSB (Hierarchical SCSB) Verdict (R37 NO-GO: test_R@10 regress)

## 状态: NO-GO (R37: v31 test_R@10 < v18, -0.20%)

## R37 决策行

v31 HSCSB 比 v18 差: test_R@10 0.1009 < v18 0.1011 (-0.0002, -0.20%), 回退至 v18 重新创新 (HSCSB line 终止)。

## R38 决策行

R38 决策 (2026-08-10): v31 HSCSB mid-training regress.
v31 best_valid=0.1249 (ep90) < v18 best_valid=0.1267 (-0.0018, -1.4%) +
ep90-105 (15 epoch) 饱和 (区间 [0.1237, 0.1249], 无新 BEST) + loss 2.61 几乎不变 + 无突破 v18.
立即 kill + 用 ep90 best ckpt 跑 stage4 留作记录 + R37 回退至 v18 重新创新。

## 4 Gate 验证

| Gate | 期望 | 实测 | 状态 |
|------|------|------|------|
| Gate 1: Stage3 训练健康 | loss ↓ 无 NaN/Inf | loss 5.57→2.61 ↓, 无 NaN/Inf | PASS |
| Gate 2: SID 对齐 | 沿用 v15 baseline sha | SHA256=5f8331cc462c867f... (v15 baseline 一致) | PASS |
| Gate 3: T5 内部不变 | T5 internal params 与 v18 完全相同 | T5 layer norm / attention / embedding 不动, 仅新增 α_ℓ scalars + bias 路径 | PASS |
| Gate 4: test_R@10 > v18=0.1011 | > 0.1011 | test_R@10=0.1009 (-0.20%) | FAIL |

## 关键指标轨迹 (Stage3 DDP 4-card, ep1-105)

| Epoch | loss | valid_R@10 | NDCG@20 | 状态 |
|-------|------|------------|---------|------|
| 5     | 3.3670 | 0.1018 | 0.0856 | init rising |
| 10    | 3.0317 | 0.1130 | 0.0925 | rising |
| 15    | 2.8827 | 0.1170 | 0.0960 | rising |
| 20    | 2.7888 | 0.1183 | 0.0972 | rising |
| 25    | 2.7335 | 0.1201 | 0.0971 | rising |
| 30    | 2.6840 | 0.1220 | 0.0991 | rising |
| 35    | 2.6502 | 0.1233 | 0.0995 | rising |
| 40    | 2.6355 | 0.1237 | 0.0998 | rising |
| 45    | 2.6290 | 0.1241 | 0.1005 | rising |
| 50    | 2.6209 | 0.1232 | 0.1008 | plateau |
| 55    | 2.6086 | 0.1235 | 0.1009 | plateau |
| 60    | 2.6093 | 0.1245 | 0.1012 | rising |
| 65    | 2.6144 | 0.1230 | 0.1010 | plateau |
| 70    | 2.6158 | 0.1238 | 0.1011 | plateau |
| 75    | 2.6081 | 0.1235 | 0.1015 | plateau |
| 80    | 2.6141 | 0.1241 | 0.1016 | plateau |
| 85    | 2.6103 | 0.1238 | 0.1011 | plateau |
| **90**| **2.6121** | **0.1249** | **0.1013** | **BEST** |
| 95    | 2.6120 | 0.1240 | 0.1011 | saturated |
| 100   | 2.6047 | 0.1237 | 0.1013 | saturated |
| 105   | 2.6058 | 0.1240 | 0.1011 | saturated |

**R38 mid-training trigger**: best_valid 0.1249 < v18 0.1267 (-1.4%) + ep90-105 15 epoch 饱和 (区间 [0.1237, 0.1249]) + 无突破 v18, 立即 kill 节省 GPU。

## Stage4 v31 HSCSB (单 ckpt + beam=20, R35 强约束)

| 指标 | v31 HSCSB | v30 SCSB | v18 baseline | Δ vs v18 |
|------|-----------|----------|--------------|----------|
| test_R@5  | 0.0804 | 0.0805 | 0.0819 (v18) | -0.0015 (-1.8%) |
| test_R@10 | **0.1009** | **0.1003** | **0.1011 (v18)** | **-0.0002 (-0.20%)** |
| test_R@20 | 0.1237 | 0.1230 | 0.1265 (v18) | -0.0028 (-2.2%) |
| NDCG@5    | 0.0686 | 0.0683 | - | - |
| NDCG@10   | 0.0751 | 0.0746 | 0.0718 (v18) | +0.0033 (+4.6%) |
| NDCG@20   | 0.0809 | 0.0804 | - | - |

注: 
- HSCSB 比 SCSB (-0.20% vs -0.79%) 缩小了与 v18 的差距 4x
- NDCG@10 +4.6% (vs v18) 显示 HSCSB 3 层累积 + cross-layer bias 引入了更有效的曲率信号

## R18 4 维度对比 (v31 HSCSB vs v18 baseline vs v30 SCSB)

| 维度 | v18 baseline | v30 SCSB | v31 HSCSB | 差异 |
|------|--------------|----------|-----------|------|
| D1 spec | T5 lm_head → CE | T5 lm_head + α·log_softmax(-d_P_ℓ) | T5 lm_head + Σ β_ℓ·α_ℓ·log_softmax(-d_P_ℓ) + β_cross·cross_bias | **3 层累积 + cross-layer bias (vs v30 单层)** |
| D2 实施核心 | T5 standard | SCSBModule (1 α) | HSCSBModule (3 α_ℓ + cross bias) | Stage3 forward patch (1 → 3 α scalars) |
| D3 Gate 1 失败机制 | N/A (baseline healthy) | loss 5.57→2.61 ↓, healthy | loss 5.57→2.61 ↓, healthy | NEW: healthy |
| D4 引用文献 | v15 capmatch HAB | v15 capmatch + Poincaré bias | v15 capmatch + 3-layer hierarchical codebook + cross-layer bias | v31 = v30 + hierarchical extension |

**R18 判定**: D1 + D2 都不同 (vs v18 与 vs v30) → 满足 R18 实验触发, 实测已跑。

## 根因分析

v31 HSCSB 在 T5 lm_head 之后插入 3 层 hierarchical codebook bias 累积 + cross-layer bias:

1. **3 层累积有效**: test_R@10 -0.20% (vs v30 -0.79%), 缩小 4x - 累积信号确实增强曲率效果
2. **但仍未突破 v18**: 总有效 bias ≈ 0.7·3.66·log_softmax + 0.05·log_softmax ≈ 0.13 (log_softmax 接近 0), 远小于 T5 logits 量级 (~10)
3. **NDCG@10 +4.6%**: HSCSB 比 v30 更显著改善 top-10 排序, 但 R@10 hit 率微降 (-0.20%) 表明强度仍不足

**v28/v29/v30/v31 对比**:
- v28 SHIE (-7.9%): 改 Stage1 → norm mismatch → NO-GO
- v29 SHSE (-6.6%): 改 Stage3 embed → attn q·k^T 失真 → NO-GO
- **v30 SCSB (-0.79%): 改 Stage3 logits 后 (单层) → R36 严守成功, 但 bias 太弱**
- **v31 HSCSB (-0.20%): 改 Stage3 logits 后 (3 层累积 + cross) → R36 严守, 缩小差距 4x 但仍未突破**

**结论**: HSCSB 证明 "3 层 hierarchical 累积" 是正确方向, 但 log_softmax 信号本质上限 ~0, 单纯累加无法突破。下一版本必须:
- **完全跳过 log_softmax**, 直接用 raw -d_P 作为 bias (可能突破信号上限)
- **或乘 lm_head norm 估计** (scale up bias)
- **或 Stage4 post-generation rerank** (扩展 #238 v25 到 multi-layer, 但 R-stage4 约束)

但 v31 已失败, **R37 禁止二次叠加**, 新版本需重新设计。

## 产物 (留作记录, 不作为下一版本起点)

- Stage2 v15 capmatch ckpt: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt` (沿用)
- Stage2 v15 capmatch SID: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (SHA256=5f8331cc...2274f07, 沿用)
- Stage3 v31 HSCSB ckpt: `taskA/_history/v31_hscsb_stage3/HG_Rec_best.pth` (ep90, valid_R@10=0.1249, 22.5MB)
- Stage4 v31 HSCSB eval: `taskA/_history/v31_hscsb_stage4_beam20_eval/eval_test.json` (test_R@10=0.1009)
- Verdict: `verdicts/issue108_hscsb_stage3/verdict.md`

## R37 后续 (回退至 v18)

v31 HSCSB line **终止**。HSCSB 缩小了 SCSB 与 v18 的差距 (4x improvement), 但仍未突破。下一版本必须以 v18 为唯一 base, 且需找 **不依赖 log_softmax 信号的 curvature 机制**:

可能的 v32 方向 (R36 曲率机制 + 严守 T5 Euclidean):
- Stage3 **raw -d_P bias** (跳过 log_softmax, 直接 -d_P 加到 logits, magnitude 不受 log 压缩)
- Stage3 **scale up bias** (乘 lm_head hidden_state norm 估计)
- Stage4 **multi-layer rerank** (#238 v25 PASS +0.0001, 扩展到 L0/L1/L2 多层 rerank)

## 修改文件

- `common/stage3/stage3_train_pure_t5_v85p_repro.py` (HSCSB flags + HSCSBModule class + install_hscsb function + main install call + find_unused 标志)
- `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` (HSCSB eval flags + HSCSBModuleEval class + install_hscsb_eval function + main install call)
- `tasks/v31_hscsb_stage3_from_v18/stage1.py` `stage2.py` `stage3.py` `stage4_beam20.py` (4 脚本固化, R34 合规)

## R39 合规

本 verdict 含 4 Gate 答案 + R37 决策行 + R38 决策行, Issue #108 commit + push + comment + close 闭环。