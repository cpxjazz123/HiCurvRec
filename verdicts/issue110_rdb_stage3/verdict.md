# Issue #110 - Stage3 RDB (Raw Distance Bias) Verdict (R37 NO-GO: -1.4%)

## 状态: NO-GO (R38 + R37: v33 RDB 比 v18 差)

## R37 决策行

v33 RDB (raw -d_P lm_head bias, α_init=1.0) 比 v18 差: test_R@10 0.0997 < v18 0.1011 (-0.0014, -1.4%), 回退至 v18 重新创新 (RDB line 终止).

## R38 决策行 (Stage3 mid-training 早停)

R38 决策 (2026-08-10): v33 RDB mid-training regress.
v33 best_valid=0.1225 (ep40) < v18 best_valid=0.1267 (ep90) (-0.0042, -3.3%) +
连续 35+ epoch (ep40 → ep75) 无新 BEST, 区间 [0.1216, 0.1225] 饱和平台 +
loss 2.61 平台期无下降 + 无突破 v18.
立即 kill (PID=3307773 + 4 children) + 用 ep40 best ckpt 跑 stage4_beam20 留记录 + R37 回退至 v18 重新创新 (RDB line 终止).

Stage4 留记录: test_R@10=0.0997 (-1.4% vs v18).

## 4 Gate 验证

| Gate | 期望 | 实测 | 状态 |
|------|------|------|------|
| Gate 1: Stage3 训练健康 | loss 5.5→2.6, 无 NaN/Inf | loss 5.57→2.61 ↓, 无 NaN/Inf | PASS |
| Gate 2: SID 对齐 | v15 capmatch sha 一致 | SHA256=5f8331cc462c867f...2274f07 (v15 baseline 一致) | PASS |
| Gate 3: T5 内部不变 | 仅新增 α + bias 路径, 不改 T5 | T5 layer norm / attention / embedding 不动, 仅新增 1 个 α scalar + raw -d_P bias | PASS |
| Gate 4: test_R@10 > v18=0.1011 | > 0.1011 | test_R@10=0.0997 (-1.4%, -0.0014) | FAIL |

## 关键指标轨迹 (Stage3 DDP 4-card, ep1-75)

| Epoch | loss | valid_R@10 | NDCG@20 | 状态 |
|-------|------|------------|---------|------|
| 1     | 5.5734 | -     | -     | init |
| 5     | 3.3678 | 0.1036 | 0.0849 | first eval |
| 10    | 3.0633 | 0.1151 | 0.0929 | rising |
| 15    | 2.8827 | 0.1153 | 0.0939 | plateau |
| 20    | 2.7893 | 0.1182 | 0.0962 | rising |
| 25    | 2.7330 | 0.1214 | 0.0989 | rising |
| 30    | 2.7183 | 0.1218 | 0.0995 | plateau |
| 35    | 2.6506 | 0.1195 | 0.0995 | dip |
| **40**| **2.6356** | **0.1225** | **0.0999** | **BEST** |
| 45    | 2.6277 | 0.1225 | 0.0999 | no improv 1/30 |
| 50    | 2.6209 | 0.1224 | 0.0991 | no improv 2/30 |
| 55    | 2.6085 | 0.1208 | 0.0995 | dip |
| 60    | 2.6088 | 0.1222 | 0.0996 | recovery |
| 65    | 2.6136 | 0.1217 | 0.0992 | no improv 5/30 |
| 70    | 2.6128 | 0.1216 | 0.0997 | no improv 6/30 |
| 75    | 2.6065 | 0.1220 | 0.0996 | saturated |

**R38 mid-training trigger**: best_valid 0.1225 < v18 0.1267 (-3.3%) + ep40→ep75 35 epoch 饱和 (区间 [0.1216, 0.1225]) + 无突破 v18, 立即 kill 节省 GPU.

## Stage4 v33 RDB (单 ckpt + beam=20, R35 强约束)

| 指标 | v33 RDB (α=1.0) | v31 HSCSB | v30 SCSB | v18 baseline | Δ vs v18 |
|------|----------------|-----------|----------|--------------|----------|
| test_R@5  | 0.0796 | 0.0804 | 0.0805 | 0.0819 (v18) | -0.0023 (-2.8%) |
| test_R@10 | **0.0997** | **0.1009** | **0.1003** | **0.1011 (v18)** | **-0.0014 (-1.4%)** |
| test_R@20 | 0.1232 | 0.1237 | 0.1230 | 0.1265 (v18) | -0.0033 (-2.6%) |
| NDCG@5    | 0.0673 | 0.0686 | 0.0683 | - | - |
| NDCG@10   | 0.0737 | 0.0751 | 0.0746 | 0.0718 (v18) | +0.0019 (+2.6%) |
| NDCG@20   | 0.0797 | 0.0809 | 0.0804 | - | - |

注: 
- RDB 比 v30 SCSB (-0.79%) 略差 (-1.4%) — **raw -d_P 反而比 log_softmax 信号差**
- RDB 比 v31 HSCSB (-0.20%) 更差 — 3 层 log_softmax 累积 > 单层 raw
- NDCG@10 +2.6% 显示 RDB 提升了 top-10 排序, 但 R@10 hit 率下降 (-1.4%)

## R18 4 维度对比 (v33 RDB vs v18 baseline vs v31 HSCSB)

| 维度 | v18 baseline | v31 HSCSB | v33 RDB | 差异 |
|------|--------------|-----------|---------|------|
| D1 spec | T5 lm_head → CE | T5 lm_head + Σ β_ℓ·α_ℓ·log_softmax(-d_P_ℓ) + β_cross | T5 lm_head + α·(-d_P_ℓ) raw (no softmax) | **raw vs log_softmax** |
| D2 实施核心 | T5 standard | HSCSBModule (3 α_ℓ + cross bias) | RDBModule (1 α + raw bias) | Stage3 forward patch |
| D3 Gate 1 失败机制 | N/A (baseline healthy) | loss 5.57→2.61 ↓, healthy | loss 5.57→2.61 ↓, healthy | NEW: healthy |
| D4 引用文献 | v15 capmatch HAB | v15 capmatch + 3-layer hierarchical codebook + cross-layer bias | v15 capmatch + raw -d_P lm_head bias | v33 = v31 去掉 log_softmax |

**R18 判定**: D1 + D2 都不同 (vs v18 与 vs v31) → 满足 R18 实验触发, 实测已跑.

## 根因分析

v33 RDB 在 T5 lm_head 之后插入 raw -d_P bias:

1. **raw -d_P 比 log_softmax(-d_P) 反而差**: v33 test_R@10=-1.4% < v31 HSCSB test_R@10=-0.20% < v30 SCSB test_R@10=-0.79% (注: 绝对差距, 排名是 HSCSB < SCSB < RDB in 此 metric)
   - RDB signal = α · (-d_P) ∈ [-α·max_d_P, 0] ≈ [-3·5, 0] = [-15, 0] (过大!)
   - T5 logits ≈ 10, bias [-15, 0] 远超 logits 量级, **bias 主导 logits, 模型学习错误梯度**
2. **α_eff=2.1932 已经超过 1.0**: Sigmoid(1.0)≈0.73, alpha_max=3.0 → α_eff≈2.19, 有效 bias [-10.97, 0] 仍主导 logits
3. **NDCG@10 +2.6% 显示 RDB 改变了排序**: 但 R@10 hit 率下降 (-1.4%) 说明 RDB 引入的 bias 与实际相关 token 不匹配
4. **centroid 输入**: RDB 用 SID history tokens 的 codebook 平均作 centroid, 与 SCSB/HSCSB 一致; 不同点是 raw activation

**v30/v31/v33 对比** (R36 严守 Stage3 logits 后 bias):
- v30 SCSB (-0.79%): α=1.0 单层, log_softmax 信号弱 → bias 量级 [-0.04, 0]
- v31 HSCSB (-0.20%): α=1.0,1.0,1.0 三层累积 + cross → 缩小 4x 但仍未突破
- **v33 RDB (-1.4%): α=1.0 单层, raw -d_P → bias 量级 [-15, 0] 主导 logits 梯度错误**

**结论**: v33 RDB 验证了 **Issue #110 假设错误**:
- Issue 假设: "raw -d_P 比 log_softmax 信号强, 能突破 v30/v31 饱和"
- 实测: "raw -d_P bias 量级远超 T5 logits, 反而主导梯度方向, 性能下降"

**v33 = -1.4%, v30 = -0.79%, v31 = -0.20%**: raw -d_P 在 lm_head logits 之后是 **错误位置**. raw -d_P 应该:
- 直接替换 attention scoring (类似 #100 v24 但被 CE 正交拒绝)
- 或 rerank (Stage4 post-generation, 但 R-stage4 禁)
- 或缩放到 logits 量级后再加 (e.g., α=0.05, log_softmax-equivalent)

## 产物 (留作记录, 不作为下一版本起点)

- Stage2 v15 capmatch ckpt: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt` (沿用)
- Stage2 v15 capmatch SID: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (SHA256=5f8331cc, 沿用)
- Stage3 v33 RDB ckpt: `taskA/_history/v33_rdb_stage3_from_v18/HG_Rec_best.pth` (ep40, valid_R@10=0.1225, 22.5MB)
- Stage4 v33 RDB eval: `taskA/_history/v33_rdb_stage4_beam20_eval/eval_test.json` (test_R@10=0.0997)
- Verdict: `verdicts/issue110_rdb_stage3/verdict.md`
- R38 line: `verdicts/issue110_rdb_stage3/r38_decision.txt`

## R37 后续 (回退至 v18)

v33 RDB line **终止**. Issue #110 假设错误 (raw -d_P 在 lm_head logits 之后是错误位置). 下一版本必须以 v18 为唯一 base, 且需找:

1. **缩放到 logits 量级**: α=0.05 raw -d_P (类似 log_softmax 量级), 验证是否与 v30 等价
2. **raw -d_P 替换 attention scoring**: 类似 #100 v24 (但被 CE 正交拒绝) — 需要新 attention scoring 机制
3. **跨 stage 联合**: Stage2 + Stage3 + Stage4 联调, 不再单 stage 修补
4. **完全离开 SCoB 家族**: 找全新 mechanism, 例如:
   - Stage3 hyperbolic residual connection (Poincaré residual 加 decoder hidden state)
   - Stage3 cross-task aux loss (类似 rec_loss 但基于 d_P)
   - Stage4 多层 rerank (#238 v25 已 PASS +0.0001, 但 R-stage4 约束)

可能的 v34 方向 (R36 曲率机制 + 严守 T5 Euclidean):
- **Stage3 hyperbolic residual** (Stage3 decoder 加 poincaré residual 路径, 不走 lm_head bias)
- **Stage3 aux hyperbolic CE** (Stage3 加 hyperbolic contrastive aux loss, 类似 MCJT 但 Stage3 范围)
- **Stage3 raw -d_P scaled** (α=0.05 等价测试, 但 R37 禁止在 v33 失败版本上二次叠加)

## 修改文件

- `common/stage3/stage3_train_pure_t5_v85p_repro.py` (RDB argparse + RDBModule + install_rdb + main install + ddp_find_unused 标志)
- `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` (RDB eval argparse + RDBModuleEval + install_rdb_eval + main install)
- `tasks/v33_rdb_stage3_from_v18/stage1.py` `stage2.py` `stage3.py` `stage4_beam20.py` (4 脚本固化, R34 合规)

## R39 合规

本 verdict 含 4 Gate 答案 + R37 决策行 + R38 决策行, Issue #110 commit + push + comment + close 闭环.