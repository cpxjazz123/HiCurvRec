# Issue #111 - Stage3 HRes (Decoder Hyperbolic Residual) Verdict (R23 NaN NO-GO)

## 状态: NO-GO (R23 NaN 触发 + R37 line 终止)

## R23 决策行 (Stage3 forward NaN 立即 kill)

v34 HRes Stage3 launch @ ep1-5 全部 loss=nan, R@10=0.0000. R23 trigger (NaN-Inf) 立即 kill PID=3394627 + 4 DDP workers. **v34 HRes line 终止** (R37 强约束, 不在失败版本上二次叠加).

NaN 根因: exp_map_0(h, c=3.917) + proj_to_ball(clamp norm≤R=0.505) + log_map_0(artanh(√c·norm)/√c · norm) 链在 ‖h‖ 较大时, √c·h_hyp_norm 接近 1.0 → artanh(1.0)=+Inf → residual = Inf → β_eff=0.2512 · Inf = NaN → lm_head logits 全 NaN → CE loss=NaN → backward NaN.

修复 (备 v34_v2 用, 当前 line 终止):
1. safety clamp log_map_0 输入 √c·norm ≤ 0.99 (留 1% 边界)
2. β_raw init 用 logit(-6.2) 而非直接 0.01 (让 β_eff init ≈ 0.001, 真 near-identity)
3. c_avg 用 c_min (L0=1.3547) 替代平均 (3.917), 数值更稳
4. 或者 per-layer HRes (hook 4 个 block 各用对应 c)

## 4 Gate 验证

| Gate | 期望 | 实测 | 状态 |
|------|------|------|------|
| Gate 1: Stage3 训练健康 | loss 5.5→2.6, 无 NaN/Inf | **loss=NaN from ep1, 100% step NaN** | **FAIL** |
| Gate 2: SID 对齐 | v15 capmatch sha 一致 | SHA256=5f8331cc462c867f...2274f07 (沿用 v15 baseline) | PASS |
| Gate 3: T5 内部不变 | 仅新增 β + residual 路径, 不改 T5 | T5 layer norm / attention / embedding 不动, 仅 1 β scalar + exp/log_map residual pre-hook | PASS |
| Gate 4: test_R@10 > v18=0.1011 | > 0.1011 | **N/A (Stage3 NaN, Stage4 未跑)** | **N/A** |

Gate 1 触发 R23 (NaN-Inf).

## Stage3 launch 报告 (R29 强制)

| 项 | 值 |
|----|---|
| Launch PID | 3394627 (DDP 4-card bf16 200 epoch) |
| Children | 13 processes (4 main DDP + helpers) |
| Product dir | /fs04/ar57/wenyu/GeneRec/taskA/_history/v34_hres_stage3_from_v18 |
| Stage2 ckpt (沿用 v15 capmatch) | hrqvae_kappa_sync.ckpt (4.6MB, SHA 一致) |
| SID (沿用 v15 capmatch) | sid_output.npy (SHA=5f8331cc, 9922 unique) |
| Stage3 ckpt | N/A (NaN, 无 save_best 触发) |
| Launch log | /fs04/ar57/wenyu/GeneRec/taskA/_history/v34_hres_stage3_from_v18/launch.log (含 ep1-5 NaN) |

## HRes install 报告

```
[Issue #111 v34 HRes] decoder hyperbolic residual ON: beta_init=0.01 beta_eff=0.2512 c_avg=3.9170 c_per_layer=['1.3547', '6.0021', '4.3941'] learnable_params=1
```

## Stage3 训练轨迹 (ep1-5, 全 NaN)

| Epoch | loss | valid_R@10 | NDCG@20 | 状态 |
|-------|------|------------|---------|------|
| 1     | nan  | (skipped) | -     | NaN |
| 2     | nan  | (skipped) | -     | NaN |
| 3     | nan  | (skipped) | -     | NaN |
| 4     | nan  | (skipped) | -     | NaN |
| 5     | nan  | 0.0000   | 0.0000 | R23 NaN trigger |

R23 触发 @ ep5: val_R@10=0 跨 ≥1 ckpt + loss=NaN 100% step + 立即 kill.

## R18 4 维度对比 (v34 HRes vs v18 baseline vs v33 RDB)

| 维度 | v18 baseline | v33 RDB | v34 HRes | 差异 |
|------|--------------|---------|----------|------|
| D1 spec | T5 lm_head → CE | T5 lm_head + α·(-d_P) raw | T5 lm_head 前 + β·(log_map_0(exp_map_0(h,c),c) - h) | **lm_head 前 vs lm_head 后** |
| D2 实施核心 | T5 standard | RDBModule (1 α + raw bias) | HResModule (1 β + exp/log_map pre-hook on lm_head) | **Stage3 lm_head pre-hook vs lm_head 后** |
| D3 Gate 1 失败机制 | N/A (baseline healthy) | loss 5.57→2.61 ↓, healthy | **NaN from ep1** | **NEW: NaN 触发** |
| D4 引用文献 | v15 capmatch HAB | v15 + raw -d_P lm_head bias | v15 + decoder hyperbolic residual | v34 = 全新路径 (lm_head 前 pre-hook) |

**R18 判定**: D1 + D2 + D3 都不同 (vs v18 + vs v33) → 满足 R18 实验触发, 实测已跑.

## 根因详细分析

### HRes 数学链 (理论上应该是 identity)

```
h ∈ R^{d=128}
h_hyp = exp_map_0(h, c) = tanh(√c·‖h‖)/(√c·‖h‖) · h        # 投影到 Poincaré ball
h_hyp = proj_to_ball(h_hyp, c) = clamp_norm(h_hyp, R=1/√c)  # 数值 hard cap
h_tan = log_map_0(h_hyp, c) = artanh(√c·‖h_hyp‖)/(√c·‖h_hyp‖) · h_hyp
residual = h_tan - h                                          # 理论上 = 0 (identity)
h_out = h + β · residual                                       # 应该 ≈ h (near-identity)
```

### 数值失败点

1. **T5 hidden state norm ‖h‖ ≈ 1-3 (d_model=128, fp32, 训练初期)**
2. **c_avg = 3.917** (从 Stage2 final_cs 平均, 单曲率简化)
3. **√c ≈ 1.98, R = 1/√c ≈ 0.505**
4. **exp_map_0(h, c=3.917)**: √c·‖h‖ ≈ 2-6, tanh(2-6) ≈ 0.96-1.0, factor ≈ 1/(√c·‖h‖) ≈ 0.16-0.5
   - h_hyp ≈ factor · h, norm(h_hyp) ≈ factor · ‖h‖ ≈ 0.16-1.5
   - 但 norm(h_hyp) > R=0.505 时 proj_to_ball 强制 clamp 到 R
5. **proj_to_ball**: clamp_norm(h_hyp, R=0.505) → h_hyp_norm = 0.505
6. **log_map_0(h_hyp_norm=0.505, c=3.917)**: √c·norm = 1.98 · 0.505 ≈ 1.0
   - artanh(1.0) = +∞ → h_tan = Inf · h_hyp_norm_dir = Inf
7. **residual = Inf - h ≈ Inf**
8. **β · Inf = NaN** (β=0.2512)
9. **lm_head(NaN · hidden) = NaN logits**
10. **CE loss = NaN → backward NaN → all params NaN**

**核心错误**: proj_to_ball clamp 到 R=0.505 太激进, 让 √c·h_hyp_norm 正好接近 1.0 → artanh 数值爆炸.

### 修复方向 (备 v34_v2 用, R23 终止当前 line)

| 方案 | 改动 | 安全性 |
|------|------|--------|
| A. safety clamp log_map_0 | 改 _log_map_0: artanh 输入 clamp 到 √c·norm ≤ 0.99 (1% 边界) | 高 (artanh(0.99) ≈ 2.65, 不会 Inf) |
| B. β_raw 真 near-identity | logit_init = log(0.001/0.5)/(1-0.001/0.5) ≈ -6.2, 让 β_eff init ≈ 0.001 | 高 (β 起步更小, 即使有 residual 也不会爆) |
| C. c 用 c_min (L0=1.3547) | c_avg=3.917 → c_min=1.3547, √c=1.16, R=0.86 | 中 (R 更大, ‖h_hyp‖ 上限更高, artanh 输入 < 1.16·0.86 ≈ 0.997) |
| D. per-layer HRes | hook 4 个 decoder block 各用对应 c_l, 不是 lm_head 前 1 次 | 中 (需要新 hook 模式, 复杂) |

**最优**: A+B+C 组合, 既保留 exp/log_map 数学, 又加数值保护.

## v34 HRes vs v33 RDB vs v32 CDR vs v30 SCSB vs v31 HSCSB (5 个 lm_head/logit 路径变体)

| 版本 | 改动 | 路径 | 结果 |
|------|------|------|------|
| v30 SCSB | α·log_softmax(-d_P) lm_head | lm_head 后 additive bias | -0.79% |
| v31 HSCSB | Σ β_ℓ·α_ℓ·log_softmax + cross | lm_head 后 additive bias | -0.20% |
| v32 CDR | d_P diversity Stage2 | Stage2 aux loss | **NaN R23** (antipodal collapse) |
| v33 RDB | α·(-d_P) raw lm_head | lm_head 后 additive bias | -1.4% |
| **v34 HRes** | **β·(log_map_0(exp_map_0(h,c),c) - h)** | **lm_head 前 pre-hook residual** | **NaN R23** (artanh 数值爆炸) |

**R36 严守 Stage3 范围, 但 R23 数值稳定是 hard constraint**: v32 + v34 都因为 Stage2/Stage3 数值发散 R23 终止. **lm_head 附近的 bias 路径 (v30/v31/v33) 仍未被 R23 NaN 触发, 但 R37 已判 NO-GO**.

**Stage3 路径穷尽 5 个变体 (v30/v31/v32/v33/v34), 全部 NO-GO**: v30/v31/v33 R37 NO-GO, v32/v34 R23 NaN. Stage3 单点修补已被 v100-v111 共 12 个 issue NO-GO 证明无效.

## 产物 (留作记录, 不作为下一版本起点)

- Stage2 v15 capmatch ckpt: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt` (沿用)
- Stage2 v15 capmatch SID: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (SHA256=5f8331cc, 沿用)
- Stage3 v34 launch log: `taskA/_history/v34_hres_stage3_from_v18/launch.log` (含 ep1-5 NaN)
- Verdict: `verdicts/issue111_hres_stage3/verdict.md`
- R23 line: `verdicts/issue111_hres_stage3/r23_decision.txt`

## R37 后续 (回退至 v18 + 离开 Stage3 lm_head 周边修补)

v34 HRes line **终止**. v32 + v34 R23 NaN 表明 Stage2 + Stage3 任何曲率机制都有数值稳定性风险. **下一版本必须以 v18 为唯一 base, 离开 Stage3 单点修补思路, 寻找**:

1. **Stage4 多层 rerank**: v238 R-stage4 约束禁, 排除
2. **Stage2 codebook 几何重设计**: 不依赖 numerical gradient, 改 codebook 几何约束 (e.g., 让 codebook 中心限定 norm 上界, 防 exp_map 数值爆炸)
3. **Stage1 hyperbolic encoding 改造**: v105 SHIE -7.9% R37 NO-GO, 但根因是 norm mismatch, 不是 artanh 数值爆炸. 可以借鉴 SHIE 思路但加 safety clamp
4. **完全离开曲率框架, 走 Embedding 路径**: 不动 HRes / HAB / 任何曲率机制, 改 token embedding geometric (但 R36 禁调参)
5. **Stage3 hyperbolic loss as auxiliary**: 不直接改 forward, 只加 hyperbolic contrastive aux loss (类似 MCJT Stage2 但 Stage3 范围)

可能的 v35 方向 (R36 曲率机制 + R23 数值稳定):
- **Stage2 codebook norm hard cap**: Stage2 训练时强制 codebook norm ≤ 0.5/√c (避免后续 Stage3 exp_map 数值爆炸)
- **Stage3 hyperbolic contrastive aux loss** (Stage3 加 hyperbolic 对比学习 aux loss, 不动 forward 主路径)
- **Stage1 SHIE safety clamp** (借鉴 v105 思路但加 norm hard cap)

## 修改文件

- `common/stage3/stage3_train_pure_t5_v85p_repro.py` (HRes argparse + HResModule + install_hres + main install + ddp_find_unused 标志)
- `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` (HRes eval argparse + HResModuleEval + install_hres_eval + main install)
- `tasks/v34_hres_stage3_from_v18/{stage1,stage2,stage3,stage4_beam20}.py` (4 脚本固化, R34 合规)

## R39 合规

本 verdict 含 4 Gate 答案 + R23 决策行 + R37 决策行 + NaN 根因 + 修复方向, Issue #111 commit + push + comment + close 闭环.