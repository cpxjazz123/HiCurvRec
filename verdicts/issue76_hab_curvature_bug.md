# Issue #76 — HAB 曲率失效 bug + 过拟合根因诊断

日期: 2026-08-07
结论: **发现并修复 HAB 曲率静默失效 bug (R2 fallback 违规); 定位过拟合真实机制 = HAB 参数不携带几何信息**

---

## 1. 触发实验: v15 真曲率 SID + v77 HAB 对照

`taskA/_history/issue141_v15_hab_stage3/` (PID 378135, DDP 4×L40S, early stop @ep154, best_epoch=144)

唯一变量 vs v77: Stage2 SID/ckpt 从 flat Euclidean 换成 v15 真曲率.

| Stage2 | κ | c | 来源 |
|---|---|---|---|
| v77 (`taskA_stage2_hyp_v2_capmatch_1000ep`) | [0, 0, 0] | [1, 1, 1] | Gate2 FAIL (κ std=0) |
| v15 (`taskA_stage2_v15_capmatch_1000ep`) | [0.304, 1.792, 1.480] | [1.355, 6.002, 4.394] | Gate2 PASS, util_3digit=[1,1,1] |

Stage3 配置严格对齐 v77: `λ_max=0.20 residual_alpha_init=-20.0 hab_lambda_lr_ratio=100 WD=0.01 dropout=0.20 label_smoothing=0.0 bs=1024 lr=4e-4 seed=42 bf16 DDP4`

### 结果

| 指标 | v77 基准 | v15 对照 | Δ |
|---|---|---|---|
| best valid R@10 | 0.1312 | **0.1358** | **+0.0046 (历史新高)** |
| best test R@10 | **0.1080** | 0.1062 | −0.0018 |
| valid/test ratio @best | 1.215 | **1.2775** | +0.0625 (恶化) |

在线 test eval 轨迹 (`stage4_test_on_best/history.json`, 16 evals):

| ep | valid | test | ratio |
|---|---|---|---|
| 20 | 0.1161 | 0.0919 | 1.2629 |
| 60 | 0.1267 | 0.1026 | **1.2353** |
| 70 | 0.1286 | 0.1046 | 1.2295 |
| 95 | 0.1320 | 0.1057 | 1.2491 |
| 110 | 0.1329 | 0.1061 | 1.2528 |
| 130 | 0.1356 | 0.1062 | 1.2775 |

**读法**: 真曲率给了纯容量增益 (valid 创历史新高 0.1358), 但增益 100% 被过拟合吃掉 (test 反而 −0.0018, ratio 从 1.215 恶化到 1.2775).

---

## 2. 根因 1 (bug): HAB 曲率被静默压成 c=1e-6

`common/hyperbolic_attention_bias.py:119` (修复前):

```python
c_l = max(-kappa_l, 1e-6)  # κ 为负, c = -κ > 0; 若 κ ≥ 0 用极小正值保护
```

这个符号约定来自 Issue #61 (当时 κ 为负, c = -κ 成立). 但 CURV_PRIOR 生效后 `c = exp(κ)`, κ 变正:

- v15 κ = [+0.304, +1.792, +1.480] → `-κ` 全负 → `max(..., 1e-6)` 全部压成 **c = [1e-6, 1e-6, 1e-6]**
- 即 HAB 用的是**近乎完全平坦的欧氏距离**, 从未真正双曲

实测验证 (修复前):
```
final_kappas from ckpt = [0.3036, 1.7921, 1.4803]
c_l used by HAB        = [1e-06, 1e-06, 1e-06]     ← 全部退化
Dbar median            = [0.7085, 0.3508, 0.3154]
p95/median             = [1.2439, 1.2044, 1.2235]  ← 三层形状比几乎一致 = 欧氏特征
```

**这是 R2 禁止的 fallback**: 用默认值 `1e-6` 静默掩盖了"κ 符号约定不匹配"这一预期外失败, 导致
Issue #64/#71/#138/#141 全系列 HAB 实验的"双曲" bias 实际都是欧氏距离的重参数化.

### 修复

`final_cs` 一直存在 Stage2 ckpt 的 top-level key 里 (`['config','final_kappas','final_cs','final_mix_weights']`),
是 Stage2 训练时实际生效的曲率, 唯一可信来源. 改动:

1. `load_hab_assets_from_stage2_ckpt` 返回 `final_cs` 而非 `final_kappas`; 缺 key → `raise KeyError`;
   长度≠3 或 c≤0 → `raise ValueError` (无 fallback)
2. `precompute_distance_matrices(codebook_list, final_cs, ...)` 直接用 c, 不再由 κ 反推;
   `kappa_l = math.log(c_l)` 仅用于 stats 记录
3. Stage3 + Stage4 两个调用点改用 `hab_final_cs`, 并 log `final_cs` 供审计

修复后实测:
```
final_cs   = [1.3547, 6.0021, 4.3941]
Dbar med   = [0.7260, 0.3607, 0.3208]   (修复前 [0.7085, 0.3508, 0.3154])
p95/med    = [1.2374, 1.1974, 1.2199]
```

---

## 3. 根因 2 (架构): 码字挤在球心, 双曲度量无从发挥

修复 c 后 Dbar 只变 **+2.5%** — 远不足以解释 test gap. 查码字径向分布:

| layer | c | ‖e‖ med | R=1/√c | ρ = 占球半径% |
|---|---|---|---|---|
| L0 | 1.355 | 0.2441 | 0.8592 | **27.67%** |
| L1 | 6.002 | 0.1233 | 0.4082 | **29.31%** |
| L2 | 4.394 | 0.1113 | 0.4770 | **22.91%** |

Poincaré 球的指数体积增长只在 ρ→1 (靠近边界) 时显著. 在 ρ < 0.3 区域, Poincaré 度量与欧氏度量
几乎无差别 —— 这解释了为什么把 c 从 1e-6 修正到 6.0 后 Dbar 只动了 2.5%.

**过拟合的真实机制**:
HAB 注入的 14342 个参数 (U/V rank=16 + λ_raw + residual_alpha) 携带的 bias ≈ 欧氏距离的重参数化,
**不携带真实几何归纳偏置**. 对 T5 而言这就是纯粹的额外自由度 → 只能记住 valid 的模式 → valid 涨 test 跌.

即: **过拟合不是正则不够 (WD/dropout/label_smoothing 都调过, v75/v85 全 NO-GO), 是加的参数没有信息量.**

`REL_STRUCT_TARGET_MAX = 0.55` 的 clamp 正是径向瓶颈 —— target 上限 0.55 且深层实际只达到 0.23–0.29.

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS (未改动). 沿用 `taskA_stage1_hyp_v2/item_emb.parquet`
(sha256 `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`), per-item radius R_MAX=0.99.

**Gate 2 (Stage2)**: PASS (复用 v15, 未重训). `taskA_stage2_v15_capmatch_1000ep` verdict `gate2_decision=PASS`,
κ=[0.304,1.792,1.480] κ_per_layer_std=0.641 (diff_ok=true), util_per_layer_3digit=[1.0,1.0,1.0], util_4digit=1.0,
9922 unique 4-digit SID 无碰撞, precheck_pass=true, ablation_diff_ok=true.
**但新发现**: 码字径向 ρ=[27.7%, 29.3%, 22.9%] 全在近欧氏区, 这是 Gate 2 现有验收项**未覆盖**的缺陷 —
κ 数值达标 ≠ 双曲几何真正生效. 建议 Gate 2 增补径向验收项.

**Gate 3 (Stage3)**: PASS (训练正常收敛). `issue141_v15_hab_stage3` early stop @ep154 best_epoch=144
best_loss=1.9751, 无 NaN/Inf, loss 单调下降 3.34→1.89, R23 七信号全未触发, ckpt 正常保存 (22.5MB).
DDP 4 卡 12s/epoch. HAB λ_eff 从 init 0.0924 正常参与梯度.

**Gate 4 (Stage4)**: **FAIL**. best test R@10 = **0.1062** < v77 0.1080 (−0.0018), 未达 0.1100 目标 (−0.0038).
虽然 valid R@10 = 0.1358 创历史新高 (+0.0046 vs v77), 但 valid/test ratio 恶化到 1.2775 (v77 1.215),
增益未转化为泛化. 失败原因 = 上述根因 1 (HAB c 静默退化为欧氏) + 根因 2 (码字 ρ<0.3 双曲度量无效),
HAB 参数不携带几何信息 → 纯增自由度 → 过拟合.

---

## 5. 下一步

1. **已启动**: `taskA/_history/issue76_habcfix_v15/` (PID 422341) — 唯一变量 = HAB c 修复
   (c: 1e-6 → [1.355, 6.002, 4.394], Dbar: [0.7085,...] → [0.7260,...]). 量化 bug 修复的端到端影响.
2. **Phase A (径向扩张)**: 提高 `REL_STRUCT_TARGET_MAX`, 把码字 ρ 从 0.27 推到 0.7+, 让双曲度量真正生效.
   这是让 HAB 参数携带真实几何信息的前提.
3. **Phase A.2 (行为监督 κ)**: 行为图 (train split only) L0/W0=10, L1/W1=3, L2 next-item + 流行度校正
   + KL(p_beh‖p_geo), 让曲率对齐推荐信号而非重构信号.

## 6. 产物

- `taskA/_history/issue141_v15_hab_stage3/` — 对照实验 (verdict.json, train.log, stage4_test_on_best/history.json)
- `taskA/_history/issue76_habcfix_v15/` — 修复验证 (进行中, PID 422341)
- `common/hyperbolic_attention_bias.py` — bug 修复 (loader 返回 final_cs + 去 fallback)
- `common/stage3/stage3_train_pure_t5.py` / `common/stage4/stage4_eval_pure_t5.py` — 调用点同步
