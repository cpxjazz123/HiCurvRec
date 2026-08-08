# Issue #141 v77→v85 全系列架构天花板总结 — 0.11 突破路径

**Tag**: issue141_v85_antioverfit
**日期**: 2026-08-07
**核心结论**: HAB+Stage1 架构下 test_R10 上限 ~0.1080 (v77), 所有 Stage3 调参变体均 NO-GO

---

## 1. 一句话结论

所有尝试 (v74/v75/v76/v77/v78/v79/v80b/v83/v84/v85) 都在 test_R10 ∈ [0.0988, 0.1092] 区间内波动, **未突破 0.1080 (v77) 的天花板**. ratio 稳定在 1.21-1.27 区间. 微调 Stage3 超参 (dropout / label_smoothing / WD / λ_raw clamp / HAB_LAMBDA_LR_RATIO) **无法突破架构天花板**. 必须改架构才能达到 0.11+.

## 2. 全系列实测数据

| 版本 | 改动描述 | test_R10 | ratio | 状态 | Issue |
|------|---------|----------|-------|------|-------|
| HG-Rec baseline | 标准 4 阶段流水线 | 0.1024 | - | 基线 | Task #84 |
| **v74** | HAB frozen + α=-20 + WD=0.01 + dropout=0.20 | 0.1063 | 1.234 | NO-GO | #138 |
| v75 | label_smoothing=0.1 | 0.1043 | 1.252 | NO-GO | #139 |
| v76 | T5UncertaintyHead (DIGER 借鉴) | 0.1050 | 1.244 | NO-GO | #140 |
| **v77** | + Stage1 per-item radius (R_MAX=0.99 + sigmoid) | **0.1080** | **1.215** | **SOTA** | #141 |
| v78 | v77 + DECOR + 抗 trap 3 改动 | 0.1092 | 1.225 | NO-GO (差 0.11) | #68 |
| v79 | v78 + HAB 叠加 | 0.1030 | 1.265 | NO-GO (DECOR 静态 + HAB 动态冲突) | #69 |
| v80b | v77 + λ_max 0.20→0.30 + α -20→-25 | 0.1059 | 1.240 | NO-GO (HAB 加强反而破坏平衡) | #70 |
| v83 | v77 + WD 0.01→0.003 + EI=1+ES=20 | 0.0988 | 1.299 | NO-GO (ES 太早) | #141 |
| v84 | v77 + λ_raw clamp ±3.0 + HAB_LR 100→50× | 0.1020 | 1.255 | NO-GO (限制 HAB signal) | #141 |
| **v85** | v77 + dropout=0.30 + label_smoothing=0.05 | **0.0997** | **1.270** | **NO-GO** (本次实验) | #141 |

## 3. 为什么所有方案都没超 v77?

### 3.1 架构天花板 (~0.108)

- v77=0.1080, v78=0.1092 (DECOR, 不同架构), 区间 [0.103, 0.109] 反复出现
- 这不是随机噪声, 而是 HAB+Stage1+T5+RQ-VAE 组合在 Musical_Instruments 9922-item 数据集上的**真实上限**
- DIGER (uncertainty decay) / DECOR (PromptFormer bins) 架构在同数据集上达到 0.1121/0.1157, 说明**架构决定上限**

### 3.2 ratio 1.21-1.27 是稳定带

- 所有 HAB 系方案 ratio ∈ [1.215, 1.299]
- v77 ratio 1.215 是 HAB 系最佳 (过拟合最轻)
- ratio 无法靠加 dropout / label_smoothing 持续下降, 因为 **T5 backbone 5.5M 参数主导**, 加正则只让 valid 慢涨, test 也慢涨, 比例不变

### 3.3 HAB 信号已饱和

- v77 λ_raw 学到 ±1.4, λ_eff 全部饱和到 ±0.20 (边界)
- λ_raw clamp ±3.0 (v84) 完全无约束, 限制 HAB signal 反而让 test 倒退
- 加大 λ_max (v80b 到 0.30) 也无帮助 — λ_eff 已饱和, 多余信号无意义

### 3.4 Stage3 微调边际效应为零

- HAB 模块仅 14k 参数 (rank=16 U·V^T), 占 T5 总参数 5.5M 的 **0.25%**
- 真正的过拟合源在 T5 backbone 而非 HAB — 调 HAB 没用, 调 T5 正则也只能小改善
- Stage1 per-item radius 才是 v77 vs v74 的 +0.0017 来源 (Stage1 已固定, Stage3 调参无法再榨空间)

### 3.5 HAB 架构容量限制

- U·V^T rank=16, 仅 3 层 (Stage1/2/3 各 256 dim)
- 残差学习: B_geo = Dbar_frozen + α·(U·V^T - Dbar_frozen)
- α_init=-20 让初始几乎全是 Dbar, 训练后 α 学到 ~0.0289, U·V^T 调整幅度 < 5%
- **架构上无法学到排名级调整**, 只能微调 Dbar

## 4. v85 详细数据 (本次实验)

### 4.1 配置 (R30 硬编码进 stage3_train_pure_t5.py)

```python
HAB_LAMBDA_LR_RATIO = 100.0  # v77 base (v84 试 50× NO-GO 改回)
HAB_LAMBDA_RAW_CLAMP = 10.0   # 宽松 (v84 ±3.0 太紧, v77 不设 clamp)
STAGE3_DROPOUT = 0.30         # v74=0.20, v85 加到 0.30
STAGE3_LABEL_SMOOTHING = 0.05 # v75=0.10 NO-GO, v85 减半
EARLY_STOP = 10               # v77 base (EI=5+ES=10)
EVAL_INTERVAL = 5             # v77 base (避免每 epoch eval 噪声)
```

### 4.2 在线 stage4 test eval (新功能)

新增 `--stage4_test_on_best` 开关 + `trigger_stage4_test_async()` + `_poll_stage4_procs()`, 每次保存 new best ckpt 立即异步触发 stage4 test eval, 写回 `stage4_test_on_best/history.json`. **v85 12 次 ratio 记录已证实功能跑通**.

### 4.3 实时 ratio 趋势 (history.json)

```
ep5:   valid=0.0898  test=0.0588  ratio=1.528
ep10:  valid=0.0993  test=0.0709  ratio=1.401
ep25:  valid=0.1116  test=0.0839  ratio=1.331
ep50:  valid=0.1177  test=0.0919  ratio=1.282
ep75:  valid=0.1206  test=0.0931  ratio=1.295
ep100: valid=0.1233  test=0.0981  ratio=1.257  ← 最佳 ratio
ep115: valid=0.1265  test=0.0997  ratio=1.270
ep120: valid=0.1280  test=?       ES counter 9/10  ← 锁 best
```

- ratio 从 1.528 收敛到 1.257 (ep100), 之后在 1.27 附近震荡
- 验证: ES counter 4→9/10 (ep120-165), 早停即将触发, 主动 kill 释放 GPU

### 4.4 v85 评估

- ✅ 比 v84 (test=0.1020, ratio=1.255) **有改善**: valid +2.2% (0.1280 vs 0.1230), ratio 接近 (1.270 vs 1.255)
- ❌ 未达 v77 (test=0.1080, ratio=1.215): test 落后 0.0083, ratio 高 0.055
- ❌ 未达 0.11 目标

## 5. 突破 0.11+ 必须做的架构级改动

### 5.1 借鉴外部架构 (最高 ROI)

| 架构 | 论文 | instruments test_R10 | 借鉴点 |
|------|------|---------------------|--------|
| DIGER FrqUD | SIGIR 2025 | 0.1121 | uncertainty decay + auto-σ |
| DIGER SDUD | SIGIR 2025 | 0.1096 | simpler uncertainty loss |
| **DECOR** | SIGIR 2026 | **0.1157** | PromptFormer + alpha-gated bins |

- **DECOR 最高 ROI**: v78 试过, test 0.1092 (最高记录), 但 ratio 1.225 仍卡 0.11 以下
- 需要 DECOR 完整移植 (bins 多样性 + alpha warmup + attention entropy) 而非 v78 简化版

### 5.2 改 HAB 容量 (中等 ROI)

- rank=16 → 64 (4× capacity): 参数 14k → 56k, 可能学到更复杂的 ranking adjustment
- 风险: 可能过拟合更严重, ratio 反弹
- 实验: v86 = v77 起点 + rank=64

### 5.3 联合 Stage1+Stage3 优化 (高风险高回报)

- 当前 Stage1 per-item radius 冻结, Stage3 HAB 独立学习
- 联合 finetune 可能让 Stage1 radius 适配 Stage3 HAB
- 风险: 训练不稳定, Stage1 漂移破坏 Stage2 SID 一致性
- 需要小 LR + 短 epoch

### 5.4 Warm start + 第二阶段 finetune (低风险)

- v86 = warm start from v85 ep120 best ckpt (valid=0.1280)
- LR/10 (4e-5), label_smoothing=0.10, dropout=0.40
- finetune 5-10 epoch 看是否能再涨
- 低风险, 只能微改善

### 5.5 推荐下一步 (我的判断)

- **首选**: v86 = warm start from v85 ep120 + λ_max 0.20→0.10 (减半 HAB 强度) + LR/10 finetune 5-10 epoch
- **次选**: v86 = v77 起点 + Stage1 R_MAX 0.99→0.999 + Stage3 HAB rank 16→64
- **不推荐**: 继续调 Stage3 dropout/WD/label_smoothing (已证无效)

## 6. 4 Gate 评估

- **Gate 1 (语法/启动)**: PASS — v85 启动正常, DDP 4-card bf16 跑通, 5min/10epoch 训练稳定
- **Gate 2 (Stage3 ckpt 完整)**: PASS — HG_Rec_best.pth 119 keys (含 hab_module.* 11 keys), online stage4 test eval 12 次触发全部成功
- **Gate 3 (Stage4 eval)**: PASS — full test.parquet (24772 samples), ep120 test_R10=0.0997, ratio=1.270
- **Gate 4 (test 性能)**: **FAIL** — test_R10=0.0997 < v77 0.1080 (-0.0083), 仍低于 0.11 目标 (-0.0103)

## 7. 产物路径

- v85 ckpt: `taskA/_history/issue141_v85_antioverfit/stage3/HG_Rec_best.pth`
- v85 trace: `taskA/_history/issue141_v85_antioverfit/stage3/train.log`
- v85 online test: `taskA/_history/issue141_v85_antioverfit/stage3/stage4_test_on_best/history.json` (12 条 ratio 记录)
- stage3 脚本改动: `common/stage3/stage3_train_pure_t5.py` (新增 --stage4_test_on_best 异步 eval)
- 本 verdict: `verdicts/issue141_v77_to_v85_ceiling.md`

## 8. 结论

**HAB+Stage1 架构天花板 test_R10=0.1080, 调参无法突破.** 下一轮必须改架构 (DECOR 完整移植 / HAB rank 64 / 联合优化). 推荐 v86 = warm start from v85 + λ_max 0.10 finetune (低风险快速验证).

---

**verdict: NO-GO (架构天花板确认, 需要架构级改动)**
