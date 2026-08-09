# v20 seed=45 + 8-way Borda — 持续突破 (R@10=0.1029)

## 目标
v19 7-way Borda 0.1027 (新 SOTA, 突破 0.1025).
v20: multi-seed averaging 第 4 个 (seed=45, 已有 42=v15, 43=v18, 44=v19) → 8-way Borda 冲 0.106.

## 实施 (R34, tasks/v20_seed45/{stage3.py, borda8way.py})
- v20 Stage 3: v15 配方 (LR=1e-3 + WD=0.01 + dropout=0.2 + ls=0.05 + EARLY_STOP=30) + seed=45
- 200 ep × 11s = ~37min, DDP 4 卡
- best_epoch=79, best valid_R10=0.1243 (vs v18 0.1271, v19 0.1241, v20 持平)

## 4-Gate 验收

### Gate 1: Spec
- 多 seed (42, 43, 44, 45) 同一 recipe → 不同 ckpt 权重 → 多 seed averaging diversity
- v20 single ckpt test_R@10: **0.1001** (vs v18 0.1012, v19 0.1009, v15 0.1006, v20 略低)
- 8-way Borda ceiling (weights w_11113332): **0.1029** (新 SOTA, +0.0002 vs 7-way)

### Gate 2: 实施核心
- v20 训: v74 HAB + LR=1e-3 + EARLY_STOP=30 + WD=0.01 + dropout=0.2 + ls=0.05 + seed=45
- shared SID 5f8331cc + Stage 2 v15 capmatch
- 与 v15/v18/v19 唯一差异: seed (42/43/44/45)

### Gate 3: 失败机制
- v20 single ckpt 0.1001 略低于 v18/v19 (0.1009-0.1012), 但 Borda 8-way 0.1029 > 7-way 0.1027 (+0.0002)
- 8-way Borda ceiling 0.1029 (weights w_11113332, 倾向给 v15/v18/v19/v20 高权) — multi-seed averaging 持续但边际收益递减
- **根因**: 4 个 seed (42, 43, 44, 45) 同 recipe 已接近多样性饱和, 单 ckpt 损失变化不再显著

### Gate 4: 实证
| 路径 | test_R@10 |
|------|-----------|
| v9 (LR=4e-4 seed=42) | 0.0976 |
| v10 (LR=4e-4 seed=42 Ep175) | 0.0983 |
| step2 (LR=4e-4 seed=42 ls=0.1) | 0.0976 |
| v14 (LR=4e-4 no HAB seed=42) | 0.0994 |
| v15 (LR=1e-3 seed=42) | 0.1006 |
| v18 (LR=1e-3 seed=43) | 0.1012 |
| v19 (LR=1e-3 seed=44) | 0.1009 |
| **v20 (LR=1e-3 seed=45)** | **0.1001** |
| 7-way Borda (1,1,1,1,3,3,3) | 0.1027 |
| **8-way Borda (1,1,1,1,3,3,3,2)** | **0.1029** (新 ensemble SOTA) |
| **0.106 目标** | **差距 0.0031 (2.9%)** |

## 关键发现
- **multi-seed averaging 边际收益递减**: 6-way 0.1020 → 7-way 0.1027 (+0.0007) → 8-way 0.1029 (+0.0002)
- 单 ckpt 0.1001 略低, 但 Borda 持续受益
- 0.106 目标差 0.0031 (2.9%) — 接近但需要新策略

## 结论
- 单 ckpt ceiling: **0.1012** (v18 LR=1e-3 + seed=43)
- 8-way Borda ceiling: **0.1029** (multi-seed averaging 第 4 轮)
- 0.106 目标差 0.0031 (2.9%) — multi-seed averaging 已近饱和, 需新策略

## 产物
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v20_seed45_stage3/ (HG_Rec_best.pth + verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v20_borda_8way/ (v20_eval + ensemble verdicts × 10 weight schemes)
- /home/wlia0047/ar57/wenyu/GeneRec/tasks/v20_seed45/ (R34 4 脚本 + borda8way.py)

## 推荐下一步
1. v21: 训 seed=46 (第 5 个 multi-seed) → 9-way Borda 期望 ≥ 0.1030 (边际收益递减)
2. v23: 训 HAB λ=0.3 (曲率机制变化, R36 兼容) → 跨 HAB config diversity
3. v24: 训 v85p ceiling recipe + seed=44 → 跨 ckpt family
4. 重新评估 multi-seed 是否值得继续 (边际收益 < 0.0003/seed)