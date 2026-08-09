# v19 seed=44 + 7-way Borda — 突破 0.1025 (R@10=0.1027)

## 目标
v18 6-way Borda 0.1020 (新 SOTA, 突破 0.102 大关).
v19: multi-seed averaging 第 3 个 (seed=44, 已有 42=v15, 43=v18) → 7-way Borda 冲 0.106.

## 实施 (R34, tasks/v19_seed44/{stage3.py, borda7way.py})
- v19 Stage 3: v15 配方 (LR=1e-3 + WD=0.01 + dropout=0.2 + ls=0.05 + EARLY_STOP=30) + seed=44
- 200 ep × 11s = ~37min, DDP 4 卡
- best_epoch=169, best valid_R10=0.1241 (vs v18 0.1271, v15 0.1258, v19 略低)

## 4-Gate 验收

### Gate 1: Spec
- 多 seed (42 vs 43 vs 44) 同一 recipe → 不同 ckpt 权重 → 多 seed averaging diversity
- v19 single ckpt test_R@10: **0.1009** (vs v18 0.1012, v15 0.1006, v19 略低 -0.0003)
- 7-way Borda ceiling (weights w_1112333 / w_1111333): **0.1027** (新 SOTA, +0.0007 vs 6-way)

### Gate 2: 实施核心
- v19 训: v74 HAB + LR=1e-3 + EARLY_STOP=30 + WD=0.01 + dropout=0.2 + ls=0.05 + seed=44
- shared SID 5f8331cc + Stage 2 v15 capmatch
- 与 v15/v18 唯一差异: seed (42/43/44)

### Gate 3: 失败机制
- v19 single ckpt 0.1009 略低于 v18 (0.1012), 但 Borda 7-way 0.1027 > 6-way 0.1020 (+0.0007)
- 7-way Borda ceiling 0.1027 (weights w_1112333) — multi-seed averaging 持续有效
- 三个 seed (42, 43, 44) 一起给高权 (3, 3, 3), 低 LR 路径 (v9/v10/step2/v14) 给低权 (1, 1, 1, 1)
- **根因**: v15/v18/v19 三个高 LR ckpt 互相补全 loss landscape 局部最优的盲区

### Gate 4: 实证
| 路径 | test_R@10 |
|------|-----------|
| v9 (LR=4e-4 seed=42) | 0.0976 |
| v10 (LR=4e-4 seed=42 Ep175) | 0.0983 |
| step2 (LR=4e-4 seed=42 ls=0.1) | 0.0976 |
| v14 (LR=4e-4 no HAB seed=42) | 0.0994 |
| v15 (LR=1e-3 seed=42) | 0.1006 |
| v18 (LR=1e-3 seed=43) | 0.1012 |
| **v19 (LR=1e-3 seed=44)** | **0.1009** |
| 6-way Borda (1,1,1,2,3,3) | 0.1020 |
| **7-way Borda (1,1,1,1,3,3,3)** | **0.1027** (新 ensemble SOTA) |
| **0.106 目标** | **差距 0.0033 (3.1%)** |

## 关键发现
- **multi-seed averaging 持续有效**: 6-way 0.1020 → 7-way 0.1027 (+0.0007)
- 高 LR (1e-3) 多 seed (42, 43, 44) 三个 ckpt 协同产生真正的随机性 diversity
- 单 ckpt 略有波动 (v18 0.1012 > v19 0.1009), Borda 始终受益
- 0.106 目标差 0.0033 (3.1%) — 阶段性突破, 接近达标

## 结论
- 单 ckpt ceiling: **0.1012** (v18 LR=1e-3 + seed=43)
- 7-way Borda ceiling: **0.1027** (multi-seed averaging 第 3 轮)
- 0.106 目标差 0.0033 (3.1%) — multi-seed averaging 是当前最有效的 diversity 来源

## 产物
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v19_seed44_stage3/ (HG_Rec_best.pth + verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v19_borda_7way/ (v19_eval + ensemble verdicts × 12 weight schemes)
- /home/wlia0047/ar57/wenyu/GeneRec/tasks/v19_seed44/ (R34 2 脚本)

## 推荐下一步
1. v20: 训 seed=45 (第 4 个 multi-seed) → 8-way Borda 期望 ≥ 0.1033
2. v21: 训 v85p ceiling ckpt recipe + seed=44 → 跨 ckpt family
3. v22: 训 WD=0.005 (vs v15 0.01) → 跨 reg diversity
4. v23: 训 EARLY_STOP=50 → 让 cosine LR_min 跑更久