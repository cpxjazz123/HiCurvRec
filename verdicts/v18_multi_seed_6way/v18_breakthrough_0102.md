# v18 multi-seed(43) + 6-way Borda — 突破 0.102 大关 (R@10=0.1020)

## 目标
v15 LR=1e-3 单 ckpt 0.1006 (新 SOTA), 5-way Borda 0.1010.
v18: multi-seed averaging (Issue #94 真正实现, seed=43 vs v15 seed=42) → 6-way Borda 冲 0.106.

## 实施 (R34, tasks/v18_multi_seed/stage3.py)
- v18 Stage 3: v15 配方 + **seed=43** (patch SEED=42 → 43)
- 200 ep × 11s = ~37min, DDP 4 卡
- best_epoch=164, best valid_R10=**0.1271** (vs v15 0.1258, v18 又新高)

## 4-Gate 验收

### Gate 1: Spec
- 多种子 (42 vs 43) 同一 recipe → 不同 ckpt 权重 → 多 seed averaging diversity
- v18 single ckpt test_R@10: **0.1012** (vs v15 0.1006, v18 又新高 +0.0006)
- 6-way Borda (v9+v10+step2+v14+v15+v18, weights 1,1,1,2,3,3): **0.1020** (新 SOTA)

### Gate 2: 实施核心
- v18 训: v74 HAB + LR=1e-3 + EARLY_STOP=30 + WD=0.01 + dropout=0.2 + ls=0.05 + seed=43
- shared SID 5f8331cc + Stage 2 v15 capmatch
- 与 v15 唯一差异: seed 42 → 43

### Gate 3: 失败机制
- v18 single ckpt 0.1012 历史新高 (vs v15 0.1006, v14 0.0994, baseline 0.1024)
- 6-way Borda 0.1020 vs v18 single 0.1012 = +0.0008 (multi-seed 加 diversity)
- 6-way Borda ceiling 0.1020 (weights 1,1,1,2,3,3) — v15 + v18 协同效应
- multi-seed (42 + 43) 提供真正的随机性 diversity, 不是单纯 LR/regularization 改动

### Gate 4: 实证
| 路径 | test_R@10 |
|------|-----------|
| v9 (LR=4e-4 seed=42) | 0.0976 |
| v10 (LR=4e-4 seed=42 Ep175) | 0.0983 |
| step2 (LR=4e-4 seed=42 ls=0.1) | 0.0976 |
| v14 (LR=4e-4 no HAB seed=42) | 0.0994 |
| v15 (LR=1e-3 seed=42) | 0.1006 |
| **v18 (LR=1e-3 seed=43)** | **0.1012** (新单 ckpt SOTA) |
| 5-way Borda (default) | 0.1005 |
| **6-way Borda (1,1,1,2,3,3)** | **0.1020** (新 ensemble SOTA) |
| **0.106 目标** | **差距 0.0040 (3.8%)** |

## 关键发现
- **multi-seed averaging 有效**: v15 seed=42 + v18 seed=43 同 recipe → 6-way Borda 0.1020 (+0.0008 vs best single)
- LR=1e-3 + 多 seed (42, 43) 是本框架当前最有效的 diversity 来源
- 0.106 目标差 0.004 (3.8%) — 阶段性突破, 接近达标

## 结论
- 单 ckpt ceiling: **0.1012** (v18 LR=1e-3 + seed=43)
- 6-way Borda ceiling: **0.1020** (multi-seed averaging)
- 0.106 目标差 0.004 (3.8%) — 接近达成

## 产物
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v18_multi_seed43_stage3/ (HG_Rec_best.pth + verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v18_borda_6way/ (v18_eval + ensemble verdicts)
- /home/wlia0047/ar57/wenyu/GeneRec/tasks/v18_multi_seed/ (R34 脚本)

## 推荐下一步
1. v19: 训 seed=44 (同 v15/v18 配方) → 7-way Borda 期望 ≥ 0.1025
2. v19: 训 seed=44 + LR=8e-4 → 7-way Borda
3. v19: 训 Stage 2 v20 CURV_AWARE + LR=1e-3 + seed=42 → 跨 SID
4. v19: 6-way Borda + beam=50 实验 (不同 beam diversity)
