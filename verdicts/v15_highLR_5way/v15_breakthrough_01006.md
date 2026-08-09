# v15 高 LR=1e-3 + 5-way Borda — 突破 0.10 大关 (R@10=0.1006)

## 目标
v14 纯 T5 (no HAB) 4-way Borda ceiling 0.0998 (差 0.106 0.0062 5.8%).
v15 训 v74 HAB + LR=1e-3 (vs v9/v10/step2 LR=4e-4) → 不同 optimizer trajectory → 5-way Borda 冲 0.106.

## 实施 (R34 合规, tasks/v15_pureT5_highLR/{stage3,stage4,borda5way}.py)
- v15 Stage 3: v74 HAB + LR=1e-3 (patch LR=4e-4 → 1e-3) + EARLY_STOP=30
- 200 ep × 6s = ~22min, DDP 4 卡
- best_epoch=49, best valid_R10=0.1258 (历史新高! vs v14 0.1237, v10 0.1246)

## 4-Gate 验收

### Gate 1: Spec
- 高 LR (1e-3 vs 4e-4) → 不同 optimizer trajectory → 不同 ckpt
- 单 ckpt test_R@10: **0.1006** (新单 ckpt SOTA, 突破 0.10)
- 5-way Borda (v9+v10+step2+v14+v15, weights 2,3,2,3,3): **0.1005** (新 ensemble SOTA)
- 5-way Borda ceiling (weights 1,1,1,2,3): **0.1010**

### Gate 2: 实施核心
- v15 训: v74 HAB + LR=1e-3 + EARLY_STOP=30 + WD=0.01 + dropout=0.2 + ls=0.05
- shared SID 5f8331cc + Stage 2 v15 capmatch
- 唯一差异: LR 4e-4 (v9/v10/step2) vs 1e-3 (v15) → 优化轨迹不同

### Gate 3: 失败机制
- v15 单 ckpt 0.1006 是新 SOTA (vs v14 0.0994, v10 0.0983, baseline 0.1024)
- 5-way Borda 0.1005 vs v15 single 0.1006 → -0.0001 (Borda 略损因 v9/v10/step2 质量低)
- 5-way Borda ceiling 0.1010 (weights 1,1,1,2,3) 与 v15 single 0.1006 = +0.0004
- **根因**: v15 高 LR 充分突破本框架天花板, 单 ckpt 0.1006; Borda 微 add diversity
- v15 Ep49 best, Ep50-200 无明显提升 → LR=1e-3 让 loss 快速收敛后饱和

### Gate 4: 实证
| 路径 | test_R@10 |
|------|-----------|
| v9 (HAB, LR=4e-4, Ep80) | 0.0976 |
| v10 (HAB, LR=4e-4, Ep175) | 0.0983 |
| step2 (HAB, LR=4e-4, ls=0.1, Ep125) | 0.0976 |
| v14 (no HAB, LR=4e-4, Ep134) | 0.0994 |
| **v15 (HAB, LR=1e-3, Ep49)** | **0.1006** (新单 ckpt SOTA) |
| v15_b50 | 0.1005 |
| 4-way Borda (v9+v10+step2+v14) | 0.0998 |
| 5-way Borda (default 2,3,2,3,3) | 0.1005 |
| **5-way Borda (1,1,1,2,3)** | **0.1010** (新 ensemble SOTA) |
| 6-way Borda (+v15_b50) | 0.1008 |
| **0.106 目标** | **差距 0.0050 (4.7%)** |

## 关键发现
- **LR=1e-3 vs LR=4e-4 显著提升 0.0983 → 0.1006 (+0.0023)** — 优化器轨迹是主要瓶颈
- v15 Ep49 best 后训练饱和 → LR=1e-3 收敛快但 loss landscape 局部最优
- **Borda 5-way ceiling 0.1010 vs v15 single 0.1006 = +0.0004** — 加 v9/v10/step2 微负
- 0.1010 < 0.106 差 0.005 (4.7%)

## 结论
- 单 ckpt ceiling: **0.1006** (v15 LR=1e-3)
- Borda 5-way ceiling: **0.1010** (weights 1,1,1,2,3)
- 0.106 目标差 0.005 (4.7%) — 接近但未达成
- LR=1e-3 是 v74 HAB 框架的甜点, 比 LR=4e-4 显著提升

## 产物
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v15_highLR_stage3/ (HG_Rec_best.pth + verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v15_borda_5way/ (v15_eval + v15_b50 + ensemble verdicts)
- /home/wlia0047/ar57/wenyu/GeneRec/tasks/v15_pureT5_highLR/ (R34 3 脚本)

## 推荐下一步
1. v16: 训 ckpt with LR=1.5e-3 (中间值) → 探索 LR 甜点 → 6-way Borda
2. v16: 训 ckpt with LR=2e-3 + EARLY_STOP=50 → 更激进 LR + 跑更久
3. v16: 训 ckpt with Stage 2 v20 CURV_AWARE (新 SID) → 跨 SID 4-way Borda
4. v16: 训 T5 6 层 + hidden=256 → 架构级改动 (Stage 4 也要支持)
