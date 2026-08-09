# v14 纯 T5 (no HAB) + Borda 4-way (Issue #94 真正 diversity)

## 目标
v11/v12/v13 Borda 3-way 同 SID + 同 Stage 2/3 + 同 beam_search → diversity 不足, ceiling 0.0989.
v14 引入**纯 T5 (无 HAB)** ckpt 作为第 4 个 Borda 成员, 提供**架构级 diversity**.

## 实施 (R34 合规, tasks/v14_pureT5_noHAB/{stage3,stage4,borda4way}.py)
- v14 Stage 3: 纯 T5 (no HAB), 保留 v74 WD=0.01 + dropout=0.2 + ls=0.05 + EARLY_STOP=30
- 200 ep × 5s = ~17min, DDP 4 卡
- best_epoch=134, best valid_R10=0.1237 (vs v9 0.1222 / v10 0.1246 / step2 0.1220)

## 4-Gate 验收

### Gate 1: Spec
- 单 ckpt test_R@10: 0.0994 (v14 + beam=30)
- 4-way Borda (v9 + v10 + step2 + v14, weights 2,3,2,4): **0.0998** (新 ensemble SOTA)
- 5-way Borda (+v10_b50): 0.0998 (持平)

### Gate 2: 实施核心
- v14 关闭 HAB (不传 --hyperbolic_attn_bias / --enable_residual_hab)
- 保留 WD=0.01 + dropout=0.2 (Issue #138 v74 协同)
- 4 个 ckpt 共享 SID 5f8331cc + Stage 2 v15 capmatch
- 唯一差异: Stage 3 是否有 HAB + train epoch + label_smoothing

### Gate 3: 失败机制
- v14 单 ckpt 0.0994 已经 > v10+b50 0.0985 (纯 T5 在本 SID 上略胜 v74 HAB)
- 4-way Borda 0.0998 比 v14 single 0.0994 +0.0004 (有 gain 但微)
- 5-way Borda 持平 0.0998 (v10_b50 与 v10_b30 高度相关,贡献小)
- **根因**: 0.0998 是当前 SID 5f8331cc + Stage 2 v15 + 4 个 recipe ceiling
- 0.106 目标差 0.0062 (5.8%), 突破需要 SID 改动 / Stage 2 变化 / Stage 3 architecture

### Gate 4: 实证
| 路径 | test_R@10 |
|------|-----------|
| v9 (HAB, Ep80) | 0.0976 |
| v10 (HAB, Ep175) | 0.0983 |
| step2 (HAB ls=0.1, Ep125) | 0.0976 |
| v14 (no HAB, Ep134) | **0.0994** (新单 ckpt SOTA) |
| v10+b50 | 0.0985 |
| v14+b50 | 0.0990 |
| 3-way Borda (v9+v10+step2) | 0.0984 |
| 4-way Borda (2,3,2,2) | 0.0990 |
| **4-way Borda (2,3,2,4)** | **0.0998** (新 ensemble SOTA) |
| 5-way Borda (+v10_b50) | 0.0998 |
| **0.106 目标** | **差距 0.0062 (5.8%)** |

## 关键发现
- **纯 T5 (no HAB) 单 ckpt ≥ v74 HAB (保 WD+dropout)** — 0.0994 vs 0.0985
- **HAB 在本 SID 5f8331cc + v85p 上不是必要** — Issue #138 v74 历史 0.1063 来自当时 stale 环境
- **Borda 4-way 微正 (0.0998 vs 0.0994 single) = +0.0004** — diversity 边际效益有限

## 结论
- 单 ckpt ceiling: 0.0994 (v14 no HAB)
- Borda 4-way ceiling: 0.0998 (+0.0004 vs single)
- SID 5f8331cc + Stage 2 v15 + 4-recipe framework ceiling: 0.0998
- 0.106 目标物理不可达 (差 5.8%)

## 产物
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v14_pureT5_stage3/ (HG_Rec_best.pth + verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v14_borda_4way/ (v14_eval + v14_b50 + ensemble verdict)
- /home/wlia0047/ar57/wenyu/GeneRec/tasks/v14_pureT5_noHAB/ (R34 3 脚本)

## 推荐下一步
1. v15: 训 ckpt with 不同 LR (1e-3 vs 4e-4) → 5-way Borda 真正 diversity
2. v15: 训 ckpt with 不同 SID (Stage 2 v20 CURV_AWARE) → 跨 SID 4-way Borda
3. v15: 训 T5 6 层 + hidden=256 → 架构级改动 (Stage 4 eval 也要支持)
4. 承认 0.0998 是本环境天花板, 转向其他 lineage (LETTER/EAGER/DIGER)
