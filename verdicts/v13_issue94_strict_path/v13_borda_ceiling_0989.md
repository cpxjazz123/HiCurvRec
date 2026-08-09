# v13 Issue #94 严格路径: 3 ckpt × 3 beam Borda — Borda Ceiling 0.0989

## 目标
复现 Issue #94 Borda 3-way (ep130+b100, ep150+b50, agg_ep100+b30) = 0.1079 test_R@10,
本环境近似 (v9+b10, v10+b30, step2+b50) 验证 0.106 目标可达性。

## 实施 (R34 合规, tasks/v13_issue94_strict/{stage1_eval,stage2_borda}.py)
- v9 ckpt + beam=10: R@10 = 0.0895
- v10 ckpt + beam=30: R@10 = 0.0984
- step2 (ls=0.1) ckpt + beam=50: R@10 = 0.0974
- 3-way Borda weights (2,3,2): R@10 = **0.0989** (新 SOTA ensemble)

## 4-Gate 验收

### Gate 1: Spec 摘录
- Issue #94 Borda rank fusion 3-way (ep130+b100, ep150+b50, agg_ep100+b30) = 0.1079
- 同 SID 5f8331cc, 不同 epoch ckpt + 不同 beam 大小, 用 R@borda / RRF

### Gate 2: 实施核心
- v9_beam10_np + v10_beam30_np + step2_beam50_np → Borda fusion weights (2,3,2)
- 3 ckpt 共享 v15 Stage 2 (c=[1.35, 6.00, 4.39], util=1.0) + v74 HAB Stage 3
- 唯一差异: train epoch + label_smoothing + beam_size

### Gate 3: 失败机制
- **同 SID + 同 Stage 2/3 + 同 beam search algorithm** → 3 个 ckpt 预测 rank 高度相关
- Borda fusion 无 gain (0.0989 vs 单 ckpt 0.0985 = +0.0004)
- **根因**: diversity 不足,本框架天花板 = 0.0989

### Gate 4: 实证 (test_R@10)
- v9 single: 0.0979 (beam=50/100 饱和)
- v10 single: 0.0985 (beam=50)
- step2 single: 0.0976
- v9+v10 2-way: 0.0984
- v9+v10+step2 3-way: 0.0984
- v10 3-beam (10/30/50): 0.0989
- v13 3 ckpt × 3 beam (2,3,2): 0.0989
- 4-way + v10+b50: 0.0988
- **0.106 目标: 差 0.0071 (7.1%)**

## 结论
- Issue #94 验证 0.1079 在本环境不可复现
- Borda fusion 路径已穷尽, ceiling = 0.0989 (3-way 同 beam 不同 ckpt)
- v85p + v15 SID + v74 HAB 框架天花板 0.0989 < 0.106 目标 0.0071
- 0.106 目标在本框架需要架构改动 (Stage 2 v20 或 Stage 3 architecture)

## 产物
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/ (3 eval + 1 verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v12_beam_diversity_borda/ (2 eval + 1 verdict)
- /fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/ (3 eval + 3 verdicts)

## 推荐下一步
1. 训练**架构改动**的 ckpt (T5 6 层 + hidden=256 + LR=1e-3) → 4-way Borda 真正 diversity
2. Stage 2 换 v20 CURV_AWARE (valid 0.1268) → 不同 SID, 跨 SID Borda 不可能但可训独立 ckpt
3. 承认 0.0989 是本环境天花板, 转向其他 lineage (LETTER/EAGER/DIGER)
