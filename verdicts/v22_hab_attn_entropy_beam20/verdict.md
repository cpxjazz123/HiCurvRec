# v22_hab_attn_entropy R37 回退 verdict (2026-08-10)

## R37 决策行
**v22 比 v18 差 (-0.0012, -1.2%), test_R@10=0.0999 < v18 test_R@10=0.1011, 触发 R37 回退至 v18 重新创新**

## 用户硬约束 (R35 + R36 + R37 + R38)
- R35: 单 ckpt + beam_search=20 ✓
- R36: 禁调参 → 必走曲率机制改善 ✓ (entropy_weight 是新曲率正则项, R36 列出的明确合规方向)
- R37: 新版本 regress → 立即终止 lineage ✓ (本 verdict 执行)
- R38: 训练期 mid-training regress → 未触发 (训练最终 best valid 0.1272 高于 v18 0.1256)

## 实施 (R36+R37 双重合规)
- Stage 3: v18 recipe + **启用 --hab_attn_entropy_weight 0.001** (新曲率正则项, R36 合规方向)
- Stage 4: `tasks/v22_hab_attn_entropy_from_v18/stage4_beam20.py`, beam_size=20 ✓
- 不引入可学习参数 (R36 v20 教训严格化, entropy_weight 是固定 scalar loss 权重)

## Stage 3 训练
- 启动 00:23:31 → 训练自然结束 01:05 (~41 min, 200 epoch)
- **best valid = 0.1272** (Ep175) vs v18 best `0.1256`, **+0.0016** (训练端破 v18)
- **Best chain**: Ep55 0.1254 → Ep80 0.1260 → Ep95 0.1263 → Ep140 0.1268 → Ep175 0.1272
- Loss 5.64 → 2.60 (健康下降+plateau)
- HG_Rec_best.pth 存于 Ep175 best=0.1272 (timestamp 01:00:02)

## Stage 4 评估 (R37 触发)
```
test_R@5  = 0.0804
test_R@10 = 0.0999  ← 比 v18 (0.1011) 差 -0.0012 (-1.2%)
test_R@20 = 0.1224
NDCG@5    = 0.0679
NDCG@10   = 0.0741
NDCG@20   = 0.0798
n_eval    = 24772
t_eval_s  = 34.1
```

## Gate 决策

### Gate 1 (Stage 3 ckpt 质量)
- best valid_R@10 = 0.1272 (高于 v18 0.1256, +0.0016)
- 训练曲线: 持续上升 Ep55→Ep175 (健康)
- **PASS** (valid 端破 v18)

### Gate 2 (Stage 4 完整性)
- raw_predictions.npz shape=(24772, 20, 4) ✓
- eval_test.json 完整 ✓
- t_eval_s=34.1s ✓
- **PASS**

### Gate 3 (R37 历史对比)
| 版本 | valid_R@10 | test_R@10 | gap | 路径 | R36 |
|------|-----------|-----------|-----|------|-----|
| baseline | 0.1267 | 0.1024 | -0.0243 | HG-Rec default | — |
| v18_branch_quantile | 0.1256 | **0.1011** | -0.0245 | branch κ | ✓ |
| **v22_hab_attn_entropy** | **0.1272** | **0.0999** | **-0.0273** | entropy 0.001 | ✓ (但 regress) |

**R37 触发**: v22 比 v18 test 差 -0.0012 → 立即终止 lineage
**valid-test gap**: -0.0273 (v18 -0.0245), **gap 增大 +0.0028**, 暗示 entropy_weight 引入**轻微 valid 偏置**

### Gate 4 (R18 4 维度)
- D1 spec: 同 SID + 同 backbone + 同 LR=1e-3 + 同 EARLY_STOP=30
- D2 实施: **新增 --hab_attn_entropy_weight 0.001** (新曲率正则项, R36 合规方向)
- D3 Gate 1: best valid 0.1272 (高于 v18 0.1256, +0.0016)
- D3 Gate 1 反差: test 反向退化 -0.0012, valid-test gap 增大
- D4 引用: Issue #86/v87/v78 已有 entropy_weight 实现
- **FAIL** (D3 test 反向 + valid-test gap 增大: entropy_weight 引入轻微 valid 偏置)

## R36 反思 (v20+v21+v22 三重教训)
- v20 引入可学习 MLP → 严重过拟合 valid (test -38%)
- v21 仅换 strategy (kmeans vs quantile) → 与 v18 几乎持平 (test -2.6%)
- v22 启用 entropy_weight → valid +0.0016 但 test -0.0012 (gap +0.0028)
- **R36 严格化**: 即使不引入可学习参数, **新正则项也可能引入 valid 偏置**, test 退化
- **关键洞察**: 训练端 valid 上升不代表 test 上升 (过拟合警告)
- **后续方向** (R36+R37 强化版):
  - Stage 2 κ_l 后处理校准 (需要 code patch)
  - Poincaré/Minkowski 距离度量替换 (需要 code patch, 几何变换)
  - Stage 3 κ frozen→learnable (需要 code patch, R36 列出方向)
  - Stage 2 重训不同 K 配置 (大改)
  - **避免**: 类似 entropy_weight 的训练端 regularizer (轻微过拟合风险)

## R35+R36+R37+R38 严守
- R35: 单 ckpt + beam=20 ✓
- R36: entropy_weight 是 R36 列出的新曲率正则项 ✓ (但 test 退化)
- R37: 失败即终止 lineage ✓ (本 verdict 执行)
- R38: 训练期 mid-training regress 早停 — 未触发 (训练端 valid 持续上升)

## 产物路径 (留作记录, 不作下版本起点)
- ckpt (Ep175 best): `/fs04/ar57/wenyu/GeneRec/taskA/_history/v22_hab_attn_entropy_stage3/HG_Rec_best.pth`
- raw_predictions: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v22_hab_attn_entropy_beam20_eval/raw_predictions.npz`
- eval_test: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v22_hab_attn_entropy_beam20_eval/eval_test.json`
- stage3 脚本: `tasks/v22_hab_attn_entropy_from_v18/stage3.py`
- stage4 脚本: `tasks/v22_hab_attn_entropy_from_v18/stage4_beam20.py`

## 下一步 (R36+R37 严格化 v2)
- 当前最优基线: **v18_branch_curvature test_R@10=0.1011**
- v23 方向 (R36 严格化: 避免 valid 偏置, 走几何变换):
  - v23_minkowski_distance_from_v18: 距离度量替换 (代码 patch, 纯几何)
  - v23_stage2_kappa_retrain_from_v18: Stage 2 重训不同 K (大改)
  - v23_kappa_learnable_from_v18: Stage 3 κ frozen→learnable (R36 明确方向)
  - v23_hab_curvature_floor_from_v18: 几何曲率下限 (防止过拟合方向)
- **不选**: 训练端 regularizer 类 (entropy_weight 类似, valid-test gap 增大风险)