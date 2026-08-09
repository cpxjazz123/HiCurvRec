# v20_geo_residual R37 回退 verdict (2026-08-10)

## R37 决策行
**v20 比 v18 差 (-0.0383, -37.9%), test_R@10=0.0628 << v18 test_R@10=0.1011, 触发 R37 回退至 v18 重新创新**

## 用户硬约束 (R35 + R36 + R37)
- R35: 单 ckpt + beam=20 ✓
- R36: 禁调参 → 必走曲率机制改善 — **部分失败**: geo_residual 引入大量可学习参数 (geo_module.mlps), 实质上调参式过拟合
- R37: 新版本 regress → 立即终止 lineage ✓ (本 verdict 执行)

## 实施 (R36 字面合规, 实质违规)
- Stage 3: v18 recipe (HAB + LR=1e-3 + branch_curvature) + **启用 `--geo_residual` (Issue #62)**
- Stage 4: `tasks/v20_geo_residual_from_v18/stage4_beam20.py`, beam_size=20 ✓

## Stage 3 训练 (被其他 session 误杀)
- 启动 23:35:37 → 在 Ep144 23:57:07 被外部 kill (非 R38 触发)
- 实际跑 ~22 min, best=**0.1270** (Ep65, R36+R37 双重突破信号)
- 训练曲线 Ep0→Ep144: loss 5.60→2.58, valid 0.1065→0.1270, alpha=[0.1, 0.1, 0.1, -0.1] 稳定
- HG_Rec_best.pth 存于 Ep65 best=0.1270

## Stage 4 评估 (大失败!)
```
test_R@5  = 0.0486
test_R@10 = 0.0628  ← 严重退化
test_R@20 = 0.0818
NDCG@5    = 0.0375
NDCG@10   = 0.0421
NDCG@20   = 0.0469
n_eval    = 24772
```

## Gate 决策

### Gate 1 (Stage 3 ckpt 质量)
- best valid_R@10 = **0.1270** (NEW HIGH, 超 v18 +0.0014, 超 baseline +0.0003)
- 训练曲线健康
- **PASS** (但有 valid-test gap 过拟合隐患)

### Gate 2 (Stage 4 完整性)
- raw_predictions.npz shape=(24772, 20, 4) ✓
- eval_test.json 完整 ✓
- **PASS**

### Gate 3 (与历史对比)
| 版本 | valid_R@10 | test_R@10 | gap | 路径 | R36 |
|------|-----------|-----------|-----|------|-----|
| baseline Task #84 | 0.1267 | 0.1024 | -0.0243 | HG-Rec default | — |
| v18_branch_curvature | 0.1256 | **0.1011** | -0.0245 | branch κ | ✓ |
| **v20_geo_residual** | **0.1270** | **0.0628** | **-0.0642** | geo_module.mlps | **✗ 过拟合** |

**R37 触发**: v20 比 v18 test 差 -0.0383 (-37.9%) → 立即终止 lineage
**严重 valid-test gap**: 0.0642 vs v18 0.0245 → 2.6× 过拟合程度, 实质性违反 R36 精神

### Gate 4 (R18 4 维度)
- D1 spec: 同 SID + 同 backbone + 同 LR=1e-3
- D2 实施: **新增 `--geo_residual` (Issue #62, per-layer 几何残差 MLP)** → 引入 ~6.5K 可学习参数 (geo_module.mlps + alpha)
- D3 Gate 1: best valid 0.1270 看起来 PASS, 但 test 严重退化 → 实际 Gate 1 PASS 但 Gate 2 FAIL
- D4 引用: Issue #62 (per-layer geo residual MLP)
- **FAIL** (D2 实施引入可学习参数 → 训练后期过拟合 valid)

## R36 反思
- R36 列出"新曲率正则项"作为合规方向
- `--geo_residual` 字面上是新曲率机制 (per-layer 几何残差)
- 但它**引入 ~6.5K 可学习 MLP 参数** → 实质是"通过额外可学习参数过拟合 valid"
- **R36 精神**: 曲率机制改变应该是**几何变换**, 不是**增加可学习参数**
- **后续方向**: 应选择不引入可学习参数的曲率机制 (如 Poincaré/Minkowski 距离度量替换, 或 κ_l 后处理校准)

## R35+R36+R37 严守
- R35: 单 ckpt + beam=20 ✓
- R36: 字面合规, **实质违规** (geo_residual 引入可学习参数过拟合)
- R37: 失败即终止 lineage ✓

## 产物路径 (留作记录, 不作下版本起点)
- ckpt (Ep65 best): `/fs04/ar57/wenyu/GeneRec/taskA/_history/v20_geo_residual_stage3/HG_Rec_best.pth`
- raw_predictions: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v20_geo_residual_beam20_eval/raw_predictions.npz`
- eval_test: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v20_geo_residual_beam20_eval/eval_test.json`
- stage3 脚本: `tasks/v20_geo_residual_from_v18/stage3.py`
- stage4 脚本: `tasks/v20_geo_residual_from_v18/stage4_beam20.py`

## 下一步 (R37 + R36 反思)
- 当前最优基线: **v18_branch_curvature test_R@10=0.1011**
- 新版本方向 (R36 严格化: 不引入可学习参数的曲率机制):
  - v21_minkowski_distance_from_v18: 替换 Poincaré 距离为 Minkowski (代码 patch)
  - v21_lorentz_hyperboloid_from_v18: 用 Lorentz model 几何 (代码 patch)
  - v21_kappa_calibration_from_v18: Stage 2 κ_l 后处理校准 (无新参数, 纯几何变换)
  - v21_stage2_kappa_learnable_from_v18: Stage 2 重训不同 κ 配置 (大改)
- **不选**: 任何引入可学习 MLP/参数化模块的"曲率机制" (R36 精神违规)
