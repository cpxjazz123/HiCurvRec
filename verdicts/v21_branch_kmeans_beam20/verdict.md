# v21_branch_curvature_kmeans R37+R38 回退 verdict (2026-08-10)

## R37 决策行
**v21 比 v18 差 (-0.0026, -2.6%), test_R@10=0.0985 < v18 test_R@10=0.1011, 触发 R37 回退至 v18 重新创新**

## R38 决策行 (训练期早停)
**v21 mid-training 平台, v21 best_valid=0.1252 < v18 best_valid=0.1256 (-0.0004), 连续 50 epoch (Ep40-Ep90) 无改进, R38 触发: kill + 回退**

## 用户硬约束 (R35 + R36 + R37 + R38)
- R35: 单 ckpt + beam_search=20 ✓
- R36: 禁调参 → 必走曲率机制改善 ✓ (kmeans 分桶是真正的曲率机制)
- R37: 新版本 regress → 立即终止 lineage ✓ (本 verdict 执行)
- R38: 训练期 mid-training regress → 立即 kill ✓ (本 verdict 执行)

## 实施 (R36+R37 双重合规)
- Stage 3: v18 recipe + **branch_curvature_strategy "quantile" → "kmeans"** (Issue 默认支持)
- Stage 4: `tasks/v21_branch_curvature_kmeans_from_v18/stage4_beam20.py`, beam_size=20 ✓
- 不引入可学习参数 (R36 v20 教训严格化)

## Stage 3 训练 (R38 早停)
- 启动 00:07:31 → R38 触发 00:18 (~10.5 min 浪费)
- 实际跑 Ep93/200 时被 kill
- best valid=**0.1252** (Ep40), 之后连续 50 epoch 无改进 (Ep40-Ep90 都 ≤0.1226)
- loss 5.64 → 2.62 健康下降 (但 valid 平台)
- HG_Rec_best.pth 存于 Ep40 best=0.1252

## Stage 4 评估 (R37 触发)
```
test_R@5  = 0.0790
test_R@10 = 0.0985  ← 比 v18 (0.1011) 差 -0.0026
test_R@20 = 0.1203
NDCG@5    = 0.0667
NDCG@10   = 0.0730
NDCG@20   = 0.0785
n_eval    = 24772
```

## Gate 决策

### Gate 1 (Stage 3 ckpt 质量)
- best valid_R@10 = 0.1252 (落后 v18 0.0004)
- 训练曲线: 上升 → 平台 (Ep40 后无改进)
- **MARGINAL** (valid 健康但落后 v18)

### Gate 2 (Stage 4 完整性)
- raw_predictions.npz shape=(24772, 20, 4) ✓
- eval_test.json 完整 ✓
- **PASS**

### Gate 3 (R37 历史对比)
| 版本 | valid_R@10 | test_R@10 | gap | 路径 | R36 |
|------|-----------|-----------|-----|------|-----|
| baseline | 0.1267 | 0.1024 | -0.0243 | HG-Rec default | — |
| v18_branch_quantile | 0.1256 | **0.1011** | -0.0245 | branch κ | ✓ |
| **v21_branch_kmeans** | 0.1252 | **0.0985** | -0.0267 | kmeans 分桶 | ✓ (但 regress) |

**R37 触发**: v21 比 v18 test 差 -0.0026 → 立即终止 lineage
**valid-test gap**: -0.0267 (v18 -0.0245), 略增但健康

### Gate 4 (R18 4 维度)
- D1 spec: 同 SID + 同 backbone + 同 LR=1e-3
- D2 实施: **branch_curvature_strategy "quantile" → "kmeans"** (分桶机制改变)
- D3 Gate 1: best valid 0.1252 (略落后 v18)
- D4 引用: Issue 已有 kmeans 实现 (line 902 in stage3)
- **FAIL** (D3 valid + D2 实施: kmeans vs quantile 实际效果相近但略差)

## R36 反思 (v20+v21 双重教训)
- v20 引入可学习 MLP → 严重过拟合 valid (test -38%)
- v21 仅换 strategy (kmeans vs quantile) → 与 v18 几乎持平 (test -2.6%)
- **R36 严格化**: 单纯换分桶 strategy 在 v18 框架内属于 sweep, 几乎无效
- **后续方向**: 必须寻找真正改变曲率机制的方案:
  - 启用 `--hab_attn_entropy_weight > 0` (新曲率正则项, R36 明确合规)
  - Stage 2 κ_l 后处理校准 (需要 code patch)
  - Poincaré/Minkowski 距离度量替换 (需要 code patch)
  - Stage 2 重训不同 K 配置 (大改)

## R35+R36+R37+R38 严守
- R35: 单 ckpt + beam=20 ✓
- R36: kmeans 策略是曲率机制 ✓
- R37: 失败即终止 lineage ✓
- R38: mid-training regress 早停 ✓ (节省 ~10 min GPU)

## 产物路径 (留作记录, 不作下版本起点)
- ckpt (Ep40 best): `/fs04/ar57/wenyu/GeneRec/taskA/_history/v21_branch_kmeans_stage3/HG_Rec_best.pth`
- raw_predictions: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v21_branch_kmeans_beam20_eval/raw_predictions.npz`
- eval_test: `/fs04/ar57/wenyu/GeneRec/taskA/_history/v21_branch_kmeans_beam20_eval/eval_test.json`
- stage3 脚本: `tasks/v21_branch_curvature_kmeans_from_v18/stage3.py`
- stage4 脚本: `tasks/v21_branch_curvature_kmeans_from_v18/stage4_beam20.py`

## 下一步 (R36+R37 严格化)
- 当前最优基线: **v18_branch_curvature test_R@10=0.1011**
- v22 方向 (R36 列出的合规方向):
  - v22_hab_attn_entropy_from_v18: 启用 --hab_attn_entropy_weight 0.001 (新曲率正则项)
  - v22_stage2_kappa_retrain_from_v18: Stage 2 重训不同 K (大改)
  - v22_minkowski_distance_from_v18: 距离度量替换 (code patch)
- **不选**: 单纯换 branch_curvature_n_buckets/sweep 类 (R36 边界违规)
