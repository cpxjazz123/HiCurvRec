# v16_midLR_15e3 单 ckpt beam=20 verdict (2026-08-09)

## 用户硬约束 (R35)
- 必须只能使用单 checkpoint
- beam_search = 20
- 绝对禁止 Borda Rank Fusion / 任何 ensemble

## 实施
- Stage 3: **v74 HAB + LR=1.5e-3** (vs v15 LR=1e-3 / v9/v10 LR=4e-4) + WD=0.01 + dropout=0.2 + ls=0.05
- Recipe 区别: LR sweep 第三步 (4e-4 → 1e-3 → 1.5e-3)
- Stage 4: `tasks/v16_midLR/stage4_beam20.py`, beam_size=20

## Stage 3 训练结果
- 启动 21:18 → 完成 21:39:51 = 21.5 min
- best_epoch=89, best valid_R@10=0.1232, best_loss=2.5909
- 训练曲线: loss 2.57-2.59, R@10 平台 0.1230 ± 0.001
- 200 epoch 跑满 + early stop 触发

## Stage 4 评估结果
```
test_R@5  = 0.0778
test_R@10 = 0.0974  ← 主要指标
test_R@20 = 0.1200
NDCG@5    = 0.0654
NDCG@10   = 0.0717
NDCG@20   = 0.0775
n_eval    = 24772
```

## LR Sweep 完整图 (R18 4 维度)

| 版本 | LR | valid_R@10 | test_R@10 |
|------|-----|-----------|-----------|
| v9 (HAB) | 4e-4 | 0.1220 | 0.0976 |
| v10 (HAB) | 4e-4 | 0.1246 | 0.0983 |
| **v15_highLR (HAB)** | **1e-3** | **0.1258** | **0.1003** ← WINNER |
| v16_midLR (HAB) | 1.5e-3 | 0.1232 | 0.0974 |

**结论**: LR=1e-3 是 sweet spot。LR=1.5e-3 test_R@10 反而下降 -0.0029 (过拟合?)

## Gate 决策 (4 维度 R18)

### Gate 1 (Stage 3 ckpt 质量)
- best valid_R@10 = 0.1232
- 训练曲线健康: 上升 → 平台 → early stop
- **PASS**

### Gate 2 (Stage 4 评估完整性)
- raw_predictions.npz shape=(24772, 20, 4) ✓
- eval_test.json 完整字段 ✓
- **PASS**

### Gate 3 (与历史对比)
- v16 test_R@10=0.0974 < v15 test_R@10=0.1003 (-0.0029)
- v16 < v13 Borda ceiling 0.0989 (-0.0015)
- **v16 NO-GO**: 优于 baseline 0.1024 距离反而加大
- v15 (LR=1e-3) 仍是单 ckpt 冠军

### Gate 4 (路径对比 R18 4 维度)
- D1 spec: 同 SID + 同 backbone + 同 HAB
- D2 实施: LR 1.5e-3 vs v15 1e-3 → 探索 LR 甜点
- D3 Gate 1 失败机制: 不适用 (Gate 1 PASS)
- D4 引用: LR sweep 实验
- **PASS** (实施有差异, 但 v15 更优, 暗示甜点)

## NO-GO 判定 (R5)
- 目标 test_R@10=0.106, 实际 0.0974, 差距 8.1%
- **NO-GO** (未达目标, 且**比 v15 退化**)
- 但**有意义产物**:
  - 验证 LR=1e-3 是 HAB sweet spot
  - 排除 LR=1.5e-3 进一步探索
  - 下一步应转向不同 recipe 维度 (dropout/label_smoothing/Stage 2 SID 等)

## 产物路径
- ckpt: `taskA/_history/v16_midLR_15e3_stage3/HG_Rec_best.pth` (Ep89 best)
- raw_predictions: `taskA/_history/v16_midLR_15e3_beam20_eval/raw_predictions.npz`
- eval_test: `taskA/_history/v16_midLR_15e3_beam20_eval/eval_test.json`
- stage3 脚本: `tasks/v16_midLR/stage3.py`
- stage4 脚本: `tasks/v16_midLR/stage4_beam20.py`

## R35 严格自检
- 本 session 仅 beam=20 产物: ✓
- 本 session 无任何 Borda/ensemble: ✓
- 本 session 无 beam=30/50: ✓

## 下一步推荐 (R11 + R19)
- v15 (LR=1e-3) 单 ckpt beam=20 仍是当前最优 test_R@10=0.1003
- 应转向新 recipe 维度:
  - **v17_dropout**: 调小 dropout 0.2 → 0.1 (v15 recipe 协同改动)
  - **v17_ls**: 关闭 label_smoothing (vs v15=0.05)
  - **v17_warmup**: 调长 warmup 让 LR=1e-3 更充分收敛
- 用户派工前不启动 (loop.md 没新指令)