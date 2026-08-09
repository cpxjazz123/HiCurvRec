# v14 单 ckpt beam=20 verdict (2026-08-09)

## 用户硬约束 (R35)
- 必须只能使用单 checkpoint
- beam_search = 20
- 绝对禁止 Borda Rank Fusion / 任何 ensemble

## 实施
- Stage 3: 纯 T5 (no HAB) + WD=0.01 + dropout=0.2 + label_smoothing=0.05
- Recipe 区别: 关闭 HAB 但保留 v74 的 WD + dropout (Issue #138 v74 三改动)
- Stage 4: `tasks/v14_pureT5_noHAB/stage4_beam20.py`, beam_size=20

## 结果
```
test_R@5  = 0.0793
test_R@10 = 0.0991  ← 主要指标
test_R@20 = 0.1196
NDCG@5    = 0.0670
NDCG@10   = 0.0733
NDCG@20   = 0.0785
n_eval    = 24772
t_eval_s  = 33.7
```

## Gate 决策 (4 维度 R18)

### Gate 1 (Stage 3 ckpt 质量)
- Ep134 best valid_R@10 = 0.1237
- 训练曲线健康: loss 2.78-2.79 平台,R@10 单调上升 0.0805 → 0.1237
- **PASS**

### Gate 2 (Stage 4 评估完整性)
- raw_predictions.npz shape=(24772, 20, 4) ✓
- eval_test.json 完整字段 ✓
- t_eval_s=33.7 (单 GPU eval, 合理)
- **PASS**

### Gate 3 (与历史对比)
| 版本 | test_R@10 | 路径 |
|------|-----------|------|
| baseline Task #84 | 0.1024 | HG-Rec default |
| v13 Borda 3-way ceiling | 0.0989 | **已禁** |
| v14 单 ckpt beam=30 | 0.0994 | 旧 Borda 路径 |
| **v14 单 ckpt beam=20** | **0.0991** | **R35 强约束** |
| 目标 | 0.106 | 差距 -0.0069 (-6.9%) |

- 单 ckpt **超过** Borda ceiling 0.0989 (+0.0002)
- 未达目标 0.106 (-0.0069)

### Gate 4 (路径对比 R18 4 维度)
- D1 spec: 同 SID (5f8331cc...) + 同 backbone (sentence-t5-base)
- D2 实施: Stage 3 关闭 HAB 是**新维度**,区别于 v9/v10/step2 (全 HAB)
- D3 Gate 1 失败机制: 不适用 (Gate 1 PASS)
- D4 引用: Issue #94 (HAB ceiling) + Issue #138 (WD/dropout 协同)
- **PASS** (实施有差异,但天花板未突破)

## NO-GO 判定 (R5)
- 目标 test_R@10=0.106, 实际 0.0991, 差距 6.9%
- **NO-GO** (未达目标),但**有意义产物**:
  - 单 ckpt 已超 v13 Borda ceiling
  - 验证 R35 强约束可行性 (单 ckpt + beam=20)
  - 纯 T5 路线天花板 = 0.0991

## 下一步 (R11 + R19 推荐)
- v15_highLR 训练中 (v74 HAB + LR=1e-3, 不同 optimizer trajectory)
- 预期训完 ~20:47 + ~20 min = ~21:07
- v15_highLR 训完 → 跑 stage4_beam20.py → 单 ckpt + beam=20 评估

## 产物路径
- ckpt: `taskA/_history/v14_pureT5_stage3/HG_Rec_best.pth` (Ep134 best)
- raw_predictions: `taskA/_history/v14_pureT5_beam20_eval/raw_predictions.npz`
- eval_test: `taskA/_history/v14_pureT5_beam20_eval/eval_test.json`
- stage4 脚本: `tasks/v14_pureT5_noHAB/stage4_beam20.py`