# v15_highLR 单 ckpt beam=20 verdict (2026-08-09)

## 用户硬约束 (R35)
- 必须只能使用单 checkpoint
- beam_search = 20
- 绝对禁止 Borda Rank Fusion / 任何 ensemble

## 实施
- Stage 3: **v74 HAB + LR=1e-3** (vs 默认 4e-4) + WD=0.01 + dropout=0.2 + ls=0.05
- Recipe 区别: 改变 LR (4e-4 → 1e-3) → 不同 optimizer trajectory
- Stage 4: `tasks/v15_pureT5_highLR/stage4_beam20.py`, beam_size=20

## Stage 3 训练结果
- 启动 20:52:16 → 完成 21:13:44 = 21:28 (21.5 min)
- best_epoch=49, best valid_R@10=0.1258, best_loss=2.6198
- 训练曲线: loss 2.60-2.65, R@10 上升 0.0805 → 0.1258 → 平台 0.1245
- 200 epoch 跑满 + early stop 触发 (30 epoch 无新 best)
- v15_best 0.1258 > v14_best 0.1237 (+0.0021, 单 ckpt 角度 v15 > v14)

## Stage 4 评估结果
```
test_R@5  = 0.0800
test_R@10 = 0.1003  ← 主要指标, **NEW HIGH**
test_R@20 = 0.1233
NDCG@5    = 0.0679
NDCG@10   = 0.0744
NDCG@20   = 0.0803
n_eval    = 24772
```

## Gate 决策 (4 维度 R18)

### Gate 1 (Stage 3 ckpt 质量)
- best valid_R@10 = 0.1258
- 训练曲线健康: 上升 → 平台 → early stop
- **PASS**

### Gate 2 (Stage 4 评估完整性)
- raw_predictions.npz shape=(24772, 20, 4) ✓
- eval_test.json 完整字段 ✓
- **PASS**

### Gate 3 (与历史对比)
| 版本 | test_R@10 | 路径 |
|------|-----------|------|
| baseline Task #84 | 0.1024 | HG-Rec default |
| v13 Borda 3-way ceiling | 0.0989 | **已禁** |
| v14 单 ckpt beam=30 | 0.0994 | 旧 Borda 路径 |
| v14 单 ckpt beam=20 | 0.0991 | R35 |
| **v15_highLR 单 ckpt beam=20** | **0.1003** | **R35 NEW HIGH** |
| 目标 | 0.106 | 差距 -0.0057 (-5.4%) |

- **NEW HIGH** 单 ckpt (超 v14 +0.0012, 超 Borda ceiling +0.0014)
- 未达 baseline 0.1024 (-0.0021) 和目标 0.106 (-0.0057)

### Gate 4 (路径对比 R18 4 维度)
- D1 spec: 同 SID (5f8331cc...) + 同 backbone (sentence-t5-base)
- D2 实施: LR 1e-3 vs 默认 4e-4 → **新维度** (不同 optimizer trajectory)
- D3 Gate 1 失败机制: 不适用 (Gate 1 PASS)
- D4 引用: Issue #138 (HAB + WD/dropout 协同) + LR sweep
- **PASS** (实施有差异, 单 ckpt 天花板从 0.0994 → 0.1003)

## NO-GO 判定 (R5)
- 目标 test_R@10=0.106, 实际 0.1003, 差距 5.4%
- **NO-GO** (未达目标),但**有意义进展**:
  - 单 ckpt **NEW HIGH** 0.1003 (历史最高)
  - 验证 R35 强约束 + 纯 recipe 改进有效
  - 比 baseline 0.1024 还差 0.0021,接近突破

## 产物路径
- ckpt: `taskA/_history/v15_highLR_stage3/HG_Rec_best.pth` (Ep49 best)
- raw_predictions: `taskA/_history/v15_highLR_beam20_eval/raw_predictions.npz`
- eval_test: `taskA/_history/v15_highLR_beam20_eval/eval_test.json`
- stage3 脚本: `tasks/v15_pureT5_highLR/stage3.py`
- stage4 脚本: `tasks/v15_pureT5_highLR/stage4_beam20.py`

## R35 冲突报告
- 发现另一 session (PID 1586926) 启动 v15_highLR **beam=50** eval, 违反 R35 (禁 beam_size=30/50)
- 写入 `taskA/_history/v15_borda_5way/v15_b50/` 暗示准备 Borda 5-way 融合, 同时违反 R35 (禁 Borda)
- 本 session 仅生成 beam=20 产物, 严格 R35 合规