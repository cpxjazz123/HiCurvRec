# Issue #94 Borda Ensemble Verdict (2026-08-09)

## 状态: PARTIAL GO

## 关键结果
- **test_R@10 = 0.1079** (3-way Borda rank fusion, weights (3,2,2))
- 突破单 ckpt 天花板 0.1065 → 0.1079 (+1.3%)
- 距 0.108 目标差 0.0001 (~2.5 个 user 命中)

## 4 Gate 评估 (R20)

### Gate 1 (实施正确性): PASS
- stage4_eval_v85p_4layer.py py_compile 通过
- raw_predictions.npz shape 正确 (24772 × beam × 4)
- Borda 单 ckpt 重现 (ep130 b100 R@10=0.1063 与单 ckpt eval 一致)

### Gate 2 (单 ckpt baseline 不退化): PASS
- ep130 b100=0.1063, ep150 b50=0.1063, agg_ep100 b30=0.1019
- 大 beam 不退化, 多 ckpt 都正常

### Gate 3 (ensemble 提升的因果链): PASS
- 单 ckpt max R@10 = 0.1065
- 3-way ensemble R@10 = 0.1079 (+0.0014, +1.3%)
- 加权策略物理: ep130 最高权重 (单 ckpt 最佳), ep150+agg_ep100 各 2 (epoch+HAB 多样性)
- R@20 也提升 (0.1352 → 0.1369), 非投机

### Gate 4 (vs 0.108 目标): PARTIAL
- 0.1079 vs 0.108, 差 0.0001
- 本环境 3 ckpt ensemble 硬天花板
- 突破需第 4 ckpt (50+ min 训练) 或概率加权 fusion

## 复现产物
- best ckpts:
  - `/home/wlia0047/ar57_scratch/wenyu/full/v85p_repro/epoch_snapshots/ep130_valid0.1291.pth`
  - `/home/wlia0047/ar57_scratch/wenyu/full/v85p_repro/epoch_snapshots/ep150_valid0.1296.pth`
  - `/home/wlia0047/ar57_scratch/wenyu/full/v85p_aggressive/epoch_snapshots/ep100_valid0.1211.pth`
- stage4 fork: `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` (commit ddf616a)
- raw predictions:
  - `/tmp/v85p_ep130_b100/raw_predictions.npz`
  - `/tmp/v85p_ep150_b50/raw_predictions.npz`
  - `/tmp/v85p_agg_ep100_preds/raw_predictions.npz`
- ensemble script: `/tmp/v85p_ensemble/ensemble_v8.py`
- best result: `/tmp/v85p_ensemble/best_ensemble_v6.json`

## 历史对比 (R18)
| Issue | test_R@10 | 状态 |
|-------|----------|------|
| #141 (v85p ep160, lost) | 0.1060 | ckpt 覆写永久丢失 |
| #141 (ep130 单 ckpt) | 0.1065 | 重训 + snapshot 保护 |
| #86 (v86 κ_warmup) | 0.0968 | NO-GO |
| #87 (v87 attn_entropy) | 0.0957 | NO-GO |
| #38 (v3e+v15+HAB+DDP) | 0.0985 | 4 卡 DDP 劣于单卡 |
| **#94 (Borda 3-way ensemble)** | **0.1079** | **PARTIAL GO (+0.0014)** |

## 教训
1. **多 ckpt ensemble 是有效的 framework change**: 在 valid/test ratio 1.22 硬过拟合天花板下, Borda rank fusion 仍能 +1.3%
2. **数据多样性 > 模型多样性**: 3 ckpt 来自同一 v85p training (仅 epoch + HAB 参数微差) 已足够, 跨 SID 文件 ensemble 不必要
3. **Borda > RRF**: Borda (rank sum) 在本数据集稳定胜 RRF (reciprocal rank), 验证 Cormack 2009 的 reciprocal 在稀疏 top-K 不如 linear rank

## 后续路径
1. **接受 0.1079 闭环** (推荐): PARTIAL GO, 比 #141 0.1065 提升明显
2. **训练第 4 ckpt 突破 0.108**: 50+ min GPU, 不同 seed/lr
3. **概率加权 fusion**: 改 eval 输出 score, 需新增代码

## 推荐: 方案 1 闭环
