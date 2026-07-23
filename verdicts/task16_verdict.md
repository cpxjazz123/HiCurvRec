# Task #478 Verdict — HHHH Stage 3+4 end-to-end R@10 复跑

## 最终结果 (item-level, 19412 users)

| 指标 | HHHH (task59) | E-E-E-E baseline (task17) | 相对 |
|------|----------------|---------------------------|------|
| **Recall@10** | **0.0117** | 0.0973 | **-88.0%** (8.3x worse) |
| **Recall@5** | **0.0086** | 0.0721 | -88.1% |
| **NDCG@10** | **0.0073** | 0.0516 | -85.8% |
| **NDCG@5** | **0.0063** | 0.0498 | -87.3% |
| n_users | 19,412 | 19,412 | — |
| n_missing | 0 | 0 | — |

## 数据来源

- **Stage 2 RQ-VAE**: `task57_HHHH_s22` (HHHH Poincaré ball, v3 z-score + KMeans-on-z recipe)
- **L=4 SID tensor**: `task_artifacts/results/exp59/task59_hhhh_l4_sid_tensor.pt` (shape (4, 11924))
- **Stage 3 TIGER ckpt**: `task59_hhhh_best.ckpt` (val/recall@10=0.0125 @ gs 700, 2560 max steps, patience=10)
- **Stage 4 推断**: `logs/inference/runs/task59_hhhh_s4/pickle/merged_predictions_tensor.pt` (shape (19412, 10, 4))
- **Eval 脚本**: `task_artifacts/scripts/task388v4_s4_item_eval.py` (与 task444 / task17 baseline 同口径)

## 关键修复 (相对 v5 失败)

1. **L=4 SID tensor shape mismatch**: task388v5 用 (12288, 3) 错误格式导致 IndexError。task59 用 task59_l4_sid_tensor.py 把 (12288, 3) → (4, 11924)，加 dedup 列匹配 Stage 3 期望。
2. **monitor metric**: v5 用 `val/recall@5_avg5` (不存在) 导致 EarlyStopping 1 min 崩。task59 改用 `val/recall@10` (从 `p4_run_one_seed.sh` 验证)。
3. **patience**: v5 用 patience=2 提前终止，task59 改 10。
4. **Hydra `=` in ckpt path**: ckpt path 含 `=` 导致 parser error，已 cp 到 safe path `task59_hhhh_best.ckpt`。

## Verdict

**HHHH end-to-end R@10 = 0.0117 << baseline 0.0973**。

这是 v7 paper section Caveats #1 的诚实 null result — **Stage 2 HHHH RQ-VAE cross-seed stability 不传递到 Stage 3 TIGER pipeline**。

### Stage 2 vs Stage 3 Gap 分析

- **Stage 2 表现 (strong)**: HHHH 6-seed cv=2-5% all layers, brand MFSR +0.07 (task476 验证)
- **Stage 3 表现 (weak)**: val/recall@10 best 0.0125 @ gs 700 / 2560 max, 早停触发 (连续 10 val_check 无提升)
- **Stage 4 表现 (catastrophic)**: item-level R@10 = 0.0117 (vs baseline 0.0973)

### 根因猜测

- **R1**: Stage 3 max_steps=2560 不够 (典型 Stage 3 收敛需 3000-5000 step)
- **R2**: v3 RQ-VAE recipe (z-score + KMeans-on-z) 生成的 SID 分布与 Stage 3 训练分布不匹配
- **R3**: Stage 3 监控 metric 用 val/recall@10 是稀疏信号 (top-10 命中率 < 5%)，EarlyStopping patience=10 不足以等到真正收敛
- **R4**: Stage 2 HHHH 本身对 Stage 3 TIGER pipeline 是中性偏负 (无 L1 H 收益传递)

### Paper 含义

v7 Caveats #1 现在变成 **verdict 而非 caveat**：
- **HHHH in Stage 2 RQ-VAE**: ✅ robust (cv 2-5%)
- **HHHH in Stage 3+4 end-to-end**: ❌ failed (R@10 < 1/8 baseline)

P5 paper section 应:
1. 报告 Stage 2 HHHH stability 作为 isolated finding
2. 显式标注 Stage 3+4 HHHH R@10 = 0.0117 作为 known limitation
3. 建议 future work: (a) Stage 3 longer training 5000+ step, (b) Stage 2/3 joint training, (c) HHHH ablate per-layer (HHHH vs HHEE vs HEEE) on Stage 4 R@10

## 状态

- [x] L=4 SID tensor built
- [x] Stage 3 v3 训练完成 (best ckpt @ gs 700)
- [x] Stage 4 inference 完成 (19412 users)
- [x] Item-level R@10 eval 完成
- [x] Verdict 文档 写完
- [ ] 更新 P5 paper section v7 Caveats #1 (将 caveat 升格为 verdict)

