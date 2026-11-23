## 最终报告

issue 正文 sha256 `7a1ee963…`，清单 16 条，逐条对账见 `results/research/issue239/VERDICT.md`。

### 结果

| 条目 | 判定 |
|---|---|
| C1–C6、C8、C10–C14、C16 | **PASS** |
| C7（复核 #238 根因） | **PASS** — 根因是码本梯度 NaN，不是尺度或分配问题 |
| C9（STE/梯度/smoke） | **PASS** — `dead_terms=[]`、曲率项 grad 3.547 |
| **C15（Stage3）** | **NOT MEASURED** — owner 于 epoch 8/150 要求结束 |

### 三个值得留下的结论

1. **#238 的阻塞是数值实现问题**：三个码本 `embed.weight` 的梯度含 NaN，源头是两处 `0/0`
   （球面块 `sqrt` 在重合点、球映射缺范数下界）。修复后同一诊断三步全绿，候选可训。
   这解释了「混合距离进 Sinkhorn 会炸、进 loss 不会」—— 不是代价矩阵的错，是梯度管线的错。
2. **曲率监督在硬 q 与 SID 重建 q 上确实生效**：`q_sid` MAE 2.681 → 0.929、Spearman 0.011 → 0.732，
   重建 MSE 9.6e-5 → 6.8e-5，碰撞率 0.1693 → 0.1425，行为边/随机 0.537 → 0.661。结构没有退化。
3. **它主要拟合正曲率**：负曲率组符号一致率 0.485（正曲率 0.970），接近随机。
   这是明确的短板，也是下一个机制该打的点。

### 未做的事（不隐瞒）

- **Stage3 未跑**：SID 层面的收益**未验证**。准入已过、SID 已重导一致，一条 `stage3_start` 即可续。
- **R2 未归档**：§6b 要求四件（源码/item_sids/rqvae_best/recall）齐全才上传；`recall.json` 因 C15 未测而不存在，
  按规则不上传。
- **§63 分层指标工具缺口**：`stage3_start` 的 `layer_record` 依赖已随 §94 删除的
  `tools.research.science.sid_layer_analysis`，现只能手算；本 issue 用 `evaluation.json` 等价覆盖。

### 证据路径

`results/research/issue239/`：`CHECKLIST`(issue) `RUN_LOG.md` `INFRA_*.json` `diagnose_238.py`
`diagnose_238.json` `calibration.json` `evaluation.json` `VERDICT.md` `FINAL.md`；
测试 `results/research/issue239/tests/`（5 项，全绿）；训练日志
`results/stage2_RQ-VAE/TIGER_curvloss_rq/logs/train_run.log`。
