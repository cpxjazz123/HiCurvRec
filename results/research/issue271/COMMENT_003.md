## #271 Stage3 已启动（150 epoch 冻结协议）

- Stage2 run `stage2_cc9a31a9325b439f98c8b1779aaa32aa` 完成：full J **0.351359 ≤ J_max 0.781941**，
  collision **0.152113**，L2 组 **6,097**（#269 6,246，未压缩），独立共同尺子局部 A.1 MAE **1.893506**
  vs TIGER **7.309217**（Spearman −0.0055 vs 0.0159、sign 0.3926 vs 0.4325，混合结果）。
- Stage3：`TIGER_global_j_rq`，单卡（world_size=1，train batch 4096，已披露与四卡不可完全比较），
  150 epoch、seed42、beam20；SID `.../result/item_sids.json`（sha256 `e83ae666…`），export gate
  allow（collision 0.1518，checkpoint 复现 SID；174 项被 checkpoint 重写）。
- 层记录已按 §63 写到 `results/stage2_RQ-VAE/TIGER_global_j_rq/stage2_metrics/e83ae666….json`
  （L1 hard ER / 正负前缀命中 / conditional G3 无自动评估器，标 `NOT_AVAILABLE`，不填 0）。
- 运行日志：`results/stage3_T5Train/TIGER_global_j_rq/logs/Amazon_2023_Instruments/…/training_metrics.jsonl`；
  epoch 1 loss 4.0198，ckpt 已保存。完成后做最终 test 一次、R2 四件套、VERDICT 与关闭。
