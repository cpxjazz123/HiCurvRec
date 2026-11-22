## round 2 训练已启动

| 项 | 治疗臂 | 打乱对照 |
|---|---|---|
| 树 | `stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq` | `stage2_RQ-VAE/TIGER_graphlet_prefix_shuffled_rq` |
| 日志 | `results/research/issue230/r2_treatment.log` | `results/research/issue230/r2_control.log` |
| GPU | 0 | 1 |
| 配置 | 3000 epoch、seed 42、全局 batch 4096 | 同 |
| 启动时状态 | epoch 51、loss 下降中 | epoch 17、loss 下降中 |

两臂源码差异仅 `ARM_NAME` + 对照开关(见 COMMENT_15 的审计方式),本轮机制代码逐字节相同。
commit `642692fa`。

round 1 的两个缺陷(预算标定、覆盖率)已修(COMMENT_20)。等跑满 3000 epoch 后按 §5 重新对账。
