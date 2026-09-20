# iter1 Gate Decision

## 机制选择
- **P1：双曲 MLR/决策边界式量化**（Agent B 唯一推荐）。
- 实现：每层增加 `mlr_normal`/`mlr_point`/`mlr_bias`，以当前 `c(t)` 计算双曲超平面有符号距离作为 assignment logits，log-domain Sinkhorn 平衡，soft assignment `@ codebook` 作为 hard-forward/soft-backward STE；前向仍是硬码字，反向梯度可同时回流 MLR 参数。
- 改动：仅 `modules/quantize.py`、`modules/hyperbolic.py`、`init/kmeans.py`（有界 max_iters=32 以消除无界初始化阻塞）、`scripts/grad_check.py`、`curvature_config.py`、`scripts/export_iter1_checkpoint.py`。

## Stage2 训练
- 4 卡 DDP，强制 `GRAD_CHECK PASS`，周期起点/中点 quantize loss 变化。
- 20000 步后 L0/L1/L2 usage 衰减到 `1/1/1/256`，符合 Skill 3(a) codebook collapse 触发条件。
- 早停于 30000 步，保留 `rqvae_step30000.pt`，退出审计写入 `logs/train_iter1.log`。
- raw SID 仅 `1/24587` 唯一组合，三层 Gini=0.0000；四 token collision extension 后全表唯一（`EXPORT_PASS step=30000`），但 Stage2 SID 完全退化。

## Stage3 训练
- 输入：iter1 的 `item_sids_recbole.json`（4 token, vocab=25358）。
- 150 epoch：valid recall@10=0.01891，valid ndcg@10=0.00914。
- 最终 `test_final.json`：`test_recall@10 = 0.01586030397465137`，远低于 0.065 硬目标。
- 与上轮正式 `test_R@10=0.05356987412733508` 相比下降 -70.4%。

## 失败根因
1. 双曲 MLR assignment 在 cyclic c(t) 引导下演化为极端 collapse，三个 codebook 各只剩一个码字，说明 MLR 几何决策面与 `c(t)` schedule 形成反馈不收敛。
2. Agent B 方向裁决的 Stage1 ceiling 风险被命中：纯 Stage1 曲率/几何变更无法把推荐语义带回。

## 动作
- 恢复 Stage3 输入：从 `item_sids_recbole.json.bak` 还原覆盖前的正式 SID；如缺失则回退到最近一次非 iter1 的正式 SID 备份（如 `item_sids_iter28.json`）。
- 删除 `stage2_RQ-VAE/curvature_RQ-VAE_iter1/` 与 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter1/`（先 commit logs 再删）。
- 后续迭代不应再使用 P1 同一实现方向。