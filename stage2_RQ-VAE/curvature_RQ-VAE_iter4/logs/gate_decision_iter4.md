# iter4 Gate Decision：Lorentz 稳定原型中心

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter4
- **机制**：`iter4_lorentzian_centroid_recenter`
- **最终裁决**：**NO-GO**
- **裁决规则**：仅当 Stage3 完整运行后的 `test_R@10 > 0.065` 时才允许 PROMOTE。
- **审计基线提交**：`127a75c7654c31084283c17258a61eb9e68572f8`
- **审计日期**：2026-09-21

## 1. Agent A / Agent B 串行证据

1. Agent A 报告：`logs/lit_search_iter4.md`，提供至少三个候选 P1/P2/P3，仅负责文献检索和候选生成，没有进行最终方向裁决。
2. Agent B 报告：`logs/direction_decision_iter4.md`，在读取 Agent A 报告后唯一推荐 P3：Lorentz 稳定原型中心（closed-form Lorentzian centroid/re-centering）。
3. 本轮只实现一个 novelty；没有加入 MLR、attention、额外 loss、Sinkhorn epsilon sweep、新 optimizer 或新曲率调度。

## 2. 实现与训练前验证

- 机制通过 Poincare↔Lorentz geometric transform，将 assignment 加权的 Lorentz 原型重心归一化后映回现有 Poincare codebook。
- 保留现有 Poincare distance、Sinkhorn、hard `argmax` SID、M2/M3 residual 和 cyclic curvature 接口。
- 已运行常驻 gradient check：
  - `total_loss.requires_grad`：PASS；
  - `total_loss.grad_fn`：PASS；
  - encoder 与 codebook 梯度：非零且有限；
  - 动态曲率 steps `(0, 25000)` 的 loss：有变化；
  - Lorentz 重心化未阻断 `loss.backward()`；
  - 未发现 `_last_*.detach()` 导致的 silent no-op。
- gradient check 输出：`GRAD_CHECK PASS`。
- 已对修改的 Python 文件运行 `python3 -m py_compile`，包括 `curvature_config.py`、`curvature_RQ-VAE.py`、`modules/hyperbolic.py`、`modules/quantize.py`、`modules/rqvae.py`、`scripts/grad_check.py` 和 `scripts/export_sids_for_stage3.py`。

## 3. Stage2 实际训练记录

- DDP：`world_size=4`。
- 输入 embedding：`item_emb.npy`，形状 `(24587, 768)`。
- 训练目标：`MAX_GLOBAL_STEPS=100000`，checkpoint 间隔 `10000`。
- `logs/train_iter4.log` 中存在一条旧的早停描述：
  `EARLY-STOP TRIGGERED at step 10000 ... downstream uses rqvae_step10000.pt`。
  该描述与实际 trainer 日志不一致，不能作为实际执行结果。
- 实际 `logs/train_migrated.log` 明确记录：
  - step 10000：`early_stop=False`；
  - step 20000：`early_stop=False`；
  - step 30000：`early_stop=False`；
  - step 40000：`early_stop=False`；
  - step 50000：保存 `rqvae_step50000.pt`。
- 因此本轮 Stage2 实际继续训练到 step50000；Stage3 使用的是 `sids_step50000.npy`，不是已被 rolling cleanup 删除的 `sids_step10000.npy`。
- Stage2 SID 指标仅作描述性记录，不作为 gate。step40000 质量文件记录：`full_gini=0.12050945102632496`、`n_unique_full=21361/24587`、`l01_unique_pairs=11818`、`h_l1_given_l0=5.1386859079049625`。

## 4. SID 导出与 Stage3 wiring

- raw SID 来源：`results/stage2_RQ-VAE/curvature_RQ-VAE_iter4/out/rqvae/instruments/sids_step50000.npy`。
- raw SID：形状 `(24587, 3)`，dtype `int32`，3-token unique 数 `21606`。
- collision extension 后：`dataset/Instruments/sids_for_hgrec.npy`，形状 `(24587, 4)`，dtype `int64`，全表 unique 数 `24587`。
- extension token 值域：`768..780`；4-token SID 全表唯一校验通过。
- Stage3 读取的 SID 是四 token，配置为 `n_digit=4`，词表为 `784`（EOS `783`）。
- Stage3 输出目录名为 `tiger_baseline`，原因是 trainer 的共享 `CODE_PATH` 文件名仍为 `item_sids_recbole.json`；该命名不能用于判断输入来源。本轮运行前该 JSON 内容已被 iter4 导出结果覆盖，运行完成后已恢复 baseline。

## 5. Stage3 完整评估

Stage3 使用 4 卡 DDP、`BEAM_SIZE=20`，完整运行到 `epoch=150/150`，随后完成最终评估；没有使用 validation 指标提前裁决。

最终文件：

```text
stage3_T5Train/logs/tiger_baseline/Amazon_2023_Instruments/Sep-21-2026_11-48-41/test_final.json
```

最终结果：

```json
{
  "best_checkpoint": null,
  "n_eval": 57439,
  "test_recall@5": 0.03675203259109664,
  "test_recall@10": 0.054022528247358065,
  "test_ndcg@5": 0.02384694801986332,
  "test_ndcg@10": 0.029401613947649348
}
```

按项目约定：

```text
test_R@10 = test_recall@10 = 0.054022528247358065
```

与硬目标比较：

```text
0.054022528247358065 <= 0.065
```

因此不得 PROMOTE。

## 6. 共享输入恢复

Stage3 结束后已恢复共享 baseline 文件：

```text
/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/dataset/Amazon_2023_Instruments/item_sids_recbole.json
```

恢复后的 SHA256：

```text
1d0b8177c346567c76251fe8fb2db996b1e63198e6409229f0697fc0a63f3ed2
```

Stage3 外层进程、`torchrun --master_port=50201` 及四个 rank 均已退出；没有残留训练进程。

## 7. 决策

**NO-GO：iter4 Lorentz 稳定原型中心未达到 Stage3 `test_R@10 > 0.065`，不进入 promoted baseline。**

本轮代码目录按 NO-GO 流程归档清理；本决策文件、Agent A/B 报告以及关键 Stage2/Stage3 日志在提交历史中保留。
