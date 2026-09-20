# iter3 Gate Decision

## 结论

**NO-GO：不晋级 iter3，保留当前 promoted baseline。**

唯一硬裁决指标为 Stage3 完整训练后的 `test_R@10`。本轮结果为 `0.05247305837497171`，未达到硬目标 `> 0.065`。

## 机制

- iteration：`iter3`
- mechanism：`iter3_hyperbolic_dense_ste_codebook_reparameterization`
- Agent B 唯一推荐：P3，不依赖 dead-code 触发的稠密 STE/codebook 更新与双曲切空间 codebook 重参数化
- 方向决策文件：`logs/direction_decision_iter3.md`
- 前置审计 commit：`2e500a16b19284066e9b43f016e84dff20d399f6`

## Stage2 验证与产物

- 常驻梯度检查：`GRAD_CHECK PASS`
- 梯度检查覆盖周期步：`0` 与 `25000`
- P3 `codebook_transform`：三层均获得非零、有限梯度
- Stage2 `global_step=100000`
- checkpoint：`results/stage2_RQ-VAE/curvature_RQ-VAE_iter3/out/rqvae/instruments/rqvae_final.pt`
- raw SID：`(24587, 3)`
- collision-extended SID：`(24587, 4)`
- collision-extended unique：`24587/24587`
- full SID Gini：`0.0510`
- per-layer Gini：`[0.0743, 0.1374, 0.1698]`
- raw unique：`23288/24587`
- `l01_pairs=15570`
- `H(L1|L0)=5.6457`

以上 SID 指标均为描述性记录，不作为 gate。

## Stage3 有效性证据

本轮只保留一组有效 Stage3 运行：

- 启动时间：`2026-09-21T05:39:40+10:00`
- 训练命令：`/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py`
- DDP：单组 `torchrun`，4 ranks
- 日志：`stage3_T5Train/logs/_stage3_launcher.log`
- 本次新日志启动标记：`[RecBole-aligned] train=396958 valid=57439 num_items=24587 vocab=786 eos=785 n_digit=4 beam=20 topk=[5, 10] exclude_history=True`
- `n_digit=4` 证明本次加载的是 iter3 collision-extended SID，而不是三 token baseline SID
- 训练完成：`epoch=150/150`
- trainer return code：`0`
- final JSON：`stage3_T5Train/logs/tiger_baseline/Amazon_2023_Instruments/Sep-21-2026_05-41-06/test_final.json`
- `n_eval=57439`

最终指标：

| 指标 | 数值 |
|---|---:|
| valid `recall@10`（epoch 150 diagnostic） | `0.05891467469837567` |
| test `R@10` / `test_recall@10` | `0.05247305837497171` |
| test `recall@5` | `0.03487177701561657` |
| test `ndcg@10` | `0.02873201442049614` |

目标判定：

```text
test_R@10 = 0.05247305837497171 <= 0.065
NO-GO
```

## 文件恢复与清理

Stage3 结束后 runner 才恢复共享 baseline SID。恢复后的 SHA256：

```text
1d0b8177c346567c76251fe8fb2db996b1e63198e6409229f0697fc0a63f3ed2
```

该哈希与 baseline 备份一致；最终核验无存活 Stage3 worker。iter3 未修改 Stage1、Stage3 源码、`item_emb.parquet` 或 `Instruments.inter.json`。

由于 `test_R@10` 未超过硬目标，iter3 不得 promote；决策文件提交后删除 `stage2_RQ-VAE/curvature_RQ-VAE_iter3/`，审计材料保留在 git 历史中。
