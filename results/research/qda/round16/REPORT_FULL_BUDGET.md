# 冻结预算(3000 epoch)下的保护层实测

此前所有运行都是缩短预算。这一次是**真正的 3000 epoch**,保护层全程激活 —— 一旦有任何一步违反,该步即死。

## 结果:跑完,而且比未保护的基线更干净

| | L1/L2/L3 使用码字 | 三级码碰撞率 |
|---|---|---|
| 已发表基线(未保护) | **249** / 256 / 256 | 0.169276 |
| 本轮(保护层,3000 epoch) | **256** / 256 / 256 | **0.155733** |

- 未保护基线在 L1 上死掉 **7 个码字**;保护层下 **一个都没死**。
- 碰撞率反而更低(0.1693 → 0.1557):码字全活并没有牺牲区分度。
- 3000/3000 epoch 完成,**没有任何一步被保护层拒绝**。

## 开销

| | 每 epoch |
|---|---|
| 未保护(round15,60 epoch) | 0.1757 s |
| 本轮(3000 epoch 稳态均值) | 0.2130 s(中位 0.2340 s) |
| 比值 | **+21 %** |

与 round15 在 60 epoch 上测得的 +18 % 一致;3000 epoch 的完整运行没有出现随时间增长的额外开销。

## 产物

`run/item_sids.json` 通过产物审计:`levels 256/256/256`,collision 0.155733。checkpoint 与 SID 留在 `round16/run/`(不入库),日志 `runs_stdout.log`。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round16/build_full_budget.py
cd results/research/qda/round16/trees/GUARDED_FULL && <genrec_env_v2>/python3.9 -u train_rqvae.py
```

约 11 分钟。所有输出重定基到 `round16/run/`,不触碰任何已发表结果。
