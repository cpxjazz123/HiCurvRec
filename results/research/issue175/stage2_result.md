# #175 DUL Stage2 结果：碰撞 0.015699（基线的 1/10），§89 准入通过

## 训练

四卡 DDP、seed 42、3000 epoch、全局 batch 4096、端口 50202；稳态约 0.4–0.7 s/epoch，全程约 11 分钟。终点 `train_end` 记录 `final_collision = 0.015699`。

## 指标（§63 先记，再看碰撞）

`stage2_metrics/8faaabf2…json`

| 量 | DUL 臂 | 冻结基线参照 |
|---|---|---|
| **入口碰撞（前三码）** | **0.015699**（24201/24587 个不同三码） | **0.162444** |
| §63 `l3_collision` | **0.030707** | — |
| 二级不同前缀 | 12284 | — |
| 三级不同组合 | 24201 | — |
| **L1 Hard ER** | **1.5388** | 16.1417 |
| L3 Refinement Gain (bits) | 1.3350 | — |
| Conditional G3 | −0.0089 | — |
| 逐层占用 | 一级 254 码（最大 2.96%）、二级 252（1.27%）、三级 256（0.663%） | — |

## 判定

- **§89 准入门（`collision < 0.40`）：通过**（0.0157）。
- 这是本系列**第一个同时把碰撞率和 L1 Hard ER 大幅改善**的机制：碰撞 16.2% → **1.57%**，Hard ER 16.14 → **1.54**。
- DUL 轨迹（同一次运行的评估点）：150 → 0.7145、300 → 0.6259、350 → 0.4215、800 → 0.0566、850 → 0.0424、1950 → 0.0301、终点 → **0.0157**。
- `L_unique` 随撞码对消失而饱和到 0（`m = 0.9` 的宽松性所致），说明该损失在"消除同码"阶段有效、在残余对上不再发力——与 Issue 要求记录的方向性矛盾一致。

## 下一步

按 Issue：进 Stage3（冻结协议：NUM_EPOCHS=150、EVAL_INTERVAL=5、EVAL_START_EPOCH=75、EARLY_EVAL_EPOCHS=(25,50,75,100,125)、SEED=42、BEAM_SIZE=20、NO_EVAL=True、SKIP_TEST=False），最终判据 `test_R@10 > 0.065`。

## Stage3 已启动

`stage3_start(item_sids="results/stage2_RQ-VAE/TIGER_dul_rq/item_sids.json")` → pid 838659，variant `TIGER_dul_rq`，四卡 DDP；冻结协议核验通过（`NUM_EPOCHS=150`、`NO_EVAL=True`、`SKIP_TEST=False`、`SEED=42`、`BEAM_SIZE=20`，`mismatched={}`、`early_stop_present=false`）。§63 分层记录被复用（`passed: true`）。
