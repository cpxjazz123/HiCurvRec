# question2 — 本臂被碰撞门槛挡在 Stage3 之外：先解释损失增长，再用新 arm 把三层碰撞压到 0.40 以下

**状态：OPEN（唯一开放追问；owner 2026-11-10 新增 Stage3 门槛）**　承接：`answer1.md`

## 0. Stage2 已验收，但 Stage3 被新门槛拒绝

`results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/` 是一次完整 Stage2（3000 epoch、四卡、全局 batch 4096、seed 42、667 s、四份产物 SHA 全对），这一项按原规则**通过**。

但 owner 2026-11-10 新增硬门槛：**Stage2 的三层码字碰撞率必须 < 0.40 才允许进入 Stage3**（已写入 `CLAUDE.md` §89，实现在 `tools/genrec-mcp/server.py::tool_stage3_start`）。我实测调用：

```
stage3_start("results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/item_sids.json")
→ started: false
  refused: "three-level collision 0.970025 is not below the 0.40 Stage3 entry gate"
  collision: 0.9700248098588685
```

对照 §63 的另一个定义：`l3_collision`（三码 item-weighted 重复占比）= **0.9858055069752308**。**两个定义都远高于 0.40，所以本臂禁止进入 Stage3**，这不是可以靠"预计召回低也跑一下"绕开的。

## 1. 本轮交付之一：损失增长的事后说明（纯文档，不训练）

| epoch | loss | recon |
|---|---:|---:|
| 1 | 9.215089376084507e-4 | 4.6866878983564675e-4 |
| 1501 | 0.03132634982466698 | 2.026476577157157e-4 |
| 3000 | **1.350724458694458** | 2.071199705824256e-4 |

`recon` 稳定在 `2e-4`，总损失涨三个数量级；逐 epoch 记录里 `quant` 是 `null`。请明确二选一：

- **(a) 规格内目标**：偏移项把码本推离重构目标，`quant_loss` 单调上升是机制的预期后果 → 记为设计层面的负结果；
- **(b) 意外的无界项**：点名哪一项、为什么无界；若属实现缺陷，按 §19g 不能重训本臂。

并给出 epoch=3000 的 `quant_loss` 实际数值（把 `quant` 记进 `training_metrics.jsonl`，或从 `rqvae_best.pth` 复算一次）。

## 2. 本轮交付之二：一条能同时满足门槛与"一次完整 Stage2"的修正版

按 §19g，**本臂不得重训**。要让这个机制继续，只能从冻结混合几何基线重建一个**新 arm 名**的候选树，并跑一次完整 Stage2。验收：

1. `stage2_start("<新 arm>")`，四卡、全局 batch 4096、`EPOCHS=3000`、`SEED=42`、无早停，3000 epoch 跑完、正常退出；
2. 产物齐全：`rqvae_best.pth`、`item_sids.json`、逐 epoch `training_metrics.jsonl`（`quant` 不再为 `null`）、`stage2_metrics/<sid_sha256>.json`；
3. **`collision = 1 − 不同三元组数/商品数 < 0.40`**（与 `stage3_start` 的门槛同一口径），同时报 §63 的 `l3_collision` 与 4-token `collision`，三者带名字并列；
4. 报告 run_id/pid/墙钟、epoch=3000 记录、SID 的 sha256。

达到门槛后，Stage3 才允许启动（冻结协议不变）。

## 3. 结构约束（不是调参，是可验证的设计条件）

本臂的偏移上界是 `β·max|s| = 0.1·τ`，但**逐商品的代价裕度**（top-1 与 top-2 的差）没有被约束，偏移足以整体压过均衡分配，于是码本坍缩、`quant_loss` 单调上升。

修正版必须让偏移与裕度挂钩：例如偏移量不超过该商品 top-1/top-2 代价差的一个固定比例（该比例由数据推导并写清），并在测试里断言"偏移介入前后码字重合率不低于某阈值"——这样碰撞门槛是可预期的，而不是训练完才知道。

## 4. 禁止项

不改冻结超参与 Stage3 协议、不恢复 early stop、不换种子、不做调参扫描或开关消融、不为 `TIGER_af3_cost_side_rq` 再跑第二次 Stage2、不因门槛未达而修改门槛定义。

## 5. 审阅基线

| 文件 | SHA-256 |
|---|---|
| `idea/af3_cost_side_rq/answer1.md` | `324b0ae49308b3b2e884e8d3df7aa93a765eb59cd215851ed4392a39d73b133e` |
| `results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/item_sids.json` | `b0e5c4f3358bc44ad2d94a326b3e6c115a0ba38e3b455108f8a3e1750a5b1774` |
| `…/rqvae_best.pth` | `106820b83f46d267741c5fe9f444fc77af0c469c1fd5a1c6df52a586174aec47` |
| `…/logs/training_metrics.jsonl` | `3e8fbc7f657fb7a875a19689f507727ef5b6e22195f515489b37ae08e30b4ffe` |
| `…/stage2_metrics/b0e5c4f3….json` | `99b947ba05fe2eb6dcf00d5ac106aa3f2b62e038f9b11d794e0386c128a77bc6` |
| `CLAUDE.md` §89（新门槛） | 见 `8fa14242` 提交 |
