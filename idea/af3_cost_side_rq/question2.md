# question2 — Stage2 已达标；本轮目标：按冻结协议跑完 Stage3 并给出 test_recall@10（外加一份损失异常的说明）

**状态：OPEN（唯一开放追问）**　承接：`answer1.md`

## 0. 上一问的验收结论：**通过**

我逐项复核了 `results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/`，与 answer1 一致：

| 项 | 实测 |
|---|---|
| 事件 | `train_start` → **3000 行** `train`(1..3000) → `train_end`；共 3003 行 |
| 配方 | `epochs=3000`、每卡 1024、**总计 4096**、`seed=42`、无早停、四卡 |
| 墙钟 | `total_wall_time_s=667.253`（≈11.1 min），正常退出 |
| 产物 SHA | `item_sids.json b0e5c4f3…`、`rqvae_best.pth 106820b8…`、`training_metrics.jsonl 3e8fbc7f…`、`stage2_metrics/…json 99b947ba…`（全部匹配） |
| 分层记录 | `status=complete`、`technically_valid_sid=false` |
| 我从 SID 独立复算 | 24587 商品、**L1 40/256**、L2 前缀 **229**、L3 三元组 **737**、三码 item-weighted 碰撞 **0.9858**、L1 最大码占比 **62.9%** |

机制确实是按规格实现的：每层代价 `C/τ + β·s`，`τ=median(C)`（批内）、`β·max|s| = 0.1·τ`，三层都走 Sinkhorn，导出仍用确定性最近码。

所以按"验收＝一次完整 Stage2"这条规则，answer1 已达标；**本轮把目标推进到主线判据：Stage3 的真实 `test_recall@10`。**

## 1. 必须先解释的一个异常（不要求改，要求说清）

训练记录里：

| epoch | loss | recon |
|---|---:|---:|
| 1 | 9.215089376084507e-4 | 4.6866878983564675e-4 |
| 1501 | 0.03132634982466698 | 2.026476577157157e-4 |
| 3000 | **1.350724458694458** | 2.071199705824256e-4 |

`recon` 一直贴在 `2e-4`，但总损失涨了三个数量级；而逐 epoch 记录里 `quant` 字段是 `None`，所以从记录看不出涨的是哪一项（`compute_loss = recon + quant_loss`，`quant_loss = (codebook_term + β·commitment).mean()/TOTAL_DIM`）。

请在答复里明确二选一：

- **(a) 这是规格内的目标**：代价侧偏移把码本推离重构目标，`quant_loss` 单调上升是机制的预期后果 → 那就是设计层面的负结果，记录在案，不改；
- **(b) 这是意外的无界项**：点名是哪一项、为什么无界、以及它是否本意；若属实现缺陷，按 §19g **不能重训本臂**，只能以新的 arm 名重建修正版并另跑一次。

无论哪种，都请给出该 epoch 的 `quant_loss` 实际数值（把 `quant` 记进 `training_metrics.jsonl`，或从 checkpoint 复算一次），这样这一轮才有可核对的账。

## 2. 本轮唯一目标：跑完冻结协议的 Stage3

```
stage3_start("results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/item_sids.json")
```

协议按 §22 冻结，不许改：`NUM_EPOCHS=150`、`EVAL_INTERVAL=5`、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、`SEED=42`、`BEAM_SIZE=20`、`NO_EVAL=True`、`SKIP_TEST=False`；四卡端口 `50201`。

**Stage3 启动不做任何准入检查**（§19，owner 2026-11-09）：`technically_valid_sid=false`、L3 碰撞 0.9858、L1 仅 40 个码，这些都只记录、不阻断。不要因为"预计召回会很低"而跳过。

## 3. 本轮验收

一次**完整** Stage3：150 epoch 跑完、进程正常退出，并交付

1. `results/stage3_T5Train/TIGER_af3_cost_side_rq/` 下同名目录的 `test_final.json`（`n_eval`、`test_recall@5/@10`、`test_ndcg@5/@10`）；
2. `HG_Rec.log` 与 `training_metrics.jsonl`；
3. 报告：结果目录名、所用 SID 的 sha256（应为 `b0e5c4f3…`）、`n_eval`、`test_recall@10`；
4. **分别**给出：(a) 与 `TIGER_RQ-VAE` 两次实测（`0.05553717857205035`、`0.04019916781281011`）的差值；(b) `test_recall@10 > 0.065` 是否成立；
5. 按 §6b 归档到 R2（`upload_stage3.py TIGER_af3_cost_side_rq`），四件齐全，编号与各文件 sha256 写进 `results/research/issue139/r2_ledger.json`。

## 4. 禁止项

不改冻结协议、不恢复 early stop、不换种子、不做调参或开关消融、不为本臂再跑第二次 Stage2。

## 5. 审阅基线

| 文件 | SHA-256 |
|---|---|
| `idea/af3_cost_side_rq/answer1.md` | `324b0ae49308b3b2e884e8d3df7aa93a765eb59cd215851ed4392a39d73b133e` |
| `results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/item_sids.json` | `b0e5c4f3358bc44ad2d94a326b3e6c115a0ba38e3b455108f8a3e1750a5b1774` |
| `results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/rqvae_best.pth` | `106820b83f46d267741c5fe9f444fc77af0c469c1fd5a1c6df52a586174aec47` |
| `…/logs/training_metrics.jsonl` | `3e8fbc7f657fb7a875a19689f507727ef5b6e22195f515489b37ae08e30b4ffe` |
| `…/stage2_metrics/b0e5c4f3….json` | `99b947ba05fe2eb6dcf00d5ac106aa3f2b62e038f9b11d794e0386c128a77bc6` |
