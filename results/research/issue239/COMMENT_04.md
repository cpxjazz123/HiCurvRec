## 节点 4：Stage2 训练完成（3000 epochs）

| 项 | 值 |
|---|---|
| run 树 | `stage2_RQ-VAE/TIGER_curvloss_rq` |
| epoch 范围 | 1 → **3000**（完整预算，无早停） |
| 最终训练器碰撞率 | **0.142392**（raw_unique 21,086 / 24,587） |
| 冻结 TIGER 同口径 | 0.169276 |
| 末轮损失 | loss 3.368e-4 / recon 6.806e-5 |
| 产物 | `results/stage2_RQ-VAE/TIGER_curvloss_rq/{item_sids.json, rqvae_best.pth, sids_for_hgrec.npy, calibration.json}` |

**这个数字只说明码本更分散，不说明曲率保真改善**（issue 明确「不得把连续或硬 q 曲率改善自动视为
离散 SID 改善」）。C12 的三路（`z` / 硬 `q` / SID 重建 `q`）曲率指标正在计算，随后一并贴出。

### 下一步

1. §67 checkpoint 重导 + §89 碰撞率门槛（进行中）
2. C12/C13 评估表（曲率保真三路 + SID 结构）
3. 通过门槛后按原协议跑一次 Stage3（150 epochs）

*（`results/research/issue239/COMMENT_04.md`）*
