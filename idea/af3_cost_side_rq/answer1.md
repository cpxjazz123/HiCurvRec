# answer1 — 曲率+行为偏移进入三层 Sinkhorn 代价：跑满 3000 epoch，但 SID 严重坍缩
**状态:完成一次完整 Stage2（3000 epoch、四卡、正常退出）；结构结果很差，如实记录**

## 1. 实现
候选树 `stage2_RQ-VAE/TIGER_af3_cost_side_rq/`，从冻结混合几何基线 `TIGER_mixed_rq` 复制，仅改写 `ARM_NAME`。

- `model/mixed_rq.py`：每层代价 `C/τ + β·s`，其中
  - `C` 为既有 `mixed_cost`（E16 欧氏 + H16 庞加莱，`phi` 与基线一致）；
  - `τ = median(C.detach())`（批内中位距离，数据推导）；
  - `s` 为行为偏移余弦，`β = 0.1·τ / max|s|`（相对比例固定为 1/10，不手调）；
  - 三层都启用 Sinkhorn（`sk_epsilon=0.003`、`sk_iters=50`），分配仍走均衡规则，无 `argmin` 替换。
- `model/model.py`：`behaviour` 为 TRAIN 图上的后继商品内容嵌入均值；`s_k(u)=cos(beh_u[:16], code_k[:16])`，导出 `get_indices` 使用同一偏移。
- `train_rqvae.py`：Dataset 附带 item_id；按 TRAIN 历史的相邻对构建 `behaviour`；关闭 F0 在 epoch 1500 的 alpha readout 轮（本机制 alpha 固定 1/2，且该轮依赖缺失的 `analysis_sid_recall/quotient_curvature.py`）。
- 开关：`BEHAVIOUR_ON`、`GEOMETRY_ON` 存在于代码；测试覆盖行为偏移改变 token、几何关改变代价、Sinkhorn 仍启用。

测试：`tests/test_q1_cost_side.py` 3 项通过；真实入口 smoke（四卡、真实梯度、导出 SID）`SMOKE_PASS`。

## 2. 与 `struct_mixed_rq` 的差别（一句话）
只把“曲率距离 + TRAIN 可见行为偏移”放进每层 Sinkhorn 的分配代价（并用批内中位 τ 与固定 1/10 比例定标），读回、结构项与均衡规则与基线逐项一致；额外改动只有：三层都启用 Sinkhorn、关闭 F0 的 epoch-1500 alpha readout、数据加载携带 item_id。

## 3. 运行
| 项 | 值 |
|---|---|
| 启动 | `stage2_start("TIGER_af3_cost_side_rq")`，pid `223767` |
| 配方 | 四卡、每卡 1024、全局 batch 4096、`EPOCHS=3000`、`SEED=42`、无早停 |
| 起止 | 2026-11-10T14:00:47+11:00 → 14:12:41+11:00 |
| 墙钟 | `667.253 s`（≈11.1 min） |
| epoch=3000 | `loss=1.350724458694458`，`recon=2.071199705824256e-04`，`lr=0.0` |
| 退出 | `train_end`，launcher 正常退出 |

训练轨迹：`loss` 从 ~3e-4 持续升到 1.35；`recon` 仍约 2e-4，说明偏移项把分配推离重构目标，属真实负结果。

## 4. 产物与 SID
| 文件 | SHA-256 |
|---|---|
| `item_sids.json` | `b0e5c4f3358bc44ad2d94a326b3e6c115a0ba38e3b455108f8a3e1750a5b1774` |
| `rqvae_best.pth` | `106820b83f46d267741c5fe9f444fc77af0c469c1fd5a1c6df52a586174aec47` |
| `stage2_metrics/<sid>.json` | `99b947ba05fe2eb6dcf00d5ac106aa3f2b62e038f9b11d794e0386c128a77bc6` |
| `logs/training_metrics.jsonl` | `3e8fbc7f657fb7a875a19689f507727ef5b6e22195f515489b37ae08e30b4ffe` |

## 5. 分层诊断（§63，门槛不作准入）
| 指标 | 值 |
|---|---:|
| L1 码使用 | 40 / 256 = 15.62% |
| L2 码使用 | 56 / 256 = 21.88% |
| L3 码使用 | 61 / 256 = 23.83% |
| 不同二级前缀 | 229 |
| 不同三级三元组 | 737 |
| L1 Hard ER | 1.0428 |
| 4-token 碰撞（训练器 `collision`） | 0.9700248098588685 |
| 三码 item-weighted 碰撞（`l3_collision`） | 0.9858055069752308 |

## 6. 结论边界
本轮只交一次真实 Stage2；不据此声称机制有效，也未启动 Stage3（按 question1 §4 留到下一轮）。
