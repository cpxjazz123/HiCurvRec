# answer34 — 四卡 Stage2 已跑满 3000 epoch，产物与分层诊断齐全
**状态:完成一次完整 Stage2(非 smoke、非中途停),进程正常退出;未启动 Stage3**

## 1. 启动与冻结配方
- 入口:`stage2_start("TIGER_struct_mixed_rq_ste_fix")`,MCP 启动成功;
- `run_id`(本框架无独立 run_id 字段):结果目录
  `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/`,launcher pid `3622503`,
  worker 完成后退出;
- 配置来自日志原文:`epochs=3000 batch_size_per_rank=1024 total_batch=4096 lr=0.001`,
  `world_size=4`(GPU0–3 各一 rank),`seed=42`,无早停;未钉单卡;
- 一 epoch smoke 先通过(`SMOKE_PASS`,24 steps,`dead_terms=[]`,`nonfinite_terms=[]`),
  随后才启动正式训练。

## 2. 时间与结束
| 项 | 值 |
|---|---|
| 启动 | 2026-11-10T12:25:52+11:00 |
| 结束 | 2026-11-10T12:43:17+11:00 |
| 总墙钟 | `1038.097 s` ≈ 17.3 min |
| 每 epoch | 0.32–0.35 s |
| 结束方式 | `train_end` 事件,launcher 与四个 rank 均已退出(进程表为空) |
| epoch=3000 | `loss=0.0001818569580791518`,`recon=7.262202416313812e-05`,`lr=0.0`,`codeword_h_max=[0.474347167,0.14896293,0.06407544]` |

`training_metrics.jsonl` 共 3004 行,其中 `"event": "train"` 恰为 **3000** 行(epoch 1..3000),
末行是 `train_end`。

## 3. 产物(均在 `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/`)
| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `rqvae_best.pth` | 4,604,533 B | `9da6aa4baa3f12bf992b48d4f06d0d48755945d30d1b8a42335487f0749f0e06` |
| `item_sids.json` | 1,260,993 B | `1839d1cea74dbc0eb5789c88b3fda1d69ff87586c561ea7b1c19ca41a5ec61d3` |
| `logs/training_metrics.jsonl` | 961,511 B | `9efc6e0d782116e9cedda66f75f6b1e68ea7aa93f3c9cc4fcb54d722215e9d57` |
| `logs/train_migrated.log` | 237,488 B | — |
| `stage2_metrics/1839d1…61d3.json` | — | `0bbe9a16ce1e234a5cf03ad029a1fdde3bd0d6f5145dbc4394a27528703ece22` |

说明:训练器把 `training_metrics.jsonl`/`train_migrated.log` 写在候选树 `logs/`,训练结束后已
**按字节复制**到结果目录 `logs/`(sha 相同),满足交付位置要求。

## 4. 分层指标(§63,读取 Stage3 前完成;诊断,不设门槛)
记录文件:`stage2_metrics/1839d1cea74dbc0eb5789c88b3fda1d69ff87586c561ea7b1c19ca41a5ec61d3.json`,
`status=complete`,`technically_valid_sid=true`,`stage3_results_used=false`。

| 指标 | 值 |
|---|---:|
| L1 码使用 | 224 / 256 = **87.50%** |
| L2 码使用 | 256 / 256 = **100%** |
| L3 码使用 | 256 / 256 = **100%** |
| 不同的二级前缀数 | **9289** |
| 不同的三级三元组数 | **21862** |
| L3 Collision | **0.1108309269125961**(末行同值) |
| L1 Hard ER | **7.5473559589581685** |
| L1 正样本命中率 | 0.07236779869454167 |
| L1 负样本命中率 | 0.009588496830952606 |
| L3 Refinement Gain | **1.8662739304133817 bits** |
| Conditional G3 | −0.03222699772633103 |

## 5. 未做(按 question34 §6)
- 未启动 Stage3;
- 未做开关消融、调参扫描、多种子;
- 未改冻结超参。

## 6. 审阅基线偏差
question34 §7 给出的 `training_hooks.py` SHA 为 `8ede9c09…`;磁盘实际为
`4046b3f8c7680dc7d9b4534b9069347792a5e224ba5a641bedfd8cd38e01fc34`(与 q33 记录一致)。
本轮训练使用磁盘实际源码;该差异如实留档。
