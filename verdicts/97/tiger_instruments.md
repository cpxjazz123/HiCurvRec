# Issue #97: TIGER 复现 (Musical_Instruments) — R@10=0.0857

## 方法
TIGER (Rajput et al., NeurIPS 2023) — 经典 RQ-VAE 语义 ID 生成检索。
RQ-VAE tokenization + T5-base 自回归生成 + beam search 解码。

## 实现
- 复用 LETTER 官方仓库内嵌 TIGER 实现 (`/fs04/ar57/wenyu/LETTER/LETTER-TIGER/`)
- ckpt: `ckpt/Instruments_t5base_redo/checkpoint-21414` (T5-base 200ep, seed=42, lr=5e-4)
- index: `.index.json` (RQ-VAE semantic ID)
- base_model: T5-base (d_model=768, 12 层)
- 评估: beam=20, leave-one-out, recall@10 口径 (与 HG-Rec 论文 Table 1 一致)

## 结果 (test, 24772 users)
| 指标 | 值 |
|---|---|
| **R@10** | **0.0857** |
| R@5 | 0.0681 |
| R@20 | 0.1067 |
| NDCG@10 | 0.0624 |

## 对比
| 方法 | test R@10 | 差距 |
|---|---|---|
| HG-Rec 论文 (Instruments) | 0.1214 | — |
| **TIGER 复现** | **0.0857** | -29.4% |

## 说明
- 与 HG-Rec 论文 TIGER=0.1214 相比低 29.4%, 符合本环境系统性偏差 (HG-Rec baseline 0.1024 vs 论文 0.1315, 差 22%)
- 评估修正: test.py 增加 `model.resize_token_embeddings(len(tokenizer))` 修复 vocab 越界 (transformers 5.x 兼容)

## 产物
- verdict: `verdicts/97/tiger_instruments.md`
- results: `tasks/4baseline_repro/results/tiger_t5base_ckpt21414_recall.json`
