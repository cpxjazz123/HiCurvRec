# Issue #97: LETTER 复现 (Musical_Instruments) — R@10=0.0953

## 方法
LETTER (Wang et al., CIKM 2024) — 可学习 item tokenization。
RQ-VAE + 对比对齐 + diversity loss 的 learnable tokenizer, instantiation 为 LETTER-TIGER (LETTER tokenizer + T5 解码器)。

## 实现
- 官方仓库: `/fs04/ar57/wenyu/LETTER/LETTER-TIGER/`
- ckpt: `ckpt/Instruments_t5base_letter_fixed` (T5-base 200ep, seed=42, lr=5e-4)
- index: `.llamaindex-sk4-sk.json` (LETTER learnable tokenizer, LLaMA embedding + 4 层 256 codebook)
- base_model: `./ckpt/TIGER` (从 TIGER ckpt 初始化)
- 评估: beam=20, leave-one-out, recall@10 口径

## 结果 (test, 24772 users)
| 指标 | 值 |
|---|---|
| **R@10** | **0.0953** |
| R@5 | 0.0778 |
| R@20 | 0.1185 |
| NDCG@10 | 0.0717 |

## 对比
| 方法 | test R@10 | 差距 |
|---|---|---|
| HG-Rec 论文 LETTER (Instruments) | 0.1219 | — |
| **LETTER 复现** | **0.0953** | -21.8% |

## 关键结论
1. **LETTER > TIGER**: R@10 0.0953 > 0.0857 (+11.2%), 复现了论文的相对排序 (论文 LETTER 0.1219 > TIGER 0.1214)
2. 绝对差 -21.8% 符合本环境系统性偏差 (HG-Rec baseline 差 22%)
3. 历史 hit@10=0.0953 确认为同值 (leave-one-out 下 recall=hit)

## 评估修正
- test.py 增加 vocab resize: `if len(tokenizer) != model.config.vocab_size: model.resize_token_embeddings(len(tokenizer))` (transformers 5.x 兼容, 修复 CUDA index 越界)

## 产物
- verdict: `verdicts/97/letter_instruments.md`
- results: `tasks/4baseline_repro/results/letter_t5base_llamaindex_recall.json`
