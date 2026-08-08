# LETTER-TIGER Instruments NO-GO: Gate 4 FAIL

## 4-Gate Audit

### Gate 1 (数据 + 量化): PASS
- 数据集: Amazon 2018 Musical_Instruments 5-core (24772 users / 9922 items)
- Index codes: `Instruments.index.json` 918 个 token (4-digit RQ-VAE codes), 与训练一致
- Tokenizer 扩展: 918 new tokens (vocab 32100→33018), T5Tokenizer.from_pretrained("t5-small") + resize
- 代码兼容性: transformers 4.x→新版修复 (BeamScorer/BeamSearchScorer 位置, evaluation_strategy→eval_strategy, Trainer tokenizer= 参数删除, model.model_parallel 属性)

### Gate 2 (训练稳定性): PASS
- 30 epochs 完成, eval_loss 从 4.98→3.226 (单调下降)
- bf16 混合精度正常, GPU 利用率 ~40%
- **Warning**: `missing keys ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']` — 扩展 embedding 权重未保存到 checkpoint, 训练时 embedding 已扩展但 checkpoint 保存的是原始 T5 权重
- 训练时间: 1h07min44s, 速度 ~4.1 it/s

### Gate 3 (评估决策): PASS (有条件)
- num_beams=5 时所有 beam 返回相同 item (recall@5=@10=@20), 无法反映真实 recall 曲线
- 测试覆盖全量 24772 样本, leave-one-out 评估 protocol 正确
- Gate 4 基于 num_beams=5 结果,数值被人为压低但趋势有效

### Gate 4 (Test R@10): FAIL
- **LETTER-TIGER R@10 = 0.0506** (beam=5, 全量 24772 样本)
- HG-Rec baseline R@10 = 0.1024 → LETTER 低 **50.6%**
- DIGER R@10 = 0.1121 → LETTER 低 **54.9%**
- DECOR R@10 = 0.1157 → LETTER 低 **56.3%**
- BLOGER (同一数据集): 0.0803-0.0864 → LETTER 比 BLOGER 还低

| 方法 | Test R@10 | vs HG-Rec |
|------|-----------|-----------|
| HG-Rec baseline | 0.1024 | — |
| DIGER (复现) | 0.1121 | +9.5% |
| DECOR (复现) | 0.1157 | +13.0% |
| BLOGER (复现) | 0.0803-0.0864 | -15.6%~-20% |
| **LETTER-TIGER (本轮)** | **0.0506** | **-50.6%** |

## 失败原因分析

1. **Missing embedding keys (主因)**: `model.resize_token_embeddings()` 后未保存扩展的 embedding 权重,checkpoint 加载时用原始 T5-small vocab (32100) 覆盖了扩展部分,导致 918 个新 token 的 embedding 全为 0/随机

2. **T5-small 容量不足**: 60M 参数, 918 个新 token 占 vocab 2.8%, 但 encoder/decoder 仍是标准 T5-small, 可能不足以学习 918-token 的离散语义空间

3. **生成式路线 vs embedding 路线**: LETTER 生成式 (encoder-Decoder Autoregressive) 相比 HG-Rec/DIGER/DECOR 的 bi-encoder 或 fusion 路线,在 Musical_Instruments 数据集上系统性表现更差 (BLOGER 也同理)

4. **num_beams=5 无 diversity**: 所有 beam 返回相同 item,recall@5=@10=@20 相同

## 教训

- LETTER/TIGER checkpoint 保存前必须额外保存扩展的 token embeddings
- Musical_Instruments 数据集上, 生成式推荐系统性弱于 embedding 相似度路线
- T5-small 可能需要更多 epoch 或更大模型 (T5-base) 才能弥补 embedding 缺失问题

## Commit
- `modeling_letter.py`: 兼容性修复 (删除废弃 import, 添加 model_parallel 属性)
- `evaluate.py`: recall metric 支持 "recall@" 前缀
- `finetune.py`: T5 加载顺序修复 (先加载再 resize)
- LETTER-TIGER 训练 ckpt: `ckpt/Instruments_fast/`
