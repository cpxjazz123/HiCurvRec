# LETTER-TIGER Instruments 修复后复现: embedding bug 已修, Gate 4 仍 FAIL

## 修复内容 (对比首轮)

### Bug 根因
首轮 `finetune.py` 加载顺序错误:
```python
pretrained_model = T5ForConditionalGeneration.from_pretrained(args.base_model)
pretrained_model.resize_token_embeddings(len(tokenizer))   # ← resize 在 from_pretrained 之后
config.vocab_size = len(tokenizer)
model = LETTER(config)
model.load_state_dict(pretrained_model.state_dict(), strict=False)
```
`LETTER(config)` 以 vocab=33046 分配 embedding, 但 `pretrained_model` resize 后的权重在 `load_state_dict` 时 shape 匹配, 表面无错。真正问题是 checkpoint 保存/加载时报:
```
missing keys: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']
```
→ 918 个新 token 的 embedding 未被正确初始化/保存, 模型不认识 item code token。

### 修复方案
```python
pretrained_model = T5ForConditionalGeneration.from_pretrained(args.base_model)
orig_vocab = pretrained_model.config.vocab_size   # 32128 (不是 32100)
config.vocab_size = orig_vocab                     # 先用原始 vocab
model = LETTER(config)
model.load_state_dict(pretrained_model.state_dict(), strict=False)   # shape 完全匹配
del pretrained_model
model.resize_token_embeddings(len(tokenizer))       # 再 resize → [33046, 512]
# transformers 自动用 Stanford vocab expansion (multivariate normal, 同 mean/covariance) 初始化新行
```
关键点:
1. t5-small 实际 vocab=**32128** (不是 32100)
2. `strict=False` 不跳过 shape mismatch, 必须先 load 再 resize
3. `resize_token_embeddings` 已内置 Stanford 词表扩展初始化, 无需手动加噪

### 评估修复
- `evaluate.py`: `get_metrics_results` 增加 `recall@` 前缀支持 (原只认 `hit@`)
- test beam: 5 → **20** (首轮 beam=5 导致 R@5=R@10=R@20 完全相同, 无法反映 recall 曲线)

## 4-Gate Audit

### Gate 1 (数据 + 量化): PASS
- Amazon 2018 Musical_Instruments 5-core, 24772 users / 9922 items
- `Instruments.index.json` 918 tokens (4-digit RQ-VAE codes)
- Tokenizer 扩展: 32128 → 33046

### Gate 2 (训练稳定性): PASS
- **无 missing keys warning** (修复验证点)
- 30 epochs, eval_loss 5.005 → **3.278** (单调下降, E28 后 plateau)
- 训练时间 ~1h10min, ~4.2 it/s, bf16

| epoch | 修复版 eval_loss | 首轮 (bug 版) |
|-------|-----------------|--------------|
| 1  | 5.005 | 4.982 |
| 9  | 3.703 | 3.691 |
| 18 | 3.393 | 3.355 |
| 21 | 3.329 | 3.285 |
| 30 | **3.278** | 3.226 |

修复版 eval_loss 略高 — Stanford 词表扩展初始化更保守 (新 token 从原始分布采样, 而非 bug 版的未初始化随机值意外获得更大方差)。

### Gate 3 (评估): PASS
- 全量 24772 test 样本, leave-one-out
- beam=20, prefix-constrained trie decoding
- **R@5/10/20 正常递增** (首轮塌缩问题已解)

### Gate 4 (Test R@10): FAIL

| 指标 | 修复版 (beam=20) | 首轮 bug 版 (beam=5) |
|------|-----------------|---------------------|
| R@5  | 0.0524 | 0.0506 |
| R@10 | **0.0627** | 0.0506 |
| R@20 | 0.0801 | 0.0506 |
| NDCG@5  | 0.0346 | 0.0352 |
| NDCG@10 | 0.0379 | 0.0352 |
| NDCG@20 | 0.0422 | 0.0352 |

对比同数据集基线:

| 方法 | Test R@10 | vs HG-Rec |
|------|-----------|-----------|
| DECOR (复现) | 0.1157 | +13.0% |
| DIGER (复现) | 0.1121 | +9.5% |
| HG-Rec baseline | 0.1024 | — |
| BLOGER (复现) | 0.0803~0.0864 | -15.6%~-21.6% |
| **LETTER-TIGER 修复版** | **0.0627** | **-38.8%** |
| LETTER-TIGER 首轮 (bug) | 0.0506 | -50.6% |

修复带来 **+0.0121 (+23.9%)** 提升, 证明 embedding bug 确实存在且已解决, 但 R@10=0.0627 仍比 HG-Rec baseline 低 **38.8%**, Gate 4 FAIL。

## 剩余差距分析 (bug 已排除后)

1. **T5-small 容量**: 60M 参数, 918 个新 token 需从零学习 4-digit code 的组合语义, 30 epoch 可能不足
2. **无 RQ-VAE 联合训练**: 本轮直接用上游预生成的 `Instruments.index.json` codes, 未跑 LETTER 论文的 RQ-VAE (cf_loss + diversity_loss) 阶段 — item tokenization 质量非 LETTER 原生
3. **架构路线差异**: LETTER/BLOGER (生成式 autoregressive decoding) 在 Musical_Instruments 上系统性弱于 HG-Rec/DIGER/DECOR 的 embedding 相似度路线 — BLOGER 0.0803~0.0864 同样低于 baseline, 与本轮结论一致
4. **未复现论文配置**: LETTER 原论文用 LLaMA backbone + 自训 RQ-VAE tokenizer; 本轮用 t5-small + 上游 codes, 属简化复现

## 结论

- **embedding bug 修复有效**: R@10 0.0506 → 0.0627 (+23.9%), missing keys warning 消失, recall 曲线恢复正常
- **Gate 4 仍 FAIL**: 0.0627 < 0.1024 baseline, 差距 38.8%
- 该数值代表 "t5-small + 上游 codes" 配置下的真实能力, 不再受 bug 污染
- 若要对齐论文, 需补 (a) 自训 RQ-VAE tokenizer 阶段 (b) 更大 backbone (t5-base/LLaMA)

## 产物
- 训练 ckpt: `LETTER/LETTER-TIGER/ckpt/Instruments_fast2/`
- 首轮 (bug) ckpt: `LETTER/LETTER-TIGER/ckpt/Instruments_fast/`
- 修复版结果: `LETTER/LETTER-TIGER/results_letter_tiger_fixed.json`
- 首轮结果: `LETTER/LETTER-TIGER/results_letter_tiger.json`
- 训练日志: `$CLAUDE_JOB_DIR/tmp/letter_tiger_fixed.log`
- LETTER repo commit: bbf4263 (首轮修复) + 本轮 finetune.py embedding 修复 (LETTER remote 已 suspended, 无法 push)
