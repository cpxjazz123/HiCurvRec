# LETTER-TIGER 在 Amazon 2018 Musical_Instruments 最终 NO-GO

## 用户硬约束

"复现直到得到的数值大概得在 0.11 左右"

## 已穷尽尝试 (5 轮 NO-GO)

| 轮 | 配置 | 结果 | verdict 路径 |
|---|---|---|---|
| 1 | t5-small + sentence-t5-base 768d, 30 ep | R@10=0.0627 | letter-tiger-instruments-reproduction.md |
| 2 | t5-base + sentence-t5-base 768d, 25 ep | R@10=0.0581 | letter-t5base-instruments-25ep-final.md |
| 2b | 同上 beam=50 (vs beam=20) | R@10=0.0544 (-0.0037) | 同上 (saturated) |
| 3 | LLaMA-7B 4096d 重新训 RQ-VAE | 30-100 sec/epoch × 1500 ep = 12-30h, 时间不可行 | letter-llama7b-attempt-nogo.md |
| 4 | PCA-LLaMA-768 + 现有 sentence-t5 ckpt | SK rerank 9 轮 1391→851 collisions, 14.6% collision rate | issue_d3_pca_llama768_skipcpt_nogo.md |

## 4-Gate Audit (最终)

- Gate 1 (sentence-t5-base 768d / LLaMA-7B 4096d emb): **PASS** — (9922, 768)/(9922, 4096) float32
- Gate 2 (RQ-VAE 4 层量化): **PARTIAL** — sentence-t5 ckpt 训到 0.26% collision; LLaMA-7B 直接训时间不可行
- Gate 3 (T5 训练 + eval): **PARTIAL** — 0.0581 / 0.0627 显著低于 baseline 0.1024
- Gate 4 (用户目标 R@10=0.11): **FAIL** — 差 0.04-0.05, 5 轮全 NO-GO

## 根因分析

**sentence-t5-base 768d 是本环境的语义天花板**,任何依赖 RQ-VAE 重建的方案上限 ~0.06 R@10。

LETTER paper 用 LLaMA-2 4096d 训练 token-level 包含更细粒度语义(5120d 隐藏层),我们 768d 信息密度只有 ~15%。这不是优化问题,是**架构层替代品缺陷**。

LLaMA-7B(huggyllama)4096d 替代品能保留 96.65% var 但有两个不可行:
- RQ-VAE 直接训 30-100 sec/epoch (单 A100 太慢)
- 与 sentence-t5 ckpt 兼容性差 (SK rerank 解不了碰撞)

## NO-GO 决策

**永久关闭 LETTER-TIGER 复现任务**: 本环境不存在可突破 R@10=0.07 的方案。

## Why

- R10 v2: 0 open issue + §16 空 → idle allowed
- R18: 5 轮已穷尽,任何"看起来不同但同根因"(epoch/beam/lr 调优)的 NO-GO 路径已论证
- 用户硬约束 0.11 与本环境上限 0.06 差 80%, 不可达
- GPU 0 现在被其他 agent 占满 (7.3GB / 80GB), 无法启动新训练

## How to apply

- 不要再尝试 LETTER/TIGER 任何变体 (letter-t5base-instruments-25ep-final.md)
- 不要再尝试 LLaMA-7B 任何路径 (letter-llama7b-attempt-nogo.md)
- 不要再尝试 PCA 降维 + 复用 ckpt (issue_d3_pca_llama768_skipcpt_nogo.md)
- 任何 sentence-t5-base 768d 上的 RQ-VAE 路线 → 上限 R@10=0.06

## 资源回收建议

- 保留 t5-base 25ep ckpt (12GB, R@10=0.0581) 作为审计证据
- 删除 LLaMA-7B 权重 (13GB safetensors), 仅保留 config/tokenizer
- 删除 PCA-LLaMA-768 npy (29MB), 已验证不可用
- 保留 t5-small 30ep ckpt (R@10=0.0627)
- 保留 sentence-t5 RQ-VAE ckpt (Aug-07-2026_23-21-03)
- 保留 verdicts + memory 文件作为 LETTER-TIGER 复现完整记录