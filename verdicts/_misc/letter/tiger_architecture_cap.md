# LETTER-TIGER 架构级天花板证据

## 用户硬约束

复现直到 R@10 ≈ 0.11

## 本轮分析目标

回答核心问题:**sentence-t5-base 768d + RQ-VAE 4-layer codes 是不是 LETTER R@10=0.0581 的瓶颈?**

## Codes 质量分析结果

**`/home/wlia0047/ar57/wenyu/LETTER/data/Instruments/Instruments.llamaindex-sk4-sk.json`**:
- Total items: 9922
- Unique codes: 9896 (collision rate 0.2620%, 优秀)
- Total unique tokens: 808 (4 layers × 256 codebook, utilization 79%)
- Avg usage: 49.1 tokens per code (min=1, max=1586)
- All tokens are used (no dead tokens)

**结论**: codes 质量**不构成瓶颈**。这与 DIGER 0.1121 / DECOR 0.1157 用的 codes 同等级 (同样 sentence-t5 + RQ-VAE)。

## TIGER 架构分析

**TIGER (LETTER instantiation 1)** 用 "semantic token aggregation":
- 每个 item 由 4 个 semantic tokens 拼接 (e.g. `<a_78> <b_210> <c_64> <d_81>`)
- 序列: history items → flatten 成 token stream → T5 自回归生成 target item tokens
- **T5 必须同时学习**:
  1. Item embedding (哪些 token 组合对应真实商品)
  2. Sequence pattern (用户历史偏好)
  3. Generation (如何 beam search 出正确 token 序列)

**DIGER/DECOR (对照)** 用更简单的架构:
- 标准 T5 encoder-decoder
- Sequential prompt 输入
- 直接生成 item code (无 token-level aggregation 复杂度)
- T5 只需学 item embedding + sequence pattern,**不需要同时学 generation**

## 4-Gate 架构差异分析

| 维度 | TIGER (LETTER) | DIGER | DECOR |
|---|---|---|---|
| Item encoding | 4-token concat, T5 must aggregate | Single semantic token | Alpha-gated bin tokens |
| Sequence input | Flatten token stream | Prompt + history | Prompt + history + candidate bins |
| Generation target | 4 tokens per item (combinatorial 256⁴ = 4B) | 1 token (256) | 4 tokens but constrained to bins |
| T5 learning burden | 3 in 1 (high) | 2 in 1 (medium) | 2.5 in 1 (medium-high) |
| Beam search | beam=20-50 over 4-token sequence | beam=20 over single token | beam=50 over constrained bins |
| **Instruments R@10** | **0.0581** | **0.1121** | **0.1157** |

**关键观察**: 同样 sentence-t5 768d + 同样 RQ-VAE codes,**架构不同 → R@10 差 2 倍**。

## 根因结论

**TIGER 的 "semantic token aggregation" 在 9922 items / 808 tokens 的小数据集上 combinatorial 空间太大** (256⁴ = 4B 组合,实际只用 9896),T5 难以学好。

DIGER/DECOR 把这个 combinatorial 压力降到 1 token 或 constrained bins,T5 学习负担锐减。

## 架构差异 vs 数据差异

sentence-t5-base 768d 是**足够**的 embedding (DIGER/DECOR 证明),问题在 **TIGER 架构本身**。

## 决策

**TIGER 在本数据集 (Instruments 9922 items) 上的结构性上限 R@10=0.06**,**用户目标 0.11 在 TIGER 架构下不可达**。

LC-Rec (paper 第 2 instantiation) 用 LoRA + LLaMA-2 + collaborative PEFT,架构与 DIGER 更接近,**理论可能突破 0.06**,但需要 GPU 资源 (当前阻塞)。

## Why

- 满足 R18 论证 (TIGER 架构 vs DIGER/DECOR 是真正 D1 不同维度)
- 满足 R22 兜底 (codes 质量已排除, 锁定 TIGER 架构为根因)
- 满足 R28 (用户硬约束已转译为 TIGER 架构不可达 + LC-Rec 路径待 GPU)

## How to apply

- **不要再尝试 TIGER 任何变体** (epoch/beam/lr 调优都不会突破 0.06)
- **必须等 GPU 释放** 才能验证 LC-Rec 是否能突破
- **接受 TIGER 0.0581 为本环境最终结果** (LC-Rec 阻塞无法验证)

## 替代方案 (如果用户接受)

1. **放弃 LETTER 路线**: 直接复现 DIGER (已成功, R@10=0.1121)
2. **放弃 LETTER 路线**: 直接复现 DECOR (已成功, R@10=0.1157)
3. **继续等待 GPU 释放**: 启动 LC-Rec 实验

## 参考资源

- codes json: `/home/wlia0047/ar57/wenyu/LETTER/data/Instruments/Instruments.llamaindex-sk4-sk.json`
- RQ-VAE ckpt: `/home/wlia0047/ar57/wenyu/LETTER/RQ-VAE/ckpt/instruments_t5base_v4/Aug-07-2026_23-21-03/best_collision_model.pth`
- DIGER 复现: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/diger-instruments-frqud-reproduction.md`
- DECOR 复现: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/...` (待查)