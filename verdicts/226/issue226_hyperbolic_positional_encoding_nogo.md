# Issue #226 Hyperbolic Positional Encoding NO-GO (T5 架构不兼容, 2026-08-09)

## Context

**用户 2026-08-09 方向**: 探索新曲率框架, 设计 Issue #226 "hyperbolic positional encoding" (HPE) — 让 T5 的 positional embedding 用 Poincaré 球面几何而非 Euclidean。

## 关键发现: T5 用 relative position bias, 不是 absolute embedding

**Transformer T5 源码 (`transformers/models/t5/modeling_t5.py`) 考古结果**:

| T5 组件 | 实现 | 含义 |
|---------|------|------|
| Token embedding | `shared.weight` (input + output tied) | ✓ 存在 |
| **Word position embedding** | **❌ 不存在** | T5 无 absolute position |
| Relative position bias | `T5LayerSelfAttention.compute_bias()` | ✓ T5 唯一的位置信号 |
| position_bias 维度 | `[batch, num_heads, query_len, key_len]` | per-query-per-key relative offset |

**T5 关键代码** (`modeling_t5.py:245-327`):
```python
class T5LayerSelfAttention:
    def compute_bias(self, query_length, key_length, device):
        # 计算 relative_position_bucket(...)
        # position_bias = relative_embedding(buckets)  # embed buckets to num_heads
        # 返回 [1, num_heads, query_len, key_len]
```

**这意味着**:
- T5 没有 `embed_positions.weight` (这是 BERT/GPT 风格)
- 任何"hyperbolic 化 absolute position" 的方案对 T5 不适用
- 真正的 T5 改造是 **hyperbolic relative position bias**: 让 `relative_position_bucket` 的输出通过 expmap0 映射到 Poincaré ball, 然后用双曲距离度量 attention bias

## 原始 Issue #226 方案 NO-GO 根因

1. **方案假设错误**: 设计时假设 T5 有 absolute position embedding, 实际只有 relative position bias
2. **架构兼容性问题**: HG-Rec 用 `T5ForConditionalGeneration`, 内部相对位置逻辑散布在 12 个 decoder layers 的 self/cross attention 中
3. **改动复杂度**: 改造 `compute_bias` 需要修改 T5 内部 forward 流程, 不是 patch 一个 embedding 层能解决的
4. **训练代价**: 修改后必须全量重新训练 (Stage3 数值彻底改变), 25-50 min/epoch × 200 ep ≈ 1.5h+

## R18 4 维度对比 (vs 历史 NO-GO)

| Issue | D1 spec | D2 实施 | D3 失败机制 | D4 引用 |
|-------|---------|---------|------------|---------|
| #76 sid incompat | 端到端新 SID | v10b stage2 OK + e2e 0.047 | T5 不认新码字 | sanity=0 |
| #87 attn_entropy | B_geo proxy attn | regularizer 0.001 | 静态 bias ≠ 动态 attn | ratio 1.287 best |
| **#226 HPE** | **hyperbolic 位置** | **T5 改 absolute embedding** | **T5 无 absolute position, 只有 relative bias** | **架构不兼容** |

## 决策

- **不实施** Issue #226 原方案
- **可选方向 (高复杂度)**: 改造 T5 `compute_bias` 为 hyperbolic (改 12 处), 1.5h+ 实施 + 1.5h+ 训练, 不保证奏效
- **R23 风险**: 改 T5 内部容易 wrapper broken (历史 #76 教训)
- **优先级**: P3 (低 ROI, 高风险)

## 替代方向 (低复杂度)

**Issue #228 Stage2 κ per-batch radius modulation** (见 `verdicts/228/issue228_stage2_per_batch_radius_mod.md`):
- Stage2 κ_eff = κ_base * (1 + α * batch_mean(item_radius))
- per-batch 标量调制 (非 per-item), 不会触发 #225 v2 40× 梯度爆炸
- 训练时间: 8 min (与 v15 capmatch 1000ep DDP 一致)
- 改动: Stage2 forward ~3 行

**Why**: T5 架构不支持 absolute position embedding 改造. 真要 hyperbolic 化位置信号, 改动规模超出 quick-patch 范畴.
**How to apply**: 不再尝试 Issue #226 HPE 原方案. Stage2 κ 改造应聚焦 per-batch 维度 (Issue #228), Stage3 端聚焦 Dbar/codeword 局部调制 (已 NO-GO).