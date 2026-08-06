# Poincaré Attention Pooling — 路径 A NO-GO 闭环

**日期**: 2026-08-07
**方案**: 用 norm-weighted Poincaré attention pooling 替换 Euclidean attention
**结果**: **NO-GO** — 三条路径全部失败, 根因不可绕过

---

## 三条路径总结

| 路径 | 设计 | Gate 3 结果 | 失败原因 |
|---|---|---|---|
| **B** distance-based | softmax(-poincare_dist) | ratio 0.03 (远 < 1) | softmax 加权 + Frechet mean 平均化 |
| **A v1** inverse norm | attn ∝ 1/‖fused_p‖ | ratio 0.64 (仍 < 1) | 投影到 ball 后 norm 范围压到 [0,1], 反转失效 |
| **A v2** softmax(-raw_norm) | attn = softmax(-‖fused‖/τ) | ratio **0.12** (< 1) | 欧氏加权求和, norm 大的 token 主导方向 |

---

## 根本原因（数学证明）

设 fused = [t0, t1, t2, t3], 其中 ‖t0‖=0.24 (抽象), ‖t2‖=3.93 (细节)

**路径 A v2 期望**:
- attn = softmax(-norm/τ) → t0 权重高
- bos_vec ≈ t0 (抽象概念)

**实测**: bos_vec 最像 **t3 (norm=1.1)**, 反转后差异只有 Euclidean 的 12%

**数学解释**: 欧氏加权求和
```
bos_vec = Σ attn_i * t_i = 0.6 * t0 + 0.05 * t2 + ...
质量贡献: t0 → 0.6 * 0.24 = 0.144
         t2 → 0.05 * 3.93 = 0.197  ← 仍主导
```

**norm 大的 token 在加和中"质量"自然更大**, 因为向量本身就长。**attention 权重设计无法克服这一点**, 除非:
- 在切空间 (logmap0) 加权 + expmap0 (双曲空间加法)
- 但我们已测试, Frechet mean 平均化同样抹平 norm 翻转 (路径 B)

---

## 结论

**单点 attention pooling 改造无法实现"层次结构敏感性"**:
- 欧氏加权 sum: norm 大主导方向
- 切空间加权 + expmap0: Frechet mean 平均化
- distance-based attention: softmax 平滑

**根因**: attention pooling 本身的数学性质, 与 Poincaré / Euclidean 无关。

---

## DECOR 借鉴点的真正价值

Poincaré attention pooling 是 **走错了路**。DECOR PromptFormer 的核心不是 attention pooling, 而是:

```
e_final = α * e_soft + (1-α) * e_fused
e_soft = Σ attn_j * candidate_j  # 候选 SID bins 的 soft attention
```

**关键**: `e_soft` 是 **对候选 SID embedding 的软加权**, 不是对 fused_embeds 的池化。
- 这让 decoder 输入 embedding 反映"该位置应该用什么 SID"
- alpha 门控让模型混合静态 lookup + context-aware 软加权

**路径 A / B 失败的本质**: 我误解了 DECOR 的设计 — 以为 attention pool fused_embeds 是核心, 实际核心是 **candidate bin attention**。

---

## 推荐下一步

放弃单点 attention pooling 改造。**实施完整 DECOR PromptFormer**:
1. 在 Stage3 T5 decoder 输入 embedding 层, 加 candidate bins (1024 个 SID embedding → 分 bin)
2. Attention pool fused_embeds → bos_vec (用 Euclidean attention pool 就够了, 不需要 Poincaré)
3. 对每个 decoder 位置, 用 bos_vec 做 query, candidate bins 做 key → e_soft
4. e_final = α * e_soft + (1-α) * e_fused, α=0.35
5. 保留现有 B_geo / HAB / CodewordGeoResidual 不动

预期增益: +3~6% (与 DECOR 11.57% 对齐)

---

## 当前产物

- 模块: `common/poincare_attention_pool.py` (90 行, 数值稳定但无层次敏感性)
- precheck 脚本: `verdicts/poincare_attn_pool_pathA_precheck.py` (Gate 1/2/4 PASS, Gate 3 FAIL)
- precheck v1 verdict: `verdicts/poincare_attn_pool_precheck_verdict.md`
- 本 verdict: NO-GO 闭环

## Why
单点 attention pooling 改造 (无论 Poincaré / Euclidean / norm-weighted / distance-based) 都不能在加权和层面实现 norm 偏好.
层次结构敏感性是 attention pooling 数学性质的天花板, 不是参数或几何空间的问题.

## How to apply
- 停止所有 attention pooling 单点改造尝试
- 下次 DECOR 借鉴方向应该是**完整 PromptFormer (candidate bins + alpha gate)**, 而非 attention pool 单独优化
- 任何 attention pooling 改造都必须在 precheck 阶段验证 Gate 3 (层次敏感性), 数值稳定 + 梯度流通 ≠ 有效