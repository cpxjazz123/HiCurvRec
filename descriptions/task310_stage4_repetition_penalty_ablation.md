# Task #310 — Stage 4 repetition_penalty ablation on #30 GO ckpt + beam=50

**日期**: 2026-07-30
**目的**: 验证 K15 (Stage 4 inference 协议层不是 generate() 参数空间普遍有效) 是否在 repetition_penalty 维度也成立. repetition_penalty ∈ {1.0(default), 1.2, 1.5}.
**R11.5 自主决策**: 复用 #30 best_ckpt + beam=50 (K14 最优), GPU 0 (与 task309 Stage 3 GPU 1 并行, R7 兼容).
**零训练成本**, ~3 min total.
**关联**:
- K14: beam_size 20→50 +2.3% R@10 ✅
- K15: length_penalty 0% ❌
- task308 K15: Stage 4 inference 协议层不是 generate() 参数空间普遍有效, beam_size 是 unique 杠杆
