# Task #308 — Stage 4 length_penalty ablation on #30 GO ckpt + beam=50

**日期**: 2026-07-30
**目的**: 验证 K14 (Stage 4 inference protocol 是真 R@10 杠杆) 是否在 length_penalty 维度也成立.
**R11.5 自主决策**: 复用 #30 best_ckpt + beam=50 (task307 已确认最优 beam_size), length_penalty ∈ {0.5, 1.0(default), 2.0}. 
**零训练成本**, GPU 0 sequential, ~3 min total.

**关联**:
- task307 K14: beam_size 20→50 +2.3% R@10, 50→100 saturation
- task290 verdict: 攻 Stage 3/4 训练协议而非 Stage 1/2 quantizer 架构
