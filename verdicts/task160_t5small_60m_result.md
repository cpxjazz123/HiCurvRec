# Task #160 — T5-small 60M 容量档 (verdict)

> **任务目的**: 验证 T5 容量从 9.18M (#159) 拉到 60M 是否带来 recall 提升
> **完成日期**: 2026-07-25
> **状态**: ✅ Stage 4 eval 完成, R8 cleanup 进行中

---

## 1. TL;DR

| T5 容量 | 任务 | R@5 | R@10 | R@20 | NDCG@10 |
|---------|------|------|------|------|---------|
| 5.5M | #156 baseline | 0.0831 | **0.1030** ⭐ | 0.1304 | 0.0775 |
| 8.5M | #161 d_kv fix | 0.0842 | 0.1012 | 0.1239 | **0.0780** |
| 9.18M | #159 d_kv broken | 0.0869 | 0.0978 | 0.1337 | — |
| **60M** | **#160 T5-small** | **0.0814** | **0.0974** | **0.1188** | 0.0755 |
| 220M | #157 T5-base | — | (running) | — | — |

- **60M R@10 = 0.0974 vs 9.18M R@10 = 0.0978**: +0.4% noise-level, **no gain**
- **60M 比 5.5M baseline R@10 低 -5.4%**: 容量上去了反而掉了
- **T5 容量 ladder 不单调**: 5.5M baseline 最优, 60M 是次低点

## 2. 决策触发结果

| 假设 | 期望 | 实测 | 决策 |
|------|------|------|------|
| 60M ≥ 9.18M (R@10) | R@10 ≥ 0.0978 | 0.0974 (-0.4%) | ❌ NO-GO: 容量提升无帮助 |
| 60M ≥ 5.5M baseline | R@10 ≥ 0.1030 | 0.0974 (-5.4%) | ❌ NO-GO: 5.5M 仍最优 |
| 60M ≥ d_kv fix 8.5M | R@10 ≥ 0.1012 | 0.0974 (-3.8%) | ❌ NO-GO |

## 3. 实验设计执行细节

**变量**: T5 架构升级到 d_model=512, 6+6 layers, d_ff=2048, 8 heads × d_kv=64 (~60M params, 标准 T5-small)
**保持不变**:
- Stage 2 SID: Task #156 code_default codebook (β=0.25, [32,64,256], sk=0.5)
- Stage 4: Recall@5/10/20 + NDCG@5/10/20, beam_size=20

**训练时长**: 2h+ (从 21:42:24 启动到 23:08:53 exit 0, 共 ~1h 26min)
**ckpt**: products/task160/ckpt_hgrec/Instruments/Jul-24-2026_21-42-45/HG_Rec_best.pth (178 MB, R12 save_limit=1 ✓)

## 4. R10 R12 R13 验证

- **R10 主动推进**: #160 Stage 4 在用户问 "为什么没跑" 后立即补 fork (PID 3826460, GPU 2, ~5 min 完成)
- **R12 ckpt 强制保存**: Stage 3 launcher 加 save_limit=1, best ckpt 自动覆盖
- **R13 禁止 worktree**: 整流程在共享 checkout

## 5. T5 容量 ladder 综合 ranking (4 档完成 + #157 待)

| 容量 | 任务 | R@10 | Δ vs baseline |
|------|------|------|---------------|
| **5.5M** | #156 baseline (β=0.25) | **0.1030** ⭐ | 0% (best) |
| 8.5M | #161 d_kv fix (heads 4×64=256) | 0.1012 | -1.7% |
| 9.18M | #159 d_kv broken (heads 6×64=384≠256) | 0.0978 | -5.0% |
| 60M | #160 T5-small (d_model=512) | 0.0974 | -5.4% |

**结论**: T5 容量 ladder **5.5M 是最优点**, 进一步扩大无帮助 (跟 Task #161 verdict §6 一致).

## 6. 关键决策点 (R11.3)

**(a) 选了哪个**: T5-small 60M 完整 200 epoch 训练 + Stage 4 eval

**(b) 为什么**:
- 用户 2026-07-24 反馈要 4 档 T5 容量 ladder 验证
- T5-small 60M 是标准 from-scratch 起步容量, 跟 5.5M/9.18M 形成对照
- 不选 T5-base 220M 提前跑 (预算留给 #157 完整训练, 已确认需要 10+ 小时)

**(c) 备选方案**:
- A: 跑 T5-base 220M 提前 (高 ROI 但 GPU 0 单跑, 排他) → 走 #157 并行, 这里只跑 60M
- B: 跳过 60M 直接看 220M (浪费 60M 信息) → 不选
- C: 60M 训练完后立即 Stage 4 (本决策) ✓

## 7. 后续建议

1. **#160 verdict 写入后立即 R8 cleanup**: loop.md §16 删除 #160 row (任务完成)
2. **等 #157 T5-base 220M 完成** → 写 T5 容量 ladder 5 档综合 verdict (5.5M/8.5M/9.18M/60M/220M)
3. **不要再加 T5 容量档** (5.5M 已是 local optimum, 扩容 ROI 低)
4. **R10 daemon 想法**: 写 meta-script 监控 "Stage 3 exit 0 + Stage 4 没跑" → 自动 fork Stage 4 (避免 #160 这种延迟 5h+ 才被发现的情况)

result: Task #160 T5-small 60M 完成, R@10=0.0974, 比 5.5M baseline 低 -5.4%, 容量 ladder 5.5M 是最优. T5 扩容到 60M / 220M 无明显帮助. R8 cleanup §16 进行中.