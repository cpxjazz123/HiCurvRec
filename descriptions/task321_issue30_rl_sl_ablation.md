# Task #321 — Issue #30 r_l + s_l ablation (D6 backlog)

**日期**: 2026-07-30
**状态**: 🔄 READY (R11.5 自主决策启动 D6 ablation, GPU 1 空闲 跟 task320 错峰)
**Stage**: Stage 1 + Stage 2 + Stage 3 + Stage 4 (3-arm: r_l only / s_l only / r_l+s_l control)
**Anchor**: Issue #30 GO marginal R@10=0.1022 (+0.2pp vs baseline 0.1020)

## 背景

Issue #30 Stage 4 端点 R@10=0.1022 是首个击败 baseline 的 GO marginal, 但只 +0.2pp. Issue #30 设计是 **per-layer Codebook Transforms r_l + s_l 同时施加** (r_l=[0.1,1,10] radius 缩放 + s_l=[2,2,2] norm scale + R_l=I). 哪个 axis 是真杠杆未知.

task318 (Issue #38 Arm 1) Stage 4 R@10 (K=100) = 0.0996 表明 K=100 amplifier 不是 universal, 但 K=50/100 amplifier 仍存在. 跟 D6 ablation 联立: 哪个 axis (r_l 或 s_l) 才是 Issue #30 marginal GO 的真杠杆.

## 设计 (3-arm ablation)

| Arm | r_l (radius) | s_l (scale) | R_l | 决策意义 |
|-----|--------------|-------------|-----|---------|
| **A** (r_l only) | [0.1, 1.0, 10.0] | [1.0, 1.0, 1.0] | I | 测试 r_l 单独效果 |
| **B** (s_l only) | [1.0, 1.0, 1.0] | [2, 2, 2] | I | 测试 s_l 单独效果 |
| **C** (Issue #30 GO) | [0.1, 1.0, 10.0] | [2, 2, 2] | I | baseline (Issue #30 marginal GO) |

每个 arm:
- Stage 1: 100 epoch, K=[64,128,256,1], c_k=baseline, codebook transforms 应用 r_l/s_l
- Stage 2: Sinkhorn 5 iter → SID unique 9922/9922 + 3-digit collision ≤ 0.20
- Stage 3: T5-mini 200 epoch 训练 (跟 Issue #30 task301 一致)
- Stage 4: R@10 @ K=20 (default Issue #30)

## 决策阈值

| Arm R@10 | 解读 |
|----------|------|
| A R@10 ≈ C R@10 ≈ B R@10 | 三者无显著差异 → Issue #30 marginal 是 noise, 都不构成杠杆 |
| A R@10 > B R@10 + ≥ 0.005 | r_l 是真杠杆, s_l 是 noise |
| B R@10 > A R@10 + ≥ 0.005 | s_l 是真杠杆, r_l 是 noise |
| A ≈ B ≈ C (三等高) | r_l + s_l 协同效应, 各自不构成杠杆 |
| C > A, B (任一) ≥ 0.005 | r_l + s_l 必须同时施加, 联立真杠杆 |

## 关键决策点 (R11.5 自主决策)

1. **启动 D6**: R11.5 兜底. R10 backlog 真空 (Issue #38 5-arm 全 NO-GO 收口), D6 是 backlog 唯一高 ROI 候选.
2. **GPU 1 启动**: task320 Arm B 退出, GPU 1 释放. 我用 GPU 1 跑 D6 跟 task320 错峰. R7 ✅ (不抢卡).
3. **3-arm 串行 (vs 并行)**: 只有一个空闲 GPU, 串行跑. 总耗时估计 ~150 min × 3 arm = 7.5 hr, 但 task320 还要 ~60 min 才能让 GPU 0/2/3 空闲.
4. **K=20 Stage 4 only**: 不跑 K=50/100 sweep, 跟 Issue #30 task301 决策阈值一致 (K=20 是 Issue #30 默认). K=100 amplifier 不是 universal (task318 evidence), 不在 D6 范围.

## 跟 task318 联立

- task318 证明: K=100 amplifier 不是 universal (task301 Issue #30 K=100 R@10=0.1045, task318 AdamW K=100 R@10=0.0996 → K=100 amplifier = task301 specific training产物).
- task318 证明: optimizer 不是 R@10 杠杆 (Adam=AdamW 数学等价).
- **D6 的核心问题**: Issue #30 +0.2pp marginal GO 是 noise 还是真杠杆? D6 拆 r_l vs s_l 验证.

## 关联

- Issue #30 (closed Stage 4 GO marginal, per-layer Codebook Transforms)
- Issue #38 (closed, Stage 3/4 训练协议改造 5-arm 全 NO-GO)
- Task #301 (Issue #30 完整 Gate 0-4 pipeline)
- Task #318 (Issue #38 Arm 1 optimizer 4-arm, K=100 amplifier 非 universal)
- Task #312/313 (Issue #35 r_l/s_l 隔离 NO-GO -17%, 但用更激进 r_l=[0.5,1,2]+s_l=[1,1,1] 测试 → Issue #30 design [0.1,1,10]+[2,2,2] 更温和)
- loop.md §16 backlog D6 候选

## GPU

- GPU 1 (空闲, task320 Arm B 退出后)
- GPU 0/2/3 (task320 Arm A/D/E 占用, 不抢)
- 启动时机: 立即 (Task #321 launch_armA_stage1.sh)

## 预期

- D6 Arm A r_l only R@10 预期: 0.100~0.102 (跟 Issue #30 marginal GO 同量级或略低, 因缺 s_l 缩放)
- D6 Arm B s_l only R@10 预期: 0.102~0.104 (s_l 单独施加 norm scale 2x 可能更强)
- D6 Arm C r_l+s_l R@10 预期: 0.1022 (Issue #30 GO baseline)
- 关键: A vs B vs C 差异是 D6 真正 output

## R11.3 透明

- 选 3-arm 而不是 4-arm (含 null control r_l=[1,1,1]+s_l=[1,1,1]): Issue #30 task301 Gate 0 回归测试已证明 baseline+r_l/s_l 等价, 不再跑 control arm. 节省 GPU.
- 选 100 epoch Stage 1: Issue #30 task301 用 100 epoch, Gate 1 PASS L0/L1/L2 100% util. 沿用.
- 选 200 epoch Stage 3: Issue #30 task301 用 200 epoch + early stop, Stage 3 PASS. 沿用.
- 串行跑 3 arm: GPU 1 唯一空闲. R7 接受.
- 不重跑 K=100 amplifier (task318 evidence): K=100 amplifier 非 universal, D6 关注 Stage 1/2 端点本身.
result: Task #321 description — Issue #30 r_l + s_l ablation (D6 backlog). 3-arm 设计: r_l only / s_l only / r_l+s_l control. 决策 Issue #30 marginal +0.2pp 是 r_l 还是 s_l 还是协同. 见 verdict (待 task320 释放更多 GPU 后跑)