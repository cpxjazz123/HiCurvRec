# Task #352 / Issue #63 — 方向 A 重开: 三层 κ 原位曲率感知同步重校准

**日期**: 2026-07-31
**触发**: GitHub Issue #63 (owner 创建) `[方向A 重开] 三层 κ 原位曲率感知同步重校准——先过 Gate -1 再复现 arXiv:2405.13979v4`
**基线**: HG-Rec Task #84 Test R@10=0.1020
**决策阈值**: R@10 > 0.1020 → GO, R@10 ≤ 0.1020 → NO-GO

---

## 1. Issue 主张

方向 A (Curvature-Aware Optimizer per #55) 重开, 但加上 **Gate -1 严格预检** (在 Gate 0 之前的 pre-flight check). 三层 (L0/L1/L2) κ 原位曲率感知同步重校准, 必须先过 Gate -1 才能进入 Gate 0.

## 2. 历史背景 (per issue body)

| Issue | 结果 | 关键产物 |
|-------|------|----------|
| #47 | κ-stereographic 公式修复 | commit 61e707c, Möbius 符号 + NaN 7 测试 PASS |
| #49 | FreeCurvHRQVAE 4 阶段完整 | commit 0e9e4a6, R@10=0.1005 (NO-GO) |
| #55/#56 | Stage 2 SID collapse | unique=0.01%, Stage 3 hard gate 触发 STOP |
| #57 | SID init 实验 | R@10=0.0921 NO-GO (hyperbolic < random 方向反向) |
| #58/#59/#60 | dead parameter 根因 | 修复 commit 5a7f35f + c47e540 + 5486155 |
| #61 | hyp_c=-1 sandbox 修复 | Gate 0 5/5 PASS, 整体 NO-GO 收口 |

## 3. Gate -1 spec (Issue #63 强制)

**目的**: 实施基础就位, 零 GPU 预检, 不启动训练即可判 NO-GO.

**检查项** (8 项):
1. 实施基础就位 + commit (代码路径)
2. L0/L1/L2 层独立 learnable curvature state (per-layer theta_m)
3. 干净 optimizer (no dead params, all params have grad)
4. forward path clean (encoder/assignment/loss 无 .item() detach)
5. batch 维度独立 (无 global state pollution)
6. optimizer state detach (θ_m autograd 验证)
7. codebook/SID update path isolation (no leakage between L0/L1/L2)
8. Stage 3/4 接口对齐 (R12 ckpt, R15 push)

**STOP 条件**: 任一 FAIL → NO-GO 收口, 不进入 Gate 0

## 4. 假设

- **H1**: #49 R@10 < baseline 根因不是几何本身, 而是 Stage 1/2 SID collapse 跟 Stage 3/4 接口不干净 (Issue #58/#59/#60 修复 + Gate -1 严格隔离能解锁)
- **H2**: 修复 Stage 1/2 SID unique/collision/utilization 后, Stage 3/4 能到 R@10=0.1020 baseline
- **H3**: Gate -1/0 通过 + 干净 optimizer/forward/loss/codebook/SID + 方向 A = R@10 > 0.1020

## 5. Gate 0/1/2/3 spec (简述)

- **Gate 0**: 跟 #47 同标准 sanity (distance scale, codebook score, assignment entropy, NaN/Inf)
- **Gate 1**: Stage 1 训练 L0/L1/L2 utilization ≥ 90%, collision 跟 #55/#56 对比
- **Gate 2**: Stage 2 SID unique/collision/per-layer utilization 健康
- **Gate 3**: Stage 3 T5 + Stage 4 R@10 eval, R@10 > 0.1020 GO

## 6. R11.5 决策

按用户 loop 指令, 先做 Gate -1 (zero-GPU), FAIL 即 NO-GO 收口.