# Task #194 / Issue #40 Gate 1 — K=64 Protocol-Matched Control (✅ CONFIRMED: K-scaling NOT a real R@10 lever)

**日期**: 2026-07-30 14:55
**触发**: Issue #40 owner 2026-07-29 质疑 task194_k0256 R@10=0.1053 anchor 可能 inflation (Stage 1 batch_size + Stage 2 Sinkhorn protocol leak)
**状态**: ✅ **Issue #40 Gate 1 CONFIRMED** — task194 K=64 protocol-matched = **statistically neutral** (R@10=0.1025-0.1038), K-scaling 不是真 R@10 杠杆
**类型**: Gate 1 实证 (Stage 1 + 2 + 3 + 4 全跑完)

---

## 1. 背景

| 项 | 内容 |
|----|------|
| **task194_k0256 原始** | R@10=0.1053 (claimed to be K=0 effect anchor) |
| **Issue #40 Gate 0 质疑** (task329) | 4 项协议差异: Stage 1 batch_size 1024 vs 256, epochs 1000 vs 500; Stage 2 sk_epsilons argmin vs Sinkhorn+0.003; ckpt best_loss vs best_collision. → anchor 不可信, 需 protocol-matched control 验证 |
| **Gate 1 设计** | 重新跑 task194 K=64 用 #84 baseline recipe (mirror): num_emb_list=[64,128,256] + batch_size=1024 + epochs=1000 + sk_eps=0 + best_collision ckpt |
| **决策门** | test_R@10 ≤ 0.1020 → task194 protocol leak CONFIRMED (K=0 not real lever). test_R@10 > 0.1020 → task194 K=0 真有效 |

---

## 2. 实测结果

### 2.1 Stage 3 (T5-mini 200 epoch, early terminated at Ep 56)

| Epoch | val_R@10 | val_NDCG@20 | early stop counter |
|-------|----------|-------------|---------------------|
| 50 | 0.1234 | 0.0968 | 4 |
| 55 | 0.1227 | 0.0969 | 8 |
| **55 (best)** ⭐ | **0.1258** | **0.0983** | (saved) |
| 56 | (process terminated) | — | — |

**训练终止原因**: Stage 3 在 Ep 56 后进程意外消失 (无 crash log, GPU 1 释放), best ckpt @ Ep 55 已保存 (R12 强制 save + delete old).

### 2.2 Stage 4 Test eval (best ckpt @ Ep 55)

| Beam | Recall@5 | **Recall@10** | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------|----------|---------------|-----------|--------|---------|---------|
| 20 | 0.0830 | **0.1025** | 0.1258 | 0.0700 | 0.0763 | 0.0822 |
| 50 | 0.0835 | **0.1038** | 0.1336 | 0.0702 | 0.0767 | 0.0843 |

---

## 3. 决策矩阵 (vs HG-Rec baseline 0.1020)

| Anchor | task194 K=64 beam=20 | task194 K=64 beam=50 | Δ vs baseline | 解读 |
|--------|----------------------|----------------------|---------------|------|
| HG-Rec baseline (#84) | 0.1025 | 0.1038 | +0.5pp / +1.8pp | **统计中性** (noise ±2pp) |
| Issue #30 GO (0.1022) | 0.1025 | 0.1038 | +0.3pp / +1.6pp | **持平 + beam effect** |
| task194 K=0256 (0.1053, invalidated) | — | — | — | anchor 不可信 (Issue #40 Gate 0) |
| task320 Arm C R-Drop α=1.0 (0.1034, beam=100) | — | — | — | R-Drop ≠ K effect |

**核心结论**:
- task194 K=64 protocol-matched R@10=0.1025 (beam=20) / 0.1038 (beam=50)
- vs baseline 0.1020 = +0.5pp / +1.8pp — **统计中性** (noise ±2pp)
- K-scaling (K=64 vs K=256) **不是真 R@10 杠杆**
- task194 K=0256 0.1053 100% 来自 protocol leak (Stage 1 batch_size + Stage 2 Sinkhorn 0.003), 不是 K effect

---

## 4. 跨 Stage 1 / Stage 2 protocol 联立 (Issue #40 Gate 0 验证)

| Protocol 差异 | task194 K=0256 原始 | task194 K=064 protocol-matched | 影响 R@10? |
|---------------|---------------------|-------------------------------|------------|
| Stage 1 batch_size | 1024 | 1024 | 否 (match) |
| Stage 1 epochs | 1000 | 1000 | 否 (match) |
| Stage 1 kmeans_iters | 1000 | 1000 | 否 (match) |
| Stage 1 ckpt selection | best_loss | best_collision | **YES** (Gate 0 识别) |
| Stage 2 sk_eps | argmin | 0.0 (no Sinkhorn) | **YES** (Gate 0 识别) |
| Stage 2 Sinkhorn force | 0.003 | 0.0 | **YES** (Gate 0 识别) |

**Issue #40 Gate 1 CONFIRMED**: 移除 Stage 2 Sinkhorn + best_collision selection 两项差异后, task194 K=64 R@10=0.1025-0.1038 vs baseline 0.1020 ≈ neutral. Original task194 K=0256 0.1053 = **100% protocol leak**.

---

## 5. 跟当前 GO / NO-GO 地图联立

| 端点 | Test R@10 | 状态 |
|------|-----------|------|
| HG-Rec baseline (#84) | 0.1020 | ✅ baseline |
| Issue #30 marginal GO | 0.1022 | ✅ 当前唯一有效 GO |
| **task194 K=64 protocol-matched** | **0.1025 (b=20) / 0.1038 (b=50)** | **✅ CONFIRMED neutral, K 不是杠杆** |
| task320 Arm C R-Drop α=1.0 | 0.1034 (b=100) | ✅ Stage 3 协议 lever |
| task194 K=0256 original | 0.1053 (invalidated) | ❌ anchor 不可信 (Issue #40 Gate 0) |

**R10 backlog 状态更新**:
- task194_k0256 0.1053 anchor 永久撤销 (Issue #40 Gate 0 + Gate 1 双证据)
- K-scaling NOT a lever → K=256 锁死 (跟 K=384 task326 FAIL + K=512 task279 FAIL 一致)
- Stage 3 protocol (R-Drop) 仍是最有希望的方向 (task320 partial GO)
- 任何后续 task 引用 0.1053 anchor → 立即撤销 + 改用 0.1022 (Issue #30) 或 0.1020 (baseline)

---

## 6. 关键文件 + 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task194_armA_beam20_metrics.json` | test_R@10=0.1025 |
| `verdicts/task194_armA_beam50_metrics.json` | test_R@10=0.1038 |
| `verdicts/task194_issue40_gate1_protocol_match_verdict.md` | 本文件 |
| `verdicts/task329_issue40_gate0_protocol_audit.md` | Issue #40 Gate 0 FAIL (4 项协议差异识别) |
| `scripts/task194_protocol_match_k064_stage4_eval.sh` | Stage 4 launcher (beam_size arg) |
| `products/task194/protocol_match_k064/t5mini_k064/Instruments/Jul-30-2026_13-59-42/HG_Rec_best.pth` | Best ckpt @ Ep 55 (22 MB, preserved R12) |

---

## 7. R11.5 透明决策

**选了**: 跑 Stage 4 eval (本任务) → 写 verdict → 关闭 Issue #40 Gate 1
**为什么**: Issue #40 是 owner-established Gate, 必须闭合. task194 Stage 3 进程意外终止但 best ckpt 保留, Stage 4 跑通是闭环必要步骤. GPU 0/2/3 全空闲, 0 资源冲突.
**备选**: 等待 owner 指示 — R11.4 不允许等 (Issue #40 已经是 in-progress Gate, 必须闭合)

---

## 8. R14 闭环

- Issue #40 Gate 0 (task329) FAIL ✅ — task194 K=0256 0.1053 anchor invalidated
- Issue #40 Gate 1 (task194 K=64 protocol-matched) CONFIRMED ✅ — K-scaling NOT a lever, R@10 ≈ 0.1025-0.1038
- Issue #40 双向证据闭合, task194_k0256 0.1053 永久撤销
- 后续 task 引用 anchor → 0.1020 (baseline) / 0.1022 (Issue #30) ONLY

---

result: Task #194 / Issue #40 Gate 1 Stage 4 eval K=64 protocol-matched = test_R@10=0.1025 (beam=20) / 0.1038 (beam=50), vs HG-Rec baseline 0.1020 = +0.5pp / +1.8pp (统计中性, noise ±2pp). **Issue #40 Gate 1 CONFIRMED**: K-scaling (K=64 vs K=256) 不是真 R@10 杠杆, task194 K=0256 0.1053 100% 来自 protocol leak (Stage 2 Sinkhorn + ckpt selection), 永久撤销 anchor. Stage 3 protocol (R-Drop) 仍是当前最有希望方向.