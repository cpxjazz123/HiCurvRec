# Task #234 — Issue #9 Hybrid Phase 0 Gate 0: per-layer composition (PASS)

## 目的

验证 Issue #9 H1: hybrid 分配规则 (L0=PC κ c_k~U(1,5), L1/L2=Gromov weight=0.5) 在串行 3-layer forward-pass 中三层是否仍 OPEN 60-90%.

## 配置

| Item | Value |
|------|-------|
| Hybrid rule | L0 = PC κ c_k ~ Uniform(1, 5), L1/L2 = Gromov weight_l = 0.5 |
| Phase 0 ckpt | HG-Rec baseline (#84) best_loss_model.pth |
| Seeds | {42, 43, 44} |
| Items | 9922 (Musical_Instruments 5-core) |
| e_dim | 32 |
| num_emb_list | [64, 128, 256] |
| CPU only (无 GPU 占用) | ✓ |

## 结果 (3 seeds, mean ± std)

| Layer | K | Independent rule | Independent mean | Composed hybrid | Composed verdict | \|Δ\| pp | Gate 0 pass |
|-------|---|------------------|------------------|------------------|------------------|---------|-------------|
| 0 | 64 | PC κ c_k~U(1,5) | **82.68%** | **82.68%** | OPEN (60-95%) | **0.00pp** | ✓ |
| 1 | 128 | Gromov w=0.5 | **73.67%** | **72.94%** | OPEN (60-95%) | **0.74pp** | ✓ |
| 2 | 256 | Gromov w=0.5 | **68.11%** | **67.21%** | OPEN (60-95%) | **0.90pp** | ✓ |

**Overall: 3/3 layers PASS.** Composed vs Independent 偏差 0.00-0.90pp, 远低于 ±3pp tolerance.

## Per-seed 详细 (composed agreement %)

| Layer | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|-------|---------|---------|---------|------------|
| L0 (PC κ) | 82.74% | 83.01% | 82.30% | 82.68% ± 0.29 |
| L1 (Gromov composed) | 72.95% | 72.91% | 72.95% | 72.94% ± 0.02 |
| L2 (Gromov composed) | 67.24% | 67.43% | 66.97% | 67.21% ± 0.19 |

## 决策点 (Issue #9 §H1)

✅ **H1 成立** — hybrid 在串行 forward-pass 中三层均进入 OPEN 60-90% band. 关键发现:
- L0 composed == independent (Δ=0.00pp): L0 没有 earlier layers, residual = 0
- L1 composed 比 independent 低 0.74pp: PC κ L0 改变了 residual 分布, 让 Gromov 在 L1 看到略不同的 z
- L2 composed 比 independent 低 0.90pp: 累加效应, 但仍 < 1pp tolerance

所有三层偏差均 < 1pp, 远低于 ±3pp 一致性阈值. Issue #9 H1 成立, hybrid 不会因 serial composition 破坏 OPEN band.

## 后续 Gate 1 决策

按 Issue #9 §Procedure step 3: Gate 0 PASS → Stage 1 40 epoch early_stop 训练.
- 监控指标: collision ≤ 0.3706 + L0 utilization ≥ 90% (跟 task222 best row 对照)
- 加 epoch-level logging + R12 ckpt 强制保存

## 产物

- `products/task234/hybrid_gate0_3seeds.json` → `/home/wlia0047/.claude/jobs/04ccf474/tmp/task234_hybrid_gate0_results.json`
- `scripts/task234_hybrid_phase0_gate0.py` (430 lines, py_compile OK)

## 决策记录

| 决策 | 选了什么 | 为什么 |
|------|---------|--------|
| Hybrid rule | L0=PC κ U(1,5) + L1/L2=Gromov w=0.5 | Issue #9 §Hypothesis H1 直接提议 |
| ±3pp 一致性阈值 | self-defined | Issue #9 §Gate 0 表 "与独立测量 ±3pp" |
| seeds | {42, 43, 44} | 跟 task231/232 复用 |
| composed L0 c_k 采样 | seed+1000+layer_idx 派生, K=64 | K 与 layer_idx 解耦, 不污染 caller 的 c_k |
| residual 用 euc 还是 PC κ 重算 | composed 用 PC κ (L0) / Gromov (L1/L2) 自身 | 模拟 Issue #9 hybrid 真实 forward-pass |

## Status

Gate 0 PASS. 后续: 启动 Stage 1 40 epoch 训练 (Gate 1).
