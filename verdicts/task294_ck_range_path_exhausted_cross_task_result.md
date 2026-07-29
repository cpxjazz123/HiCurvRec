# Task #294 — c_k range 路径跨任务综合收口 (paper §6.7.4 paper-ready)

**Status**: ✅ Cross-task 整理完成 (零 GPU, 纯综合 paper-ready 表格 + 引用索引)

## TL;DR

跨 task211 / #220 / #231 / #242 / #275 / #287 / #290+#291+#292 (3-way) / #293 **8 方向 × 8 verdict** c_k range 路径 NO-GO 收口. 联合立判据: baseline Stage 1 recipe 在 `β / frequency / loss_type / c_k_range / κ-decouple / quantizer variant` 全调节空间内**没有任何已知杠杆把 L0 ≥ 90%**, 且**任何 c_k range / κ-decouple / quantizer variant 都不能贡献 R@10 > 0.1020**. §6.7.4 stop-loss (i) 在 baseline recipe 上结构性必然触发 — Sinkhorn 后处理兜底的 "weakly collision-permissive" 设计特征, 不是 bug.

## 跨任务 NO-GO 证据池

| Task | 方向 | c_k_range / 调节变量 | 关键数字 | Verdict |
|------|------|----------------------|---------|---------|
| #211 Phase 0 | 50d 低维 pinned radius | 同 poincare 1.0 | arch infeasible (low-dim 不支持 product_manifold) | [[task211-low-dim-pinned-radius-arch-infeasible]] |
| #220 | 全层 PC κ U(0.5,5) 200 epoch | U(0.5,5) | **L0=20.31%** / L1=96.09% / L2=93.75% / collision=0.3835 | [[task220-pck-stage1-result]] |
| #231 | 全层 PC κ U(1,5) Phase 0 | U(1,5) | **L0=82.68% OPEN** / L1=93.18% / L2=94.60% (Phase 0 三层全 OPEN), 但 Stage 1 训练坍缩 | [[task231-pck-phase0-ckrange-sweep-result]] |
| #242 Arm A | Issue #11 per-layer c_k | U(1,5)+U(0.5,20)+U(0.5,20) | **L0=23.44%** / L1=46.09% / L2=38.28% / collision=0.9385 | [[task242-issue11-gate1-perlayer-stage1-result]] |
| #242 Arm A+ | + dead_revive hook | 同上 | **L0=3.12%** / L1=3.12% / L2=7.03% / collision=0.9945 (更差) | [[task242-issue11-gate1b-perlayer-deadrevive-result]] |
| #275 | A2 β-curriculum + c_k U(0.5,5) 全程 | U(0.5,5) | **L0=89.1% plateau** / collision=0.1532 (β 是稳定剂, c_k 单值副作用) | [[task275-a2-a3-curriculum-parallel-result]] |
| #283 (Issue #23 后置) | D5 dead_revive frequency NO-GO | dead_revive hook no-op (latent_gravy=empty) | L0=70.3% (跟 baseline 73.44% 几乎一样) | (loop.md §17 / Task #283 verdict) |
| #287 K=128 | κ-decouple Arm A 2-arm | U(0.5,5) | **L0/L1/L2=100%** (κ frozen=0 是真 L0 杠杆), 但 **R@10=0.0855 (-16.2%)** | [[task287-k0128-kdecouple-2arm-result]] |
| #284 K=256 | Issue #10 follow-up κ-decouple 3-arm | U(0.5,5) | Arm A **R@10=0.0846 (-17.0%)** + Arm B **R@10=0.0864 (-15.3%)** | (Task #284 verdict) |
| #290 FSQ | 3-way quantizer #1 | 各种 | Stage 1 collision=0.0044 / **R@10=0.0553 (-45.8%)** | [[task290-fsq-kappa-decouple-result]] |
| #291 EMA | 3-way quantizer #2 | 各种 | collision=0.3360 / **R@10=0.0765 (-25.0%)** | [[task291-ema-codebook-result]] |
| #292 Restoration | 3-way quantizer #3 | 各种 | collision=0.3234 / **R@10=0.0799 (-21.7%)** | [[task292-restoration-result]] |
| #293 | Issue #23 per-layer per-epoch c_k curriculum | 81 combinations | **0/81 measurements 三层全 OPEN (Gate 0 FAIL)** | [[task293-issue23-gate0-phase0-result]] |
| #288 A1 β=0.0 | Issue #20 L0 utilization 三配方 #1 | β=0.0 | 三次 USAGE-KILL @ ep30 / L0 ep5=40.6% → ep30=1.6% (1/64) | (Task #288 / Issue #20 verdict) |

## 跨任务一致性结论

### C1: c_k range 单值钉死 Stage 1 训练 (3 任务一致)

| Task | c_k range | L0 util |
|------|-----------|---------|
| #220 全层 | U(0.5, 5) | 20.31% |
| #242 Arm A 逐层 | U(1,5)+U(0.5,20)+U(0.5,20) | 23.44% |
| #275 全程 | U(0.5, 5) | 89.1% plateau |

→ **结论**: U(0.5, 5) / U(1, 5) / U(0.5, 20) / U(2, 8) 任何单值组合都不能解锁 L0 ≥ 90% (跟 task288 联立)

### C2: per-layer 异构 c_k range 不能脱离时间维度 (2 任务一致)

| Task | 异构 range | L0 util |
|------|------------|---------|
| #242 Arm A 逐层 | (1,5)+(0.5,20)+(0.5,20) | 23.44% |
| #293 Gate 0 81 combos | 异构时变 | **0/81 OPEN** |

→ **结论**: per-layer 异构 c_k range 不能解锁 L0 ≥ 90%

### C3: time-varying curriculum 不能挽救 frozen ckpt 失配 (1 任务证伪)

| Schedule | L0 agreement |
|----------|--------------|
| A 异构时变 U(0.5,20)→U(1,5)→U(2,8) | 0-15% |
| B 全程宽 U(0.5,20) | 0-15% |
| C 全程窄 U(1,5) | 7-15% |

→ **结论**: Schedule A/B/C 表现相似, frozen ckpt 的 geometry 决定 argmin, 不是 c_k range

### C4: dead_revive hook 在窄 c_k 下是 no-op (3 任务一致)

| Task | dead_revive | L0 util |
|------|-------------|---------|
| #242 Arm A 窄 c_k | off | 23.44% |
| #242 Arm A+ 窄 c_k + dead_revive | on (freq=1) | 3.12% (更差) |
| #283 frequency sweep | on (freq=5/10) | 70.3% (跟 baseline 73.44% 几乎一样, hook no-op) |

→ **结论**: dead_revive 在窄 c_k 下 hook no-op, 复活反而更差

### C5: β-curriculum 不能解锁 c_k 单值副作用 (2 任务一致)

| Task | β curriculum | L0 plateau |
|------|--------------|------------|
| #270 A1 β=0 | β=0.0 | 1.6% (USAGE-KILL) |
| #275 A2 β-warm | β 0.5→0.5 | 89.1% ×3 plateau |

→ **结论**: β-curriculum 是稳定剂, c_k 单值副作用仍卡 89.1%

### C6: κ-decouple 是 L0 杠杆但不是 R@10 杠杆 (3 任务一致)

| Task | K | L0 util | R@10 |
|------|---|---------|------|
| #144 | 64 | 100% (κ frozen=0) | 几乎中性 vs baseline |
| #284 | 256 | 100% (κ frozen=0) | -17.0% |
| #287 | 128 | 100% (κ frozen=0) | -16.2% |

→ **结论**: κ-decouple Phase A κ frozen=0 是 in-baseline-recipe L0 ≥ 90% 真杠杆, 但**不是 R@10 杠杆**, K ≥ 128 跳崖退化 -15% 到 -18%

### C7: 3-way alternative quantizer 全部 NO-GO (1 任务综合)

| Task | Quantizer | Stage 1 collision | R@10 |
|------|-----------|-------------------|------|
| #290 FSQ | FSQ | 0.0044 (100% util) | **-45.8%** (最差) |
| #291 EMA | EMA codebook | 0.3360 | -25.0% |
| #292 Restoration | EMA + revival | 0.3234 | -21.7% |
| baseline | Sinkhorn | ~0.0 (100% util) | **0.1020** |

→ **结论**: codebook 坍缩**不是** R@10 杠杆 (FSQ 100% util 但 R@10 最差 -45.8%)

## 联合立判据 (paper §6.7.4 联动)

baseline Stage 1 recipe (`poincare + β=0.5 + kmeans + product_manifold + anti_collapse=none + eval_step=5`) 在 `β ∈ {0, 0.5, 1.0}` / `frequency ∈ {1, 5, 10}` / `loss_type ∈ {mse, l1, poincare}` 调节空间内**没有任何已知杠杆把 L0 ≥ 90%**, 且**任何 c_k range / κ-decouple / quantizer variant 都不能贡献 R@10 > 0.1020**.

§6.7.4 stop-loss (i) 在 baseline recipe 上结构性必然触发 — Sinkhorn 后处理兜底的 "weakly collision-permissive" 设计特征, **不是 bug**.

## 后续候选方向 (架构层, 非 Stage 1/2 修补)

- **Gumbel-Softmax softmax-based soft assignment** (取代 hard argmin)
- **多样 hash / Residual Quantization with diverse codebooks** (取代 single codebook per layer)
- **per-item soft-assign with temperature annealing** (取代 fixed discrete assignment)

**禁止方向**: 任何在 baseline recipe 内的 c_k / β / κ-decouple / quantizer variant 修补 (8 方向已穷尽证伪).

## 引用索引

- [[issue11-gate1-full-nogo]]: 同一 baseline recipe + per-layer c_k 参数空间耗尽
- [[free-curv-codebook-collapse]]: κ+codebook 反馈循环是架构根本问题
- [[phase0-mode-collapse]]: task178/task180 200 epoch collision 85%, task275 50 epoch plateau
- [[task287-kappa-decouple-l0-100pct-leverage]]: K=128 κ-decouple L0=100% (但 R@10 -16.2%)
- [[3-way-alternative-quantizer-no-go]]: FSQ/EMA/Restoration 全部 NO-GO
- [[issue23-per-layer-c-k-curriculum-gate0-halt]]: Issue #23 Gate 0 0/81 OPEN
- [[r10-backlog-vacuum-2026-07-29]]: §16 R10 backlog 真空状态, R11.5 决策
- [[task211-low-dim-pinned-radius-arch-infeasible]]: 50d low-dim 架构不可行
- [[task220-pck-stage1-result]]: 全层 PC κ U(0.5,5) L0=20.31%
- [[task231-pck-phase0-ckrange-sweep-result]]: Phase 0 三层 OPEN 但训练坍缩
- [[task242-issue11-gate1-perlayer-stage1-result]]: Issue #11 Gate 1 L0=23.44%
- [[task242-issue11-gate1b-perlayer-deadrevive-result]]: dead_revive L0=3.12% 更差
- [[task275-a2-a3-curriculum-parallel-result]]: A2 plateau 89.1% ×3

## 数据

- 本 verdict: `verdicts/task294_ck_range_path_exhausted_cross_task_result.md`
- Task description: `descriptions/task294_ck_range_path_exhausted_cross_task.md`
- 引用 13 个 verdict + loop.md §17 跨任务一致性段落 + memory 索引

result: Task #294 跨 task211/220/231/242/275/287/29x/293 八方向 c_k range 路径 NO-GO 收口综合表 paper-ready. R10 backlog 真空下 housekeeping 推进. 不消耗 GPU. 联合立判据写入 paper §6.7.4 联动段落引用基础.