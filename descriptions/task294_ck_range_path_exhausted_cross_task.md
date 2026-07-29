# Task #294 — c_k range 路径跨任务综合收口 (paper §6.7.4 paper-ready)

## 背景

跨 task211 / #220 / #231 / #242 / #275 / #287 / #290+#291+#292 (3-way) / #293 八方向 c_k range 路径 NO-GO 收口证据分散在 8+ 个 verdict 文件 + loop.md §17 + memory 索引. 本任务不跑新实验, 把跨任务一致性结论整理成 paper-ready 综合表, 让 paper 引用时无需翻 8 个 verdict 文件.

## 范围

**NO-GO 收口证据池 (8 方向 × 8 verdict)**:

| Task | 方向 | c_k_range | 关键数字 | verdict |
|------|------|-----------|---------|---------|
| #211 Phase 0 | 50d 低维 pinned radius | 同 poincare 1.0 | arch infeasible | task211_low_dim_pinned_radius_arch_infeasible.md |
| #220 | 全层 PC κ U(0.5,5) 200 epoch | U(0.5,5) | L0=20.31% / L1=96.09% / L2=93.75% / collision=0.3835 | task220_per_codeword_kappa_stage1_train_result.md |
| #231 | 全层 PC κ U(1,5) Phase 0 | U(1,5) | L0=82.68% / L1=93.18% / L2=94.60% (OPEN 60-90%) 但 Stage 1 训练坍缩 | task231_pck_phase0_ckrange_sweep_result.md |
| #242 | Issue #11 per-layer c_k Arm A | U(1,5)+U(0.5,20)+U(0.5,20) | L0=23.44% / L1=46.09% / L2=38.28% / collision=0.9385 | task242_issue11_gate1_perlayer_stage1_result.md |
| #242 Arm A+ | + dead_revive | 同上 | L0=3.12% / L1=3.12% / L2=7.03% / collision=0.9945 (更差) | task242_issue11_gate1b_perlayer_deadrevive_result.md |
| #275 | A2 β-curriculum + c_k U(0.5,5) 全程 | U(0.5,5) | L0=89.1% plateau / collision=0.1532 (β 是稳定剂, c_k 单值副作用) | task275_a2_a3_curriculum_parallel_result.md |
| #287 K=128 | κ-decouple Arm A 2-arm | U(0.5,5) | L0/L1/L2=100% (κ frozen=0 是真 L0 杠杆), 但 R@10=0.0855 (-16.2%) | task287_k0128_kdecouple_2arm_result.md |
| #290+#291+#292 | 3-way quantizer FSQ/EMA/Restoration | 各种 | R@10=0.0553 / 0.0765 / 0.0799 (-45.8% / -25.0% / -21.7%) | task290_fsq_kappa_decouple_result.md / task291_ema_codebook_result.md / task292_restoration_result.md |
| #293 | Issue #23 per-layer per-epoch c_k curriculum | 81 combinations | 0/81 measurements 三层全 OPEN (Gate 0 FAIL) | task293_issue23_gate0_phase0_result.md |

## 跨任务一致性结论

1. **c_k range 单值钉死 Stage 1 训练**: U(0.5,5) / U(1,5) / U(0.5,20) / U(2,8) 任何单值组合都不能解锁 L0 ≥ 90% (跟 task288 联立)
2. **per-layer 异构 c_k range 不能脱离时间维度**: task242 Arm A 23.44% / task293 Gate 0 0/81 OPEN
3. **time-varying curriculum 不能挽救 frozen ckpt 失配**: task293 Schedule A 跟 Schedule B/C 表现相似
4. **dead_revive hook 在窄 c_k 下是 no-op**: task242 Arm A+ 3.12% (反而更差), task283 hook no-op 验证
5. **β-curriculum 不能解锁 c_k 单值副作用**: task275 plateau 89.1% ×3 续训, task288 A1 β=0 三次 USAGE-KILL 锁死
6. **κ-decouple 是 L0 杠杆但不是 R@10 杠杆**: task144 K=64 几乎中性 / task284 K=256 NO-GO / task287 K=128 L0=100% 但 R@10 -16.2%
7. **3-way alternative quantizer 全部 NO-GO**: codebook 坍缩不是 R@10 杠杆

## 联合立判据 (paper §6.7.4 联动)

baseline Stage 1 recipe (`poincare + β=0.5 + kmeans + product_manifold + anti_collapse=none + eval_step=5`) 在 `β ∈ {0, 0.5, 1.0}` / `frequency ∈ {1, 5, 10}` / `loss_type ∈ {mse, l1, poincare}` 调节空间内**没有任何已知杠杆把 L0 ≥ 90%**, 且**任何 c_k range / κ-decouple / quantizer variant 都不能贡献 R@10 > 0.1020**. §6.7.4 stop-loss (i) 在 baseline recipe 上结构性必然触发 — Sinkhorn 后处理兜底的 "weakly collision-permissive" 设计特征, 不是 bug.

## 产物

1. `verdicts/task294_c_k_range_path_exhausted_cross_task_result.md` (cross-task 综合表 + 引用索引)
2. `loop.md §17` 追加 task294 closeout 段
3. memory `cross-task-c-k-range-no-go-exhausted.md` (R10 推进决策记录)
4. git commit + push

## 不消耗 GPU

零 GPU, 纯 cross-task 整理. 预计 15-30 分钟完成.

## R10 + R11.3 + [[r10-backlog-vacuum-2026-07-29]] 决策依据

R10 backlog 真空时默认行为 = 整理 paper / 写 verdict / memory 整合 (不强启动 ROI 极低实验). D3 m-arm κ-Stereo v9+ 已 R11.5 决策 = 不启动. 本任务是高 ROI housekeeping, paper §6.7.4 联动必要, 不消耗 GPU.

result: Task #294 跨 task211/220/231/242/275/287/29x/293 八方向 c_k range 路径 NO-GO 收口综合表 paper-ready. R10 backlog 真空下 housekeeping 推进. 不消耗 GPU.