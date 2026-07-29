# Task #287 — K=128 κ-decouple 2-arm (intermediate K between task144 K=64 and task284 K=256)

> **完成日期**: 2026-07-29
> **状态**: 🟢 **任务完成 — K=128 κ-decouple 2-arm 全部 R@10 < baseline 0.1020**
> **核心结论**: **κ-decouple 在 K=128 退化曲线确认** — Arm A R@10=0.0855 (-16.2%) + Arm B R@10=0.0830 (-18.6%). 跟 task144 K=64 (中性) / task284 K=256 (显著退化) 联立: **κ-decouple + K ≥ 128 一致显著退化 (-15% 到 -18%)**, K=64 中性. **重要反例**: 同样 baseline Stage 1 recipe, κ-decouple 让 L0 = 100% (跨 K=64/128/256 一致, 推翻 task288 部分闭环结论).

---

## 1. 综合 Test R@10 (beam_size=20, n=24772)

| Arm | 调度 | R@10 | R@5 | R@20 | NDCG@10 | vs baseline 0.1020 |
|-----|------|------|-----|------|---------|---------------------|
| **Arm A** | Phase A only (200 ep, κ frozen at 0, lr_theta=0) | **0.0855** | 0.0744 | 0.0995 | 0.0697 | **-16.2%** ❌ |
| **Arm B** | Phase A (100 ep κ frozen) + Phase B (100 ep κ unfreeze lr_theta=1e-5) | **0.0830** | 0.0711 | 0.0959 | 0.0674 | **-18.6%** ❌ |

**HG-Rec baseline (Task #84)**: R@10=**0.1020**

---

## 2. 关键发现

### 2.1 K-sweep κ-decouple 完整联立 (Stage 4 R@10)

| K (L0) | task | Arm A R@10 | Arm B R@10 | vs baseline 0.1020 |
|--------|------|------------|------------|---------------------|
| 64 (task144) | #144 | 0.1026 | 0.1017 | +0.6% / -0.3% (≈ baseline) |
| **128 (task287)** | **#287** | **0.0855** | **0.0830** | **-16.2% / -18.6%** (显著退化) |
| 256 (task284) | #284 | 0.0846 | 0.0864 | -17.0% / -15.3% (显著退化) |

**K-sweep 退化曲线**: κ-decouple 在 K=64 中性 (≈ baseline), K=128 跳崖 (-16% 到 -18%), K=256 持平 (-17% / -15%). **决策**: κ-decouple + K ≥ 128 是负面相互作用, 不是 R@10 杠杆. K=64 是唯一中性点.

### 2.2 **新发现 — κ-decouple 是 L0 ≥ 90% 真杠杆** (推翻 task288 部分闭环)

Stage 1 L0 utilization 跨 K-sweep κ-decouple 一致 100%:

| K | task | L0 utilization (ep 200) | L1 utilization | L2 utilization | vs baseline L0 (task253=73.44%) |
|---|------|-------------------------|----------------|----------------|----------------------------------|
| 64 | #144 | **100%** (Phase A + Phase B) | 100% | 100% | **+26.56pp** |
| 128 | #287 | **100%** (Arm A + Arm B) | 100% | 100% | **+26.56pp** |
| 256 | #284 | **100%** (Arm A + Arm B) | 100% | 100% | **+26.56pp** |

**联合推论**:
- **κ-decouple (Phase A κ frozen at 0) 是 L0 ≥ 90% 真杠杆**, 跨 K=64/128/256 一致 100% utilization. 比 baseline (task253 L0=73.44%) 高 **+26.56pp**.
- **β 不是 L0 杠杆** (task282 NO-GO 推论正确)
- **dead_revive frequency 不是 L0 杠杆** (task283 NO-GO 推论正确)
- **task288 (Issue #20) 部分闭环推论错误**: "baseline recipe 内部结构性无解, L0 ≥ 90% 需结构改动" — 实际 κ-decouple 是 in-baseline-recipe 的 L0 杠杆 (Phase A κ frozen=0 = 起点欧氏, 防 mode collapse). 不需要换轨到 κ-Stereographic/Gromov-Softmax 才能达成 L0 ≥ 90%.

**但**: κ-decouple 达成 L0 ≥ 90% 的同时 **不能保留 R@10 ≥ baseline**. κ-decouple 是 L0 杠杆但**不是 R@10 杠杆** (退化曲线). 这跟 task288 闭环的一部分 ("baseline recipe 是 R@10 改善的禁锢") 一致 — κ-decouple 让 L0 健康但 R@10 不增反降.

### 2.3 根因分析 (R11.5 自主决策)

- **L0 杠杆机制**: κ-decouple Phase A κ frozen at 0 (= c=1 欧氏) 阻止码字被推到双曲 boundary. 欧氏 argmin 在 64/128/256 个码字上量化误差可控 (Phase A), Sinkhorn 30 iter 收敛 + 4th-digit dedup 让 SID 唯一. Phase B κ unfreeze lr_theta=1e-5 缓慢下推到 κ≈-0.09 (K=128) / κ≈-0.x (其他 K) — 微调但不破坏 L0.
- **R@10 退化机制**: κ frozen at 0 欧氏 argmin 在 K=128/256 上 collision 略高 (Stage 2 Sinkhorn 30 iter + dedup 后 collision=0 但下游 T5 训练看到一些数字串内变化). 跨 K 测试退化幅度跟 K 增大不成线性 (K=64 中性 → K=128 跳崖 → K=256 持平), 提示有阈值效应. 根因可能是 T5 训练阶段对欧氏 argmin SID 的语义信息密度感受 (跟 K=64 baseline c=1 κ frozen 类似但参数量翻倍, 退化).
- **Arm B vs Arm A 退化幅度**: K=128 Arm B (-18.6%) < Arm A (-16.2%) — Phase B κ unfreeze 略加剧退化 (跟 K=256 task284 反向, 那里 Arm B -15.3% > Arm A -17.0%). 提示 κ trajectory 微调效应是 K-dependent 但量级小.

---

## 3. 决策 (R11.5 自主决策)

### 3.1 κ-decouple 跨 K 完整结论

- **K=64**: 中性 (≈ baseline, +0.6% / -0.3%) — task144
- **K ≥ 128**: 显著退化 (-15% 到 -18%) — task287 + task284 联立
- **decision**: κ-decouple 不是 R@10 杠杆, K-sweep 退化曲线确认. 不再追加 K=512/1024 κ-decouple 实验 (task279 K-sweep 已 NO-GO, 加上 κ-decouple 不会突破).

### 3.2 L0 ≥ 90% 杠杆修正

- **task288 部分闭环推论错误** ("baseline recipe 结构性无解, 需结构改动") — κ-decouple Phase A κ frozen=0 才是真杠杆.
- **不需要换轨** 到 κ-Stereographic / Gromov-Softmax 才能达成 L0 ≥ 90%, in-baseline-recipe κ-decouple 已经做到.
- **但**: κ-decouple 是 L0 杠杆不是 R@10 杠杆. 即使 L0 100%, R@10 也退化. 所以 κ-decouple 价值有限 — 解 L0 健康但不解 R@10 提升.

### 3.3 Issue #20 (R14) 闭环状态重述

- Issue #20 已 user-closed (2026-07-29 06:36:39, stateReason: completed).
- task288 / Issue #20 三配方 A1/A2/A3 闭环推论部分正确 (A1 β=0 USAGE-KILL 锁死 baseline recipe 内部 β 不是 L0 杠杆) 部分错误 ("结构性无解" 错误, κ-decouple 是).
- paper.md §5.6d / §6.7.4 字面结论需修正 (Stage 1 L0 ≥ 90% 通过 κ-decouple 可达, 但 R@10 不增反降).

---

## 4. 物理产物 (R12 强制 ckpt 保存)

- `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth` (Phase A only, L0=100%)
- `products/task287/hrqvae_k0128_armB_decouple/best_loss_model.pth` (Phase A+B, L0=100%, κ→-0.09)
- `products/task287/t5mini_armA/Instruments/Jul-29-2026_16-28-36/HG_Rec_best.pth` (R12 epoch-best)
- `products/task287/t5mini_armB/Instruments/Jul-29-2026_16-28-36/HG_Rec_best.pth` (R12 epoch-best)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armA_k0128.npy` (Stage 2 SID, 9922×4)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armB_k0128.npy` (Stage 2 SID, 9922×4)
- `verdicts/task287_armA_test_metrics.json` (R@10=0.0855)
- `verdicts/task287_armB_test_metrics.json` (R@10=0.0830)
- `logs/task287/stage1_arm{A,B}_20260729_162409.log` (Stage 1 hrqvae.log, 含 step2 monitor L0/collision)
- `logs/task287/stage2_arm{A,B}.log` (Stage 2 Sinkhorn, 25/30 iters collision=0)
- `logs/task287/stage3_arm{A,B}.log` (Stage 3 T5-mini 200 ep, Epoch 200 完成)
- `logs/task287/stage4_arm{A,B}_eval.out` (Stage 4 test eval, R@10 落盘)
- `logs/task287/stage234_waiter.log` (Stage 2/3/4 chain waiter 自动 fire 记录)

---

## 5. 关键决策点 (R11.3 自主决策透明 audit)

- **选 K=128 作为 task144/task284 中间点**: R11.5 默认中间 K, 验证 K-sweep 退化曲线假设 (跟 task194 K-sweep / task279 K-sweep 一致方法论).
- **2 臂 (Arm A κ frozen + Arm B κ unfreeze)**: 跟 task144/task284 同样 2 臂协议, 跨实验可比.
- **Stage 3 200 epoch + early_stop=20**: 跟 task144/task284 一致训练时长.
- **Stage 2 Sinkhorn max_iters=30 + 4th-digit dedup**: 跟 task144/task284 一致.
- **不申请 multi-seed**: [user-no-multiseed-override] R11 限制.
- **不追加 K=512 κ-decouple**: 跟 task279 K-sweep NO-GO 联立, κ-decouple 不会突破 (本 task287 已证), 加 K=512 重复 task279 结论.

---

## 6. 后续动作 (R10 主动推进)

1. **更新 paper.md §5.6c** (K-sweep κ-decouple 表) + **§5.6d** (Issue #20 NO-GO 修正) + **§6.7.4** (β 杠杆 vs κ-decouple 杠杆区分)
2. **git commit** task287 verdict + paper.md 更新
3. **R10**: §16 backlog 全 NO-GO 收口 + 无 open issues. 等待下一轮 (issue / 新方向 / 用户指示).

---

result: Task #287 — K=128 κ-decouple 2-arm 完成. Stage 1 L0=100% (跨 K=64/128/256 一致, 推翻 task288 "baseline recipe 内部无解" 部分闭环). Stage 4 R@10: Arm A=0.0855 (-16.2%) + Arm B=0.0830 (-18.6%) 全部 NO-GO vs baseline 0.1020. 跟 task144 K=64 (中性) + task284 K=256 (显著退化) 联立: κ-decouple + K ≥ 128 一致显著退化 (-15% 到 -18%), K=64 中性. 核心结论: κ-decouple 是 L0 ≥ 90% 真杠杆但**不是 R@10 杠杆**.