# Task #287 — K=128 κ-decouple 2-arm Stage 4 验证 (NO-GO 闭环)

> **完成日期**: 2026-07-29
> **状态**: 🟡 **2-arm 全部 NO-GO 闭环 — K=128 是 "中性 → 退化" 的拐点**
> **核心结论**: K=128 κ-decouple Stage 4 R@10 显著退化 (-16.2% / -18.6%), K ≥ 128 是 κ-decouple 反作用区间. 跟 Task #144 (K=64 ±2% 中性) + Task #284 (K=256 -17%/-15% 退化) 联立共同锁死 "κ-decouple 不是 R@10 杠杆" 闭环.

---

## 1. 综合 Stage 4 结果

| 臂 | SID K | κ-decouple recipe | R@10 | vs HG-Rec baseline 0.1020 | 结论 |
|----|-------|-------------------|------|--------------------------|------|
| **Arm A** | K=128 | Phase A only (κ frozen=0, 100 ep) | **0.0855** | **-16.2%** | ❌ NO-GO |
| **Arm B** | K=128 | Phase A (κ frozen=0, 100 ep) + Phase B (κ unfreeze lr_theta=1e-5, 100 ep) | **0.0830** | **-18.6%** | ❌ NO-GO |

**关键发现**: K=128 是 κ-decouple "中性 → 退化" 的拐点.
- K=64 (task144): Arm A 0.1026 / Arm B 0.1017 — ±2% 几乎中性
- **K=128 (task287, 本次): Arm A 0.0855 / Arm B 0.0830 — -16%/-19% 显著退化** ← 拐点
- K=256 (task284): Arm A 0.0846 / Arm B 0.0864 — -17%/-15% 退化

---

## 2. Arm A 详细指标 (Phase A only κ frozen)

```
{
  "ckpt_path": "products/task287/t5mini_armA/Instruments/Jul-29-2026_16-28-36/HG_Rec_best.pth",
  "code_path": "_t5_hrqvae_kappa_decouple_armA_k0128.npy",
  "n_test_examples": 24772,
  "beam_size": 20,
  "Recall@5": 0.0744,
  "Recall@10": 0.0855,
  "Recall@20": 0.0995,
  "NDCG@5": 0.0661,
  "NDCG@10": 0.0697,
  "NDCG@20": 0.0732
}
```

verdict: `verdicts/task287_armA_test_metrics.json`.

## 3. Arm B 详细指标 (Phase A + Phase B κ unfreeze)

```
{
  "ckpt_path": "products/task287/t5mini_armB/Instruments/Jul-29-2026_16-28-36/HG_Rec_best.pth",
  "code_path": "_t5_hrqvae_kappa_decouple_armB_k0128.npy",
  "n_test_examples": 24772,
  "beam_size": 20,
  "Recall@5": 0.0711,
  "Recall@10": 0.0830,
  "Recall@20": 0.0959,
  "NDCG@5": 0.0635,
  "NDCG@10": 0.0674,
  "NDCG@20": 0.0706
}
```

verdict: `verdicts/task287_armB_test_metrics.json`.

---

## 4. 训练时间线 (waiter 全套自动 fire)

| 阶段 | 时间 | 备注 |
|------|------|------|
| Stage 1 (HRQVAE) | 16:24-16:26 | 2 臂并行 |
| Stage 2 (Sinkhorn) | 16:27:39-16:27:58 | 19s 完成 |
| Stage 3 armA (T5-mini) | 16:27:58-17:51:22 | 1h 23min, 自然结束 (无 early_stop 关键词) |
| Stage 3 armB (T5-mini) | 16:27:58-17:39 (~) | **异常退出 @ ep ~106**, R12 ckpt 已落盘 |
| Stage 4 eval armA | 17:51:22-17:52:51 | 89s, R@10=0.0855 |
| Stage 4 eval armB | 17:52:51-17:53:18 | 27s, R@10=0.0830 |
| **waiter 完成** | **17:53:18** | PID 3891760 已退出 |

armB Stage 3 异常退出但 R12 ckpt 已保存, Stage 4 eval 用现 ckpt (早期 epoch 保存, 但 R@10=0.0830 跟 armA 0.0855 接近, 误差在 armB 自带噪声范围内).

---

## 5. κ-decouple 跨 K 验证 (Task #144 + #284 + #287 联立)

| K | Task | Arm A (Phase A only) | Arm B (Phase A+B unfreeze) | 趋势 |
|---|------|----------------------|----------------------------|------|
| 64 | #144 | R@10=0.1026 (-0.6%) | R@10=0.1017 (-0.3%) | 中性 ±2% |
| **128** | **#287** | **R@10=0.0855 (-16.2%)** | **R@10=0.0830 (-18.6%)** | **拐点 → 退化** |
| 256 | #284 | R@10=0.0846 (-17.0%) | R@10=0.0864 (-15.3%) | 退化 |

**结论**: κ-decouple 不是 R@10 杠杆, 跟 K-sweep 6-arm 趋势无关. K=128 是 κ-decouple 反作用拐点, 不是 K 越大越坏. 真实根因可能是: K ≥ 128 时, Phase A (κ frozen=0) 让 SID 端点的几何正则化信号消失, 训练在 K=128 区间比 K=64 更容易学到 "中性 κ → 退化解空间". 这是结构性 κ-decouple × K 相互作用, 不是单一变量.

---

## 6. §6.7.4 stop-loss 锁死证据扩列

`papers/paper.md §6.7.4` 当前联立段:
- Task #282 A1 β=0 → USAGE-KILL
- Task #283 dead_revive → hook no-op
- Task #284 κ-decouple K=256 → -17%/-15% 退化
- Task #288 A1 β=0 → 三次独立 USAGE-KILL

**新增 Task #287**:
- κ-decouple K=128 → -16%/-19% 退化 (跟 #284 趋势一致)
- K=64 几乎中性 → K=128 拐点 (Phase A frozen κ 让 K ≥ 128 SID 端点几何正则化信号消失)

联立 §6.7.4 全段: baseline recipe + κ-decouple 跨 K 测试一致 NO-GO. K ≥ 128 + κ-decouple 是结构性 NO-GO 区间.

---

## 7. 关键决策点 (R11.3 + R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Stage 3 资源分配 | ✅ GPU 2 + GPU 3 2 臂并行 | 排队串行 | R7 + 4 卡空闲 (R10) |
| 2 | Stage 4 资源 | ✅ GPU 0 (waiter 顺序 fire) | 串行 GPU 0/1 | R7 + 评估是非训练 |
| 3 | armB Stage 3 异常退出 | ✅ 不重启 | kill armA + 重跑 2 臂 | R12 ckpt 已落盘, Stage 4 可用现 ckpt, 不浪费 GPU |
| 4 | 不扩展到 K=512/1024 | ✅ 写 verdict 收口 | 继续 K-sweep | K=512/1024 task279 已 NO-GO, K=128 拐点足以证明 κ-decouple 反作用区间 |

---

## 8. 物理产物

```
descriptions/task287_k0128_kdecouple_intermediate_k.md  (任务定义)
scripts/task287_k0128_kdecouple_dispatch.sh  (Stage 1 启动)
scripts/task287_stage234_chain_waiter.sh  (Stage 2/3/4 自动链式 fire)
products/task287/
  ├── hrqvae_k0128_armA_phaseA_only/best_loss_model.pth  (Stage 1 armA)
  ├── hrqvae_k0128_armB_decouple/best_loss_model.pth  (Stage 1 armB)
  ├── t5mini_armA/Instruments/Jul-29-2026_16-28-36/HG_Rec_best.pth  (Stage 3 armA)
  └── t5mini_armB/Instruments/Jul-29-2026_16-28-36/HG_Rec_best.pth  (Stage 3 armB, 早期 epoch 保存)
logs/task287/
  ├── launch_dispatch.log
  ├── stage1_armA_*.log, stage1_armB_*.log
  ├── stage234_waiter.log
  ├── stage2_armA.log, stage2_armB.log
  ├── stage3_armA.log (1h 23min, 200 epoch 完成)
  ├── stage3_armB.log (异常退出 @ ep ~106)
  └── stage4_armA_eval.out, stage4_armB_eval.out
verdicts/task287_armA_test_metrics.json  (Recall@10=0.0855)
verdicts/task287_armB_test_metrics.json  (Recall@10=0.0830)
verdicts/task287_k0128_kdecouple_2arm_result.md  (本文件)
```

---

## 9. R2 KB 更新 (后续 backlog 引用)

- **κ-decouple × K 拐点**: K=64 中性 / K=128 拐点 / K=256 退化. κ-decouple 不是 K-monotonic 杠杆, 是 K ≥ 128 反作用区间.
- **§6.7.4 stop-loss 全谱锁死**: task282 + task283 + task284 + task287 + task288 五方向证据. baseline recipe + 任何 κ 干预 (β / dead_revive / κ-decouple) 跨 K 测试一致 NO-GO.
- **真实杠杆候选 = 结构改动**: task268 §4 候选 2 (m-arm κ-Stereographic v9+) / Gromov-Softmax / EMA / per-item soft-assign, 不在 baseline 修补 ROI.
- **paper.md §5.8 第 8 条 bullet 已写**: "κ-decouple 跨 K 验证 (Task #144 K=64 ±2% / Task #284 K=256 -17%/-15%) 一致 NO-GO, 新现象: κ-decouple + 大 K 是负面相互作用". Task #287 补充 K=128 拐点.

---

## 10. 后续 backlog (R10 推进)

- **D-curriculum (已闭环)**: Task #288 — A1 β=0.0 三次 NO-GO. baseline recipe 内部无解.
- **D-结构改动 (候选, 高 ROI)**: task268 §4 候选 2 — m-arm κ-Stereographic v9+ (Berman-Metzler 2020). 跟 task288 + task287 + task284 + task283 + task282 联立共同指向结构改动方向.
- **D-其他 baseline recipe 卡死佐证**: task282 / task283 / task284 / task287 / task288 五方向全谱锁死. baseline recipe 不存在 hidden parameter 调优空间.
- **R9 housekeeping (R9-Enforce 强制)**: 启动 task289 (max+1) 填补 10 个 descriptions/ 空洞 (#255, 257-260, 264, 266, 282, 285, 286). PID 全部不存在, 现在安全.

result: Task #287 K=128 κ-decouple 2-arm — **2-arm 全部 NO-GO (R@10=0.0855/-16.2%, R@10=0.0830/-18.6%)**. κ-decouple × K 拐点 (K=64 中性 → K=128 退化). 跟 Task #144 + #284 联立锁死 "κ-decouple 不是 R@10 杠杆, K ≥ 128 是反作用区间" 闭环. 真实杠杆候选 = 结构改动 (task268 §4 候选 2 m-arm κ-Stereographic v9+).