# Task #200 Stage 3 dual_v5 — verdict 模板 (待训练完成后填充)

> **状态**: 🟡 训练中 (PID 281151 alive, GPU 0 96%, epoch 3+)
> **完成日期**: 2026-07-26 (待估)

---

## 0. 实验目的

用户清单 4 选项 B (R11.3 推荐): 验证 v5 collision 84% SID 在 Stage 3 性能"持平".
用户原话兜底: "性能持平即通过".

baseline 对照:
- HG-Rec Task #84: test R@10=0.1020
- Task #181 Phase 0.6: test R@10=0.1057
- Task #188 val K0=64 mean: val R@10=0.1241

期望: val R@10 ≈ 0.10 (持平或略掉).

---

## 1. Stage 2 推断产物 (v5 SID)

| 项 | 值 |
|----|----|
| SID file | `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_dual_v5.npy` |
| Shape | (9922, 4) |
| collision_rate | 0.8387 (vs HG-Rec 99.9%, vs #181 0.94%) |
| 4-digit dedup | OK |
| Stage 2 ckpt | products/task200/dual_arm_C_v5/Jul-26-2026_01-53-.../best_collision_model.pth |

---

## 2. Stage 3 训练配置

| 项 | 值 |
|----|----|
| 模型 | T5-mini 9.18M (d_model=128, 6 enc + 4 dec, d_ff=1024) |
| Epochs | 200 |
| early_stop | 20 |
| batch_size | 256 |
| lr | 1e-4 |
| seed | 42 |
| beam_size (eval) | 20 |
| GPU | 0 (96%, 6.1GB) |
| PID | 281151 |
| 启动时间 | 2026-07-26 02:04:38 AEST |

---

## 3. Stage 3 结果 (待训练完成填充)

| 指标 | dual_v5 期望 | #84 baseline | #181 (Phase 0.6) | dual_v5 实际 | 决策 |
|------|--------------|--------------|-----------------|-------------|------|
| val R@5 | — | 0.0816 | — | _ | — |
| **val R@10** | ≈ 0.10 | **0.1020** | 0.1057 | _ | **持平/略掉/掉>** |
| val R@20 | — | 0.1279 | — | _ | — |
| val NDCG@10 | — | 0.0755 | — | _ | — |
| Best epoch | — | — | — | _ | — |
| 早停 epoch | — | — | — | _ | — |

---

## 4. 决策 (待填充)

| R@10 区间 | 结论 |
|-----------|------|
| ≥ 0.1057 (持平/超 #181) | ✅ 通过 — v5 collision 84% SID 不影响下游性能 |
| 0.0900 - 0.1057 | 🟡 略掉但可接受 (用户原话兜底: "性能持平即通过") |
| < 0.0900 | ❌ NO-GO — collision 84% 显著掉性能, 需重做 (回用户清单 4 选项 A/C/D) |

---

## 5. R12 ckpt + Stage 4 eval

- Stage 3 R12 best_collision_ckpt: products/task200/t5mini_dual_v5/*/best_collision_model.pth
- Stage 3 R12 best_loss_ckpt: products/task200/t5mini_dual_v5/*/best_loss_model.pth
- Stage 4 eval launcher: scripts/task200_stage4_dual_v5_eval.sh (待训练完成启动)

---

## 6. 后续 (待定)

- ✅ Stage 3 dual_v5 完成 → 跑 Stage 4 eval → 写 final verdict
- ⏳ #201 Stage 3 逐层可学习 κ: 等 #200 决策 (用户清单 4 选项决定方向)
- ⏳ #196/#197/#198: 等 #200 决策

---

## 7. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 02:04 | 修 launcher cd + PYTHONPATH → $REPO/HG-Rec | 之前 cd $REPO → ModuleNotFoundError: data.dataset |
| 2026-07-26 02:04 | 修 launcher SID file check (去掉 "Instruments" 前缀避免双前缀) | 上游拼接模式: dataset_path + dataset_name + "/" + dataset_name + code_path |
| 2026-07-26 02:04 | 启动 Stage 3 dual_v5 训练 | 用户清单 4 选项 B (R11.3 推荐 + R11.5 自主) |