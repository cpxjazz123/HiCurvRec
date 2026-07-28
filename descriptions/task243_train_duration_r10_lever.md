# Task #243 — 训练时长作为 R@10 真正变量 (Task #237 §后续建议)

## 来源
- Task #237 §后续建议 (2026-07-29): "调查 Arm A → Arm C R@10 增益 (+3.7pp) 来自 Sinkhorn 之外的什么变量. 这才是 Issue #10 真正想解的因果问题. 候选: (1) Stage 3 训练时长 (Arm C 可能跑了更久); (2) 码字分布熵 (Sinkhorn 让码字分布更均匀); (3) lr 协议差异. 跟 task200 dual_v5 (Stage 3 截断导致 -10.3%) 对照, 可能训练时长就是真正变量."
- Issue #9/#10/#11/#12 全部 NO-GO 关闭后, R10 必须主动推进. Task #243 是当前 backlog 最高 ROI 候选.

## 背景

| Arm | Sinkhorn | Stage 3 训练时长 | collision | R@10 | 来源 |
|-----|----------|------------------|-----------|------|------|
| task84 baseline (200 ep) | max_iters=0 | 200 ep + early_stop | 0.99 | **0.1020** | task84 |
| task237 Arm B (200 ep) | max_iters=10 | 200 ep + early_stop | 0.1005 | 0.1021 | task237 |
| task237 Arm C (200 ep) | max_iters=30 | 200 ep + early_stop | 0.05 | 0.1058 | task237 |
| task200 dual_v5 (200 ep 截断) | max_iters=0 | **93/200 截断 (silent death)** | 0.83-0.93 | **0.0915** | task200 |
| task233 dual_v5 RERUN (200 ep) | max_iters=0 | 68/200 (clean finish) | 0.83-0.93 | 0.0934 | task233 |

观察:
- task200 vs task233: 同一 Stage 2 SID (dual_v5), Stage 3 budget 93 ep (截断) vs 68 ep (clean) 差 -1.9pp; 但都是 "短跑" 离 200 ep 还远
- task237 Arm B (200 ep, Sinkhorn=10) R@10=0.1021 vs task237 Arm C (200 ep, Sinkhorn=30) R@10=0.1058 — Arm C 多 3.7pp, 但 Sinkhorn 之外是否有训练时长混淆?
- task84 baseline (200 ep + early_stop, hit clean) R@10=0.1020 是干净 baseline

**关键假设**: 如果 Stage 3 训练时长是真正变量, 那么延长 Stage 3 (e.g. 400 ep / 800 ep) 应该单调提升 R@10.

## 实验设计

**自变量**: Stage 3 num_epochs ∈ {200, 400}, 其它完全相同.
- **Arm A** (200 ep, 已有 baseline 不重跑): task84 R@10=0.1020, task237 Arm A 跟它等同
- **Arm B** (400 ep): 跟 Arm A 同一 SID (baseline Instruments_t5_rqvae_code_default.npy), 但 num_epochs=400
- **Arm C** (备选 800 ep): 仅在 Arm B > Arm A + 1pp 时跑

**受控因素**: 
- 同一 Stage 2 SID: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy` (task84 baseline, 无 Sinkhorn)
- Stage 3 协议同 task233: T5-mini 9.18M, batch_size=256, lr=1e-4, seed=42, early_stop=20 (但 epochs 上限翻倍)
- Stage 4 eval 同 task237: task233 协议 + slice (Head/Body/Tail)
- 单 seed (用户 2026-07-23 撤回 multi-seed, 单 seed 足够)

**资源**:
- 2 GPU 并行 (GPU 1/2 — GPU 0 留给其他任务)
- 200 ep × ~13 s/epoch ≈ 44 min
- 400 ep × ~13 s/epoch ≈ 88 min
- 并行 wall ≈ 88 min
- Stage 4 eval × 2 ≈ 6 min

### 决策阈值 (R11.3 自主推荐)

| Gate | 指标 | 阈值 | Pass 含义 |
|------|------|------|-----------|
| Gate 1 (核心) | 400 ep R@10 - 200 ep R@10 | ≥ +1pp | 训练时长是 R@10 杠杆 |
| Gate 1 (核心) | 400 ep R@10 | ≥ 0.1020 (baseline) | 400 ep 不退化 |
| Gate 2 (备选) | 800 ep R@10 - 400 ep R@10 | ≥ +1pp | 长训持续提升 |
| Gate 2 (备选) | 800 ep R@10 | ≥ 0.1058 (Arm C 当前最佳) | 值得采纳 |

**Hard STOP** (任一即停):
- 400 ep R@10 退化 (< 0.1010) → 训练时长不是 R@10 杠杆, 方向关闭
- 400 ep R@10 ≈ 200 ep R@10 (Δ < 0.5pp) → 训练时长不是 R@10 杠杆
- 400 ep 训练中 cos_mean 崩 / silent death → STOP, 标记协议问题

## 步骤

1. ✅ 写 description (本任务)
2. ⏳ 写 launcher: 2 个 Stage 3 训练并行 (epoch=200 / 400)
3. ⏳ 启动并行 Stage 3 (200 ep GPU 1, 400 ep GPU 2)
4. ⏳ 等待训练完成 (~88 min)
5. ⏳ Stage 4 eval × 2 (各 ~3 min)
6. ⏳ 写 verdict, 三臂表对照 (200/400 ep)
7. ⏳ commit + Issue #10 follow-up comment

## 产物

- descriptions/task243_train_duration_r10_lever.md (本文件)
- scripts/task243_stage3_epoch200.sh + scripts/task243_stage3_epoch400.sh (launcher)
- products/task243/t5mini_epoch200/.../HG_Rec_best.pth
- products/task243/t5mini_epoch400/.../HG_Rec_best.pth
- verdicts/task243_train_duration_result.md

## 依赖
- HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy (task84 baseline Stage 2 SID, 不重训)
- scripts/task84_hgrec_stage3_train.py (Stage 3 R12 + heartbeat)
- scripts/task237_stage4_armB_eval.sh (Stage 4 eval template)

## Status
R10 主动推进, 启动本任务不依赖用户决策. GPU 1/2 空闲 (nvidia-smi 已验证).
