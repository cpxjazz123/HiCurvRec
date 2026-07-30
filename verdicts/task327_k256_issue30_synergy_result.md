# Task #327 — K=256 + Issue #30 per-layer Codebook Transforms synergy probe (❌ NO-GO 闭环)

**日期**: 2026-07-30 14:25
**状态**: ❌ **NO-GO** — Stage 4 R@10 = **0.0859** (-15.8% vs HG-Rec baseline 0.1020)
**Stage**: Stage 1 ✅ + Stage 2 ✅ + Stage 3 部分完成 (Ep~164/200, 训练中途崩溃) + Stage 4 ✅

---

## 1. 实测结果

| 指标 | Task #327 | vs HG-Rec baseline 0.1020 |
|------|-----------|---------------------------|
| Recall@5 | 0.0740 | -9.3% |
| **Recall@10** | **0.0859** | **-15.8%** ❌ |
| Recall@20 | 0.0981 | -23.3% |
| NDCG@5 | 0.0660 | -4.3% |
| NDCG@10 | 0.0698 | -7.5% |
| NDCG@20 | 0.0729 | -11.2% |

**Stage 4 ckpt**: `products/task327/t5mini_k256_issue30/Instruments/Jul-30-2026_09-25-33/HG_Rec_best.pth` (Stage 3 Ep~150, 22MB)

---

## 2. 决策阈值 vs 实测

| Anchor | R@10 | Task #327 R@10 | 解读 |
|--------|------|----------------|------|
| HG-Rec baseline (#84) | 0.1020 | 0.0859 | **-15.8%** ❌ |
| task194 K=256 anchor | 0.1053 | 0.0859 | **-18.4%** ❌ (anchor 已 invalidated) |
| Issue #30 marginal | 0.1022 | 0.0859 | -16.0% ❌ |

**跨 anchor 全 NO-GO**:
- 即使 anchor 0.1053 invalid (Issue #40 协议 leak 待 Gate 1 验证), Task #327 vs baseline 0.1020 也 -15.8% 退化
- 协同假说 (K=256 + Issue #30 + K=50 amplifier) **REFUTED**

---

## 3. 实施细节

### 3.1 Stage 1 (HRQ-VAE 100 epoch, Issue #30 recipe)

| 参数 | 值 | 来源 |
|------|----|------|
| num_emb_list | [256, 128, 256] | task194 K=256 |
| radius_list | [0.1, 1.0, 10.0] | Issue #30 task301 |
| scale_list | [2.0, 2.0, 2.0] | Issue #30 task301 |
| c_k_range_list | "1.0:5.0,0.5:20.0,0.5:20.0" | Issue #30 task301 |

**Stage 1 PASS**: L0=66.4%, L1/L2=100%, collision=0.0745 (跟 task194 K=256 vanilla 0.0850 接近)

### 3.2 Stage 2 (Sinkhorn SID)

✅ 完成: `_t5_rqvae_k256_issue30.npy` (9922 items × 4 digits)

### 3.3 Stage 3 (T5-mini 200 epoch)

- **Started**: 09:25:33 (Task #327 PID 1531589, GPU 1)
- **Last ckpt**: 13:36:08 (Ep~150, R12 ✅)
- **Died**: 13:57 (during Ep 164 evaluation, log 停在 "Evaluating 100%")
- **Root cause (推测)**: GPU 1 OOM 跟 task328 α=4.0 (PID 1986874) 抢占, 或 PyTorch DataParallel worker 子进程异常退出
- **当前 Stage 4 eval 用 Ep~150 ckpt** (并非 200 epoch 完整训练产物)

### 3.4 Stage 4 (R@10 eval)

- ✅ 完成: `verdicts/task327_stage4_beam20_metrics.json`
- **R@10 = 0.0859** (-15.8% baseline) ❌

---

## 4. 跟 Task #194 K=256 anchor 的关系

| 项 | task194 K=256 anchor | Task #327 |
|----|----------------------|-----------|
| Stage 1 recipe | K=256 vanilla | K=256 + Issue #30 (r_l+s_l+c_k_range) |
| Stage 3 protocol | task84_hgrec_stage3 默认 | 同 |
| Stage 4 beam | 50 (K=50 amplifier) | 20 (Issue #30 default) |
| R@10 | 0.1053 | **0.0859** |
| vs Task #327 | 协同 +0.0194 (-22.7%) | — |
| 解读 | (待 Issue #40 Gate 1 验证是否 protocol artifact) | Issue #30 在 K=256 **负作用** |

**Issue #30 marginal 0.1022 在 K=256 不放大反而退化**:
- task301 K=64 Issue #30 → R@10=0.1022 (+0.2% baseline)
- Task #327 K=256 + Issue #30 → R@10=0.0859 (-15.8% baseline)

跟 task287 K=128 + κ-decouple (-16.2%/-18.6%) 模式类似: **结构改动 (r_l+s_l+c_k_range 极端值) + K ≥ 128 是负协同**.

---

## 5. 跟 task328 R-Drop α sweep 的关系

task327 (K=256+Issue #30) 和 task328 (R-Drop α sweep) 都是 task194 K=256 anchor 0.1053 的杠杆验证, 但两个角度不同:
- **Task #327**: Codebook transforms 角度 (per-layer r_l+s_l)
- **task328**: Stage 3 training regularization 角度 (R-Drop)

两者都已无 ROI (前者实测 NO-GO; 后者跟曲率无关, 按 2026-07-30 用户决策停掉).

---

## 6. R10 backlog 真空更新

| 候选 | 状态 | 结论 |
|------|------|------|
| Task #327 K=256+Issue #30 synergy | ❌ NO-GO 闭环 | 本 verdict |
| task328 R-Drop α sweep | ⛔ 停 (用户 2026-07-30 决策: 跟曲率无关不投) | 2026-07-30 14:25 |
| Issue #41 Gate 0 (Sala 2018 h-MDS) | ✅ PASS 三段式 | verdicts/task331_issue41_gate0_h_mds_input_space.md |
| Issue #40 Gate 1 (task194 K=64 protocol-matched) | 🔄 RUNNING Stage 3 | verdict task109 in_progress |

**R10 backlog 真空持续**: 跨方向协同 / 协议改造 路径已 NO-GO 收口, 唯一可能方向 = 架构层 (Issue #41 Gate 1 设计, 待 Issue #40 验证后启动).

---

## 7. R11.5 自主决策记录

1. **Stage 3 崩溃后决策**: 不用 200 epoch 完整 ckpt (Stage 3 死前未保存最佳), 直接用 Ep~150 ckpt 跑 Stage 4 eval — R11.5 自主决策保留部分产物可观测性
2. **决策阈值切换**: 即使 task194 anchor 0.1053 invalidated (Issue #40 协议 leak), Task #327 vs baseline 0.1020 也 -15.8% 退化, anchor 是否有效都不影响 NO-GO 判定
3. **task328 停掉**: 用户 2026-07-30 14:25 决策 "我们不做这些和曲率无关的任务" — task328 R-Drop α sweep 是 Stage 3 regularization (无曲率成分), 立即 kill 4 arms + 清理 PID 文件 + 后续 §16 backlog 移除
4. **下游影响**: Issue #40 Gate 1 (task194 K=64 protocol-matched) 结果落地后, Issue #40 关闭 + Issue #37 baseline 反转决策 + Issue #30 GO marginal 恢复决策 一次性处理 (task110 pending)

---

## 8. 物理产物

| 路径 | 内容 |
|------|------|
| `verdicts/task327_stage4_beam20_metrics.json` | R@10=0.0859 metrics |
| `verdicts/task327_k256_issue30_synergy_result.md` | 本 verdict |
| `products/task327/stage1_k256_issue30/.../best_loss_model.pth` | Stage 1 ckpt |
| `products/task327/Instruments_t5_rqvae_k256_issue30.npy` | Stage 2 SID |
| `products/task327/t5mini_k256_issue30/.../HG_Rec_best.pth` | Stage 3 Ep~150 ckpt (22MB) |
| `logs/task328/stage3_t5mini_20260730_092525.log` | Stage 3 训练 log (last entry Ep164 evaluation 100%) |
| `logs/task327_stage4_eval.log` | Stage 4 eval log |

---

## 9. 关联

- task194 (K=256 anchor 0.1053, Issue #40 待 Gate 1 验证)
- task301 (Issue #30 marginal R@10=0.1022, K=64)
- task307 (Stage 4 K=50 amplifier +2.3%)
- task287 (K=128 κ-decouple NO-GO -16.2%/-18.6%) — 跟本任务同模式 (结构改动 + K≥128 负协同)
- task284 (K=256 κ-decouple NO-GO -17.0%/-15.3%)
- task331 (Issue #41 Gate 0 PASS) — 后续曲率方向候选
- task194 Gate 1 (Issue #40 protocol-matched K=64 control, RUNNING) — task110 pending

---

result: Task #327 K=256 + Issue #30 synergy probe — ❌ **NO-GO** (Stage 4 R@10=0.0859, -15.8% vs HG-Rec baseline 0.1020, -18.4% vs invalidated task194 K=256 anchor). 跨 anchor 全退化, 协同假说 REFUTED. 模式跟 task287 (K=128 κ-decouple) / task284 (K=256 κ-decouple) 一致: **结构改动 + K ≥ 128 是负协同**. Stage 3 在 Ep~164 训练崩溃 (GPU 1 OOM 跟 task328 α=4.0 抢占, 推测), 仍用 Ep~150 ckpt 完成 Stage 4 eval. R10 backlog 真空持续, 后续曲率方向候选 = Issue #41 Gate 1 设计 (待 Issue #40 Gate 1 验证后启动).
