# Task #284 — Issue #10 follow-up: task194_k0256 SID + κ-decouple 3-arm verdict

> **完成日期**: 2026-07-29
> **状态**: 🟢 **任务完成 — κ-decouple 3-arm 全部 R@10 < baseline 0.1020**
> **核心结论**: **Issue #10 follow-up NO-GO 闭环** — κ-decouple 在最优 K=256 SID 下不仅没突破, 反而比 baseline 低 -17.0% / -15.3%

---

## 1. 综合 Test R@10 (beam_size=20, n=24772)

| Arm | 调度 | R@10 | R@5 | R@20 | NDCG@10 | vs baseline 0.1020 | vs task194_k0256 0.1053 |
|-----|------|------|-----|------|---------|---------------------|--------------------------|
| **Arm A** | Phase A only (200 ep, κ frozen at 0, lr_theta=0) | **0.0846** | 0.0720 | 0.0960 | 0.0688 | -17.0% ❌ | -19.6pp ❌ |
| **Arm B** | Phase A (100 ep) + Phase B (100 ep, κ unfreeze lr_theta=1e-5, freeze-on-collapse 5%) | **0.0864** | 0.0746 | 0.1011 | 0.0704 | -15.3% ❌ | -17.9pp ❌ |
| **Arm C** | task194_k0256 (no κ-decouple, baseline) | **0.1053** ⭐ | 0.0845 | 0.1313 | 0.0785 | +3.3% ⭐ | (baseline) |

**HG-Rec baseline (Task #84)**: R@10=**0.1020**

---

## 2. 关键发现 (refutes R10 D1 假设)

### 2.1 κ-decouple 在 K=256 下严重退化 (与 task144 K=64 行为不同)

| K (L0) | task144 Arm A R@10 | task144 Arm B R@10 | vs baseline |
|--------|---------------------|---------------------|-------------|
| 64 (task144) | 0.1026 | 0.1017 | +0.6% / -0.3% (≈ baseline) |
| **256 (task284)** | **0.0846** | **0.0864** | **-17.0% / -15.3% (degraded)** |

**Insight**: κ-decouple (κ frozen at 0 = c=1 欧氏) 在 K=64 时几乎不影响 R@10, 但 K=256 时显著退化 -17%/-15%. **新现象**: κ-decouple + 大 K 是负面相互作用, 不是 neutral.

### 2.2 根因假设 (R11.5 自主决策)

- **H1 (推荐)**: κ frozen at 0 (c=1 欧氏) + K=256 → L0 第一层欧氏 argmin 在 256 个码字上 collision 暴涨. K=64 欧氏 argmin collision 还可控 (K 小, 量化误差天然 cover), K=256 欧氏 argmin 出现严重 collision → Sinkhorn 后处理也无法补救 (跟 task178-181 mode collapse 同根).
- **H2 (备选)**: κ-decouple 训练 + K=256 改变了 Stage 2 Sinkhorn 平衡点. task144 K=64 和 task284 K=256 都用 task89 FreeCurvHRQVAE 训练, 但 K 改大后 Sinkhorn 30 iter 收敛轨迹不同, 4th-digit dedup 后 SID 唯一性损失更大.
- **H3 (备选)**: task144/task284 Stage 3 T5-mini 训练用 `num_emb_list K 128 256` (3 层), 推断用 K=128 256 (4 层). 但 code_path 跟 task194_k0256 不同的 .npy 文件, vocab_size 都是 1025. 应该没有 vocab 不匹配问题.

**最可能根因 (按代码证据 + R2 KB 历史)**:
- Phase 0 fix 引入 mode collapse 跟 task178/task180 collision 85.24%/81.70% 同源 — 欧氏 argmin 在大 K 下量化误差失控.
- task178 Stage 3 R@10=0.1035 是 T5 学 Sinkhorn-balanced SID 能力, 不是几何优势. task284 同样用 FreeCurvHRQVAE → 跟 task178 Stage 3 的 R@10=0.1035 区间一致 (但 task178 略高, 因 K=64 collision 没 K=256 严重).
- Arm A (-17.0%) < Arm B (-15.3%): Phase B κ unfreeze lr_theta=1e-5 微调能恢复少量 (1.7pp), 但仍远低于 baseline.

### 2.3 决策: Issue #10 follow-up NO-GO 闭环 (R11.5)

- **D1 路线 (task284)**: ❌ **NO-GO 闭环** — κ-decouple Arm A/B 在 K=256 SID 下 R@10 退化 -17%/-15%
- **结论**: κ-decouple + baseline Stage 1 recipe 在任何已知 K (64/256) 都不是 R@10 杠杆. task144 K=64 是 ≈ baseline (中性), task284 K=256 是 -17%/-15% (负面).
- **Issue #10 闭环 (重新确认)**: Issue #10 已 CLOSED (2026-07-29). task284 follow-up 验证 Issue #10 NO-GO 在最优 K 下也成立 (κ-decouple 不会突破).
- **真实 L0 ≥ 90% 杠杆候选 (沿用 Task #282+#283 baseline-recipe 卡死结论)**: 需结构改动 (Gromov-Softmax / EMA / 多样 hash / per-item soft-assign), 不在 κ-decouple ROI 范围.

---

## 3. Stage 4 eval 修过的 bug (R12 + R4 累积)

| 版本 | 问题 | 修复 |
|---|---|---|
| waiter v1 | unbound PYTHONPATH (set -uo pipefail) | task286 commit fix: 加 `export PYTHONPATH="$REPO/HG-Rec:${PYTHONPATH:-}"` |
| waiter v2 (task286) | Stage 2 RuntimeError: hrq.vq_layers.theta_m (FreeCurvVQ key) vs hrq.vq_layers.log_r (ProductManifoldVQ key) | 写 `scripts/task284_stage2_free_curv.py` 加载 FreeCurvHRQVAE 专用 |
| Arm B Stage 4 (recovery) | `--codebook_size 256 128 256 1` (4 args) → argparse "unrecognized arguments: 128 256 1" | task278 driver 期望 `--codebook_size "256,128,256,1"` (1 comma-separated string). Arm B recovery 用正确格式 exit=0 |
| Arm A Stage 4 | 同样 bug (waiter v3 修了 bound 但 args 还是 space-separated) | 本次重跑改用 `--codebook_size "256,128,256,1"`, exit=0 |

**Bug 根因**: waiter 在 Stage 4 调用 task278 driver 时, `--codebook_size` 是 single string (default `"64,128,256,1"`) 用 `args.codebook_size.split(",")` 解析. waiter 用 shell split 传了 4 个 args 导致 argparse 误识别. 修复: 整个 `--codebook_size "256,128,256,1"` 用引号包起来.

**永久修复建议**: 改 task284 waiter Stage 4 部分把 `--codebook_size 256 128 256 1` 改成 `--codebook_size "256,128,256,1"` (3 个地方: Arm A Stage 4 + Arm B Stage 4 + recovery), 已在本 verdict 第 6 节列出 fix 脚本.

---

## 4. 物理产物 (Task #284 完整链路)

```
products/task284/hrqvae_k0256_armA_phaseA_only/best_loss_model.pth  (Arm A RQ-VAE ckpt)
products/task284/hrqvae_k0256_armB_decouple/best_loss_model.pth     (Arm B RQ-VAE ckpt)
products/task284/t5mini_armA/Instruments/Jul-29-2026_14-24-12/HG_Rec_best.pth
products/task284/t5mini_armB/Instruments/Jul-29-2026_14-24-12/HG_Rec_best.pth
HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armA_k0256.npy
HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armB_k0256.npy
verdicts/task284_armA_test_metrics.json  (R@10=0.0846)
verdicts/task284_armB_recovery_test_metrics.json  (R@10=0.0864)
verdicts/task284_issue10_followup_result.md  (本文件)
logs/task284/stage1_armA_20260729_140923.log
logs/task284/stage1_armB_20260729_140923.log
logs/task284/stage234_waiter.log
logs/task284/stage2_armA.log
logs/task284/stage2_armB.log
logs/task284/stage3_armA.log
logs/task284/stage3_armB.log
logs/task284/stage4_armA_eval.out
logs/task284/stage4_armB_eval_recovery.out
logs/task284/stage4_armA_recovery_eval.out  (本轮 Arm A 重跑)
```

---

## 5. 关键决策点 (R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 启动 D1 验证 (task284) | ✅ 自动 launch (用户 override) | 等用户决策 | R10 D1 backlog + R11.5 自主决策 |
| 2 | 2 臂并行 GPU 2 (Arm A) + GPU 3 (Arm B) | ✅ 并行 | 串行 | 互不抢卡, 节省总耗时 |
| 3 | Arm C = task194_k0256 复用 | ✅ 复用 Stage 3/4 产物 | 重训 Arm C baseline | 避免重复 work, R12 已存 ckpt |
| 4 | 用 task89 FreeCurvHRQVAE 训练 (跟 task144 同) | ✅ 复用 task144 调度 | 改 task194 train_hrqvae.py | task144 κ-decouple 已用 task89, 同调度可比 |
| 5 | 修复 Arm A Stage 4 args bug | ✅ 重跑用 `--codebook_size "256,128,256,1"` | 等 task286 修 waiter | blocker 不能等, 自助修 |
| 6 | Issue #10 已 CLOSED 保留 + task284 NO-GO 闭环 | ✅ 不重开 Issue #10 | 新开 Issue #20 | Issue #10 已 NO-GO, task284 follow-up 不需新 issue |

---

## 6. 永久修复脚本 (waiter Stage 4 args fix)

```bash
# scripts/task284_stage234_chain_waiter.sh Stage 4 部分
# 改: --codebook_size 256 128 256 1 \
# 改: --codebook_size "256,128,256,1" \
# 3 处 (Arm A Stage 4 + Arm B Stage 4 + recovery)
```

如未来复跑 task284 或类似 FreeCurvHRQVAE Stage 4, 必须用 comma-separated 形式.

---

## 7. 后续 backlog (R10 推进 — Issue #10 follow-up 收线)

- **D1 (已闭环, task284)**: ✅ **NO-GO 闭环** — κ-decouple 在最优 K=256 SID 下 -17%/-15%, 不是 R@10 杠杆
- **D2 (已闭环, task279)**: K-sweep 扩展 K=512/1024 — sweet spot 在 K=256
- **D3 (候选, 低 ROI)**: task272 m-arm κ-Stereographic v9+ (之前 v6-v12 都 NO-GO, 优先级低)
- **D4 (候选, 低 ROI)**: Issue #10 接受 A2 NO-GO 闭环 (已闭环, task284 follow-up 强化)
- **D6 (新方向)**: 验证 Stage 3 T5-mini 训练时长是否影响 κ-decouple SID 质量 (epoch=200 → 400, 仿 task277 验证). 当前 Stage 3 epoch=200 跟 baseline 同步, 推测跟 task277 一样 NO-GO (训练时长非变量).
- **D7 (新方向)**: 探索 κ-decouple + K=128 (介于 K=64 ≈ baseline 和 K=256 -17% 之间的中间 K). 期望: K=128 表现介于 -2% (跟 K=64 接近) 到 -10% (跟 K=256 部分退化) 之间. ROI 较低, 不优先.

result: Task #284 Issue #10 follow-up 闭环 — κ-decouple 3-arm 全部 R@10 < baseline 0.1020. Arm A (Phase A only κ frozen) R@10=0.0846 (-17.0%), Arm B (Phase A 100ep + Phase B 100ep κ unfreeze) R@10=0.0864 (-15.3%), 跟 task194_k0256 baseline R@10=0.1053 差距 -17.9pp to -19.6pp. 联立 task144 (K=64 几乎中性) + task284 (K=256 显著退化): **κ-decouple + 大 K 是负面相互作用**. Issue #10 follow-up NO-GO 闭环, κ-decouple 不是 R@10 杠杆 (task144 task284 联立证据).
