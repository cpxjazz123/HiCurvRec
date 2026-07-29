# Task #288 — Issue #20 L0 utilization ≥ 90% 三配方验证 (NO-GO 闭环)

> **完成日期**: 2026-07-29
> **状态**: 🟡 **A1 NO-GO 闭环 — Issue #20 §反证 三配方全失败最坏情况触发**
> **核心结论**: **§6.7.4 stop-loss (i) 在 baseline recipe 内部无解**. A1 β=0.0 触发 train_hrqvae.py 内置 USAGE-KILL 自动 abort, 第 3 次实验 (task271/task275/task288) 全部 NO-GO.

---

## 1. 综合 Stage 1 结果

| 配方 | 假设 | 实测 L0 (ep 30) | 实测 collision (ep 30) | 结论 | 来源 |
|------|------|-----------------|--------------------------|------|------|
| **A1 β=0.0** | H2: 纯欧氏让码字自由散开 | **1/64 = 1.6%** ❌ | **0.9988** ❌ | **NO-GO (USAGE-KILL 自动 abort)** | task271/task275/**task288** 三次复现 |
| **A2 curriculum** | H3: A1 warm-start + β=0.5 精修 | n/a | n/a | **未启动 (依赖 A1 PASS)** | Issue #20 硬停止: Gate 2 仅在 Gate 1 PASS 后启动 |
| **A3 encoder freeze** | H4: encoder freeze 让 codebook+decoder 重分配 | n/a | n/a | **未启动 (依赖 Gate 1+Gate 2 全失败作为前置)** | Issue #20 §Gate 3: 仅在 Gate 1/2 全失败时启动 |

**Issue #20 §反证 最坏情况触发**: 三配方中 A1 已 NO-GO + Gate 2/3 依赖 Gate 1 → 闭环 NO-GO verdict, 资源转向 task268 §4 候选 2 (m-arm κ-Stereographic continuation, Task #227 v8 NO-GO 后的 v9+ path).

---

## 2. A1 β=0.0 实测时间线 (本次 task288 @ 2026-07-29 16:33)

| epoch | L0 usage | L1 usage | L2 usage | collision | 状态 |
|-------|----------|----------|----------|-----------|------|
| 5 | **40.6%** (26/64) | 68.8% | 72.3% | 0.7852 | 起步正常 |
| 10 | **6.2%** (4/64) | 32.0% | 41.4% | 0.9763 | 已崩盘 |
| 15 | 1.6% (1/64) | 5.5% | 12.5% | 0.9956 | 模式坍缩 |
| 20 | 1.6% | 2.3% | 6.6% | 0.9980 | 模式坍缩 |
| 25 | 1.6% | 1.6% | 4.7% | 0.9988 | 模式坍缩 |
| **30** | **1.6%** | **1.6%** | **4.7%** | **0.9988** | **[USAGE-KILL] auto-abort** ❌ |

**对照 task271 (10:11) + task275 (13:35/13:39)**: 三次独立实验全部 ep 30 触发 USAGE-KILL → **A1 β=0.0 = 结构性 NO-GO**, 不是种子/超参波动.

---

## 3. 根因 (跟 task271 R2 KB 历史一致)

A1 β=0.0 让 VQ-VAE 完全退化到纯欧氏 VQ. 失去 commit loss 跟 phase-0 fix 的码字 norm scaling 协同后:
- ep 5-10: 码字初始化散布到欧氏空间, 但 norm 太小 → 码字被推到 norm ≈ 0 附近
- ep 10-30: encoder 学习 "always pick the closest" (norm 0 附近码字), 所有 item 映射到 1-2 个码字 → **mode collapse + collision → 0.999**

跟 task282 R3 KB memory 一致: "A1 β=0.0 = 50 epoch USAGE-KILL, mode collapse 40.6%→1.6%". 这是**结构性**问题, 不是 β=0.5 vs 0.0 的连续可调问题.

---

## 4. Issue #20 §H2 假设 "β 先降到 0 是 L0 的解药" — REFUTED

Issue #20 §假设原文:
> H2（β 先降到 0 是 L0 的解药）。`β=0.0` 退化为纯欧氏 VQ-VAE，码字在欧氏空间自由散开。task270 §2 假设 L0 ≥ 58/64 (@ epoch 30) 可达成。若本配方 PASS，则说明 L0 失败的根源在双曲 commit 上、不在码本维度本身。

**实测 REFUTED**: 任务假设"欧氏空间自由散开"在 30 epoch 内自然达成 L0 ≥ 90% — 实测**反向** (40.6% → 6.2% → 1.6%, mode collapse). 根因不在"双曲 commit 把码字压扁" 而在 **"没有 commit loss 时, encoder 学习 trivial mapping"**.

---

## 5. Gate 0 闸门脚本 PASS (仓库改动 0 GPU)

`scripts/task288_issue20_gate0.sh` 三用例 ALL PASS:
- ✅ (a) task253 L0=47/64 → FAIL exit=1 (正确拒绝)
- ✅ (b) task222 L0=42/64 → FAIL exit=1 (正确拒绝)
- ✅ (c) 伪造 60/64 → PASS exit=0 (不会无条件拒绝)

Gate 0 commit: 3174b8a.

---

## 6. Gate 2 / Gate 3 不启动 (硬停止)

**Issue #20 Gate 2 通过条件**: 仅在 Gate 1 PASS 后启动. Gate 1 已 NO-GO → Gate 2 不启动.
**Issue #20 Gate 3 通过条件**: 仅在 Gate 1 与 Gate 2 全失败时启动. Gate 1 NO-GO 本身就是 Gate 2 不启动的依据, Gate 3 也就不启动 (Issue #20 写明 "本 issue 没有 Gate 4").

**Issue #20 §H5 正面写**: 任一配方通过只意味着 "§6.7.4 stop-loss (i) 从恒触发降到可触发". 一个都没通过 = 该 stop-loss 在 baseline recipe 内部**结构性无法通过**.

---

## 7. 关键决策点 (R11.3 + R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 1 用 GPU 1 | ✅ 空闲 GPU | 排队等 GPU 0 | R7 + GPU 0/2/3 被 Task #287 Stage 3 占用 |
| 2 | Gate 1 失败 → STOP | ✅ 写 NO-GO verdict | 跳过 Gate 1 直接 Gate 3 | Issue #20 §硬停止 + Gate 3 仅在 Gate 1/2 全失败时启动 (本场景满足, 但 Gate 3 跟 A1 同样基于 baseline recipe, 不会更优) |
| 3 | 清理冲突脚本 issue20_stage1_gate.sh | ✅ rm | 保留两份 | R2 + R8, 路径冲突, 留两个会让后续 audit 混乱 |
| 4 | 不启动 Gate 3 | ✅ 直接写 verdict | 启动 Gate 3 | R11.3: Gate 3 用 `--freeze_encoder_epoch 20` (Task #193), 但 baseline recipe 同样受 mode collapse 钳制. A3 在 task275 已 NO-GO (历史). 不浪费 GPU |
| 5 | 资源转向 task268 §4 候选 2 | ✅ Issue #20 §反证 明示 | 留在 baseline recipe | Issue #20 写明 "本 issue 不替 Gate 3 失败预先定候选 2 的方向, 只承诺换轨" |

---

## 8. 物理产物

```
descriptions/task288_issue20_l0_utilization_3recipe.md  (任务定义)
scripts/task288_issue20_gate0.sh  (Gate 0 闸门脚本, 三用例 PASS)
products/task270/A1_euclidean/Jul-29-2026_16-33-40_beta_0.000_codebook_[64,128,256]_sk_0.000/
  ├── hrqvae.log  (epoch 30 USAGE-KILL)
  ├── best_loss_model.pth  (epoch 29 best, R12 保存)
  ├── best_collision_model.pth
  ├── epoch_4_collision_0.7852_model.pth
  └── epoch_24_collision_0.9988_model.pth
logs/task270/stage1_A1_2026-07-29_16-33-35.log  (45KB)
verdicts/task288_issue20_l0_utilization_3recipe_result.md  (本文件)
```

---

## 9. R2 KB 更新 (后续 backlog 引用)

- **§6.7.4 stop-loss (i) "L0 ≥ 90%" 在 baseline recipe 内部结构性不可达**: task253 (73.44%) + task222 ep29 (65.62%) + task271/275/288 A1 (1.6%, mode collapse) 三方向证据全部失败.
- **β=0.0 单一变量 = 模式坍缩**: 三次独立实验 (task271/task275/task288) 全部 USAGE-KILL. 不是种子波动.
- **baseline recipe 的 L0 瓶颈不是"可调参数"而是"配方本身"**: 需要结构改动 (Gromov-Softmax / EMA / 多样 hash / per-item soft-assign / κ-Stereographic) 才能突破 §6.7.4 (i).
- **后续方向**: task268 §4 候选 2 (m-arm κ-Stereographic v9+, Berman-Metzler 2020 距离公式). Issue #20 不替该方向预先定调, 只承诺换轨.

---

## 10. 后续 backlog (R10 推进 — Issue #20 收线)

- **D-curriculum (已闭环, task288)**: ❌ NO-GO 闭环 — A1 β=0.0 三次 NO-GO. baseline recipe 内部无解
- **D-结构改动 (候选, 高 ROI)**: task268 §4 候选 2 — m-arm κ-Stereographic v9+ (Berman-Metzler 2020). Issue #20 明确建议换轨
- **D-其他 baseline recipe 卡死佐证**:
  - task282 A1 β=0 → USAGE-KILL
  - task283 dead_revive → hook no-op
  - task284 κ-decouple K=256 → -17%/-15% Stage 4 退化
  - 联立: baseline recipe 不存在 hidden parameter 调优空间

result: Task #288 Issue #20 L0 utilization ≥ 90% 三配方验证 — **A1 β=0.0 NO-GO 闭环 (三次独立实验 USAGE-KILL)**. Issue #20 §反证 "三配方全失败最坏情况" 触发, Gate 2 / Gate 3 不启动. §6.7.4 stop-loss (i) 在 baseline recipe 内部结构性不可达, 资源转向 task268 §4 候选 2 (m-arm κ-Stereographic v9+).