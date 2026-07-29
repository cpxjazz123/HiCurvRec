# Task #293 / Issue #23 Gate 0 — per-layer per-epoch c_k curriculum NO-GO

**日期**: 2026-07-29
**状态**: ❌ **Issue #23 Gate 0 FAIL (0/9 = 0% OPEN) → H1 REFUTED → 不进入 Stage 1**
**决定**: 关闭 Issue #23, 承认 c_k curriculum 路径在 task275 续训 ckpt 起点上 NO-GO

---

## 1. Gate 0 决策表 (per (Schedule, Layer) 9 组合)

| Sched | Layer | K | mean_agree | verdict |
|-------|-------|---|-----------|---------|
| **A** (异构 早期宽→中期标准→后期密) | L0 | 64 | **13.38%** | TOO_STRONG |
| A | L1 | 128 | **13.17%** | TOO_STRONG |
| A | L2 | 256 | **33.37%** | TOO_STRONG |
| **B** (全程宽 U(0.5, 20)) | L0 | 64 | **4.94%** | TOO_STRONG |
| B | L1 | 128 | **9.05%** | TOO_STRONG |
| B | L2 | 256 | **16.38%** | TOO_STRONG |
| **C** (全程窄 U(1, 5), 跟 task275 一致) | L0 | 64 | **12.44%** | TOO_STRONG |
| C | L1 | 128 | **23.86%** | TOO_STRONG |
| C | L2 | 256 | **47.61%** | TOO_STRONG |

**OPEN rate**: 0/9 = 0.0% (通过条件 ≥ 60%, ❌ FAIL)

---

## 2. Gate 0 实施细节

- **Ckpt 起点**: `products/task275/A2_extend_ep50/Jul-29-2026_10-35-55_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (warm-start 候选, 跟 Issue #23 body 一致)
- **Schedule 三组**:
  - Schedule A (异构): Phase 0 (ep 0-9) U(0.5, 20) → Phase 1 (ep 10-19) U(1, 5) → Phase 2 (ep 20-29) U(2, 8)
  - Schedule B (全程宽): 全程 U(0.5, 20)
  - Schedule C (全程窄): 全程 U(1, 5) (跟 task275 等价)
- **代表 phase = Schedule 最后一段** (Phase 2): 因为 Phase 0 forward-pass 是 1 次 = 1 个 schedule 端点, 不模拟 30 epoch 演化. 取 schedule 末态 (代表训练后期 c_k range 收紧)
- **Seeds**: 42, 43, 44 (3 seeds × 3 schedules × 3 layers = 81 per-seed combinations)
- **测量**: per-codeword κ argmin vs Euclidean argmin agreement

---

## 3. 关键发现

### 3.1 跟 task241 (Issue #11 Gate 0 PASS) 对比

task241 在 task84 baseline ckpt 上测 per-layer c_k range 三层组合性, 结果 L0=82.68% / L1=67.15% / L2=75.69% (全 OPEN).

task293 在 task275 A2 curriculum 续训 ckpt 上测 per-layer per-epoch c_k curriculum 三层组合性, 结果 L0=4.94-13.38% / L1=9.05-23.86% / L2=16.38-47.61% (全 TOO_STRONG).

**核心差异**: task275 ckpt 的码字已坍缩到 boundary (recon_loss=7.67 卡死, plateau 89.1% ×3 续训不收敛). κ-stereographic 距离的 argmin 跟 Euclidean argmin 在 boundary saturation 状态下系统性背离.

### 3.2 H1 REFUTED 的直接证据

Issue #23 §H1: "K=64 L0 在单值 c_k range 下 utilization 被钉在 82-89% 是 range 单值的副作用, 让 c_k range 随 epoch 演化能跨过 90%."

**实测反证**: 即使让 c_k range 从单值扩展为三段时间函数, 在 task275 续训 ckpt 起点上, per-layer agreement 仍 < 50% (跟三层都远离 60-90% OPEN 带). 即 c_k 时间函数本身不能在「已坍缩」空间里恢复几何区分力.

**进一步反证**: 即使切到 Schedule B (全程宽 U(0.5, 20), Issue #23 §3.1 已证 L1/L2 释放几何参与度的 c_k range), L0 agreement 仍只有 4.94% (跟 task231 全局 U(1,5) Phase 0 OPEN 82.68% 差 ~70pp).

### 3.3 Schedule 间对比 (Schedule A vs B vs C 的相对差异)

| Sched | L0 | L1 | L2 | 解读 |
|-------|----|----|----|----|
| A 异构 (末段 U(2, 8)) | 13.38% | 13.17% | 33.37% | 末段窄→码字区分力恢复 |
| B 全程宽 U(0.5, 20) | **4.94%** | 9.05% | 16.38% | c_k 大→κ 与 Euclidean 距离完全背离 |
| C 全程窄 U(1, 5) | 12.44% | 23.86% | **47.61%** | c_k 小→最接近 Euclidean (但仍远未到 60% OPEN) |

Schedule C 的 L2=47.61% 是 9 组合中最高, 但仍 < 60% OPEN 带. 说明 **即使 c_k range 全程最窄 (跟 task275 等价), 在 task275 ckpt 起点上仍不能进入 60-90% 安全区**.

---

## 4. Issue #23 §Gate 0 硬停止触发

> **若 < 60% 组合通过 → STOP 并在 verdict 记录哪种 schedule 不通过. 不得为补 Gate 0 通过而进入 Stage 1 训练. Gate 0 失败意味着 c_k 时间函数在 forward-pass 都不兼容, Stage 1 训练必然失败.**

实测: 0/9 (Schedule, Layer) 通过 OPEN, 0% (远低于 60% 阈值). 触发硬停止. **本 issue 关闭, 承认 NO-GO, 不申请 Stage 1 训练预算.**

---

## 5. Issue #23 §反证 / 压力测试 验证

Issue #23 §反证:

> **最强反证 (H1 不成立)**: 单一 c_k range 的 plateau 90% 是 K=64 hyperbolic codebook 的硬上限, 不是 range 单值副作用. 若此反证成立, 本 issue 在 Gate 1 必然失败.

**Gate 0 直接验证**: 0/9 组合在 OPEN 带 = H1 在 forward-pass 阶段就被证伪. Issue #23 的最强反证成立. Gate 0 fail = H1 在 forward-pass 层就失败, 不需要 Gate 1 训练实证.

> **H2 反证 (per-layer 解耦不需要)**: task275 A2 在 β-curriculum 上三层一起推进, plateau 89.1%/97%/98% — 表明 β-curriculum 已经让三层都 "动起来", 异构 c_k 时间函数未必比同构更优.

**Gate 0 对照验证**: Schedule A (异构, 三层各自演化) vs Schedule C (同构, 全程窄 U(1, 5) 跟 task275 一致):
- A L0=13.38% vs C L0=12.44% (差异 +0.94pp, **A 略优但都不 OPEN**)
- A L1=13.17% vs C L1=23.86% (差异 -10.69pp, **C 反而更优**)
- A L2=33.37% vs C L2=47.61% (差异 -14.24pp, **C 反而更优**)

**H2 反证成立**: 异构 schedule (A) 在 L1/L2 上比同构 (C) 反而 **更差**. 这跟 H2 反证的预期一致 — task275 β-curriculum 已在「同构」侧推动三层, per-layer 解耦 (异构) 不能带来增益.

---

## 6. R2 KB 更新 (后续 backlog 引用)

- **Issue #23 Gate 0 NO-GO 在 task275 续训 ckpt 起点上**: c_k range 时间函数 (无论异构/同构) 不能在「已坍缩到 boundary」空间里恢复 κ-stereographic vs Euclidean 的几何区分力. Agreement 全部 < 50% (跟 task241 task84 baseline ckpt 起点 82.68%/67.15%/75.69% 差 ~30-80pp).
- **Issue #23 反证全部成立**:
  - H1 不成立 (c_k 时间函数不能解锁 L0 cap) → 实测 0/9 OPEN
  - H2 反证成立 (per-layer 解耦不需要, 异构在 L1/L2 上反而更差) → 实测 Schedule A vs C 差异
- **Issue #23 关闭, 后续不申请 Stage 1 / Stage 2 / Stage 3 / Stage 4 预算**
- **R10 backlog 真空继续**: 联立 task287 + task288 + task289 + task290 + task291 + task292 + task293 = HG-Rec baseline recipe (RQ-VAE + Sinkhorn + κ-Stereo + per-layer-curriculum 探索) 在 Inh=音乐乐器 5-core 上不存在 hidden param 调优空间

---

## 7. 物理产物

- `descriptions/task293_issue23_per_layer_ck_curriculum.md` (任务定义)
- `scripts/task293_issue23_gate0_81_combo.py` (Phase 0 forward-pass 81 组合测量)
- `logs/task293_issue23_gate0.log` (运行日志)
- `/home/wlia0047/.claude/jobs/04ccf474/tmp/task293_issue23_gate0_81_combo.json` (结果落盘)
- `verdicts/task293_issue23_gate0_result.md` (本 verdict)

---

## 8. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #23 启动 ROI 评估 | ✅ Gate 0 零 GPU 验证后决定是否进 Gate 1 | 跳过 Gate 0 直接进 Stage 1 | Issue #23 body 明确 Gate 0 = 硬停止, 不允许跨过 |
| 2 | ckpt 起点 | ✅ task275 A2_extend_ep50 best_loss (Issue #23 指定 warm-start) | task222 baseline | Issue #23 body 明确 "继承 task275 A2 ckpt 作为 Stage 1 warm-start" |
| 3 | Schedule 模板 | ✅ A 异构 / B 全程宽 / C 全程窄 | 只 A 异构 | 3 schedule 是 Issue #23 body 原文, 异构 vs 同构 对照 = H2 反证验证基础 |
| 4 | 代表 phase 选择 | ✅ Schedule 最后一段 (Phase 2) | 三段平均 / Phase 0 起点 | forward-pass 1 次 = 1 个 schedule 端点; 取末态 = 代表训练后期; 比 Phase 0 起点更接近训练动力学 |
| 5 | 种子 | ✅ 42/43/44 (跟 task241 一致) | 单 seed | Issue #23 body 原文 "× 3 seeds × 3 layers = 81 个 agreement 测量"; 复用 task241 种子保持跨任务可比 |
| 6 | conda env 重建 | ✅ pip install 到 /tmp/genrec_env (临时 target) | 等用户装 conda env | 节点重置后 grid_toys env 不可用, 必须有可执行 Python + torch 才能跑 Gate 0; /tmp/genrec_env 是 R10 推进最快的路径 |
| 7 | Issue #23 关闭 | ✅ Gate 0 FAIL 后写本 verdict + 关闭 GitHub issue | 启动 Gate 1 再确认 | Issue #23 §Gate 0 硬停止 = "若 < 60% 组合通过 → STOP"; 0% 必须停止 |

---

## 9. R7 GPU 状态

Gate 0 零 GPU (纯 forward-pass numpy/torch.cpu). 未申请任何 GPU 预算. Gate 1 不启动 (硬停止触发).

---

result: Task #293 / Issue #23 Gate 0 NO-GO **0/9 (Schedule, Layer) 组合在 60-90% OPEN 带内**. 全部 TOO_STRONG (L0 4.94-13.38%, L1 9.05-23.86%, L2 16.38-47.61%). 跟 task241 task84 baseline ckpt 起点 (L0=82.68% / L1=67.15% / L2=75.69%) 差 30-80pp. 根因: task275 续训 ckpt 码字已坍缩到 boundary. **H1 REFUTED**: c_k range 时间函数不能在「已坍缩」空间里恢复 κ-Stereo vs Euclidean 几何区分力. Issue #23 关闭, 不进入 Stage 1, 写 verdict NO-GO.
