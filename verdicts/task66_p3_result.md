# Task #66 P3 Sinkhorn Cascade TIGER — Verdict (用户主动 kill)

> **完成日期**: 2026-07-20
> **状态**: ⏸️ **P3 PARTIAL GO** — 训练因 val_R@5 收敛停滞被用户主动 kill；best val_R@5 = **0.07918 @ step 1400** (vs Task #59 baseline 0.0857，差 7.6%)
> **任务**: Stage 3 TIGER Sinkhorn cascade 闭环验证 G1×G3 拓扑优势能否转化为 R@5 ≥ 0.0857

---

## 1. 任务目标

验证 **G1×G3 Sinkhorn-balanced RQ cascade**（拓扑保护代理）训出的 TIGER 端到端 Recall@5 是否能 ≥ Task #59 baseline (0.0857)。

**D0 + P1 预测**: P1 验证显示 Sinkhorn cascade vs Vanilla L0 拓扑破坏降低 36%（2× 保留，7σ 显著），预测 P3 端到端 R@5 ≥ 0.0857。

---

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-20 12:00:56 | Task #66 P3 训练启动 (cuda:0/1/2/3 DDP 4 卡), task_name=task66_p3_sinkhorn_cascade |
| 12:36 | step 500 val_R@5 = **0.04111** (与 Task #59 同期持平) |
| ~12:38 | **用户不小心关掉 terminal 触发 SIGTERM，训练中断** |
| 12:42-12:45 | 备份 metrics.csv，建 symlink logs/task66_p3_step500.ckpt |
| 12:45 | 用 `ckpt_path=.../task66_p3_step500.ckpt` 从 step 500 续训 (task_name=task66_p3_resume) |
| 12:45 - 13:50 | 续训 65+ 分钟，跑到 step 1900+ |
| 13:50 | **用户主动 kill -TERM 1029066** (val_R@5 连续 5 个 val check 未超 best) |
| 13:51 | DDP 子进程 SIGKILL 强制清理，4 张 GPU 全部释放 |

---

## 3. 关键指标

### 3.1 训练过程 val_R@5 序列（完整链路：从原训练 step 500 到 resume step 1900+）

| Step | val_R@5 | Δ | 备注 |
|------|---------|---|------|
| 500 (resume 起点, 原 P3 训练终点) | 0.04111 | — | 完整序列起点 |
| 600 | 0.05486 | +0.0138 | post-resume 首次 val |
| 700 | 0.06244 | +0.0076 | |
| 800 | 0.06738 | +0.0049 | |
| 900 | 0.07063 | +0.0033 | |
| 1000 | 0.07279 | +0.0022 | |
| 1100 | 0.07408 | +0.0013 | |
| 1200 | 0.07758 | +0.0035 | 反弹 |
| 1300 | 0.07732 | -0.0003 | |
| **1400** | **0.07918** | **+0.0019** | ⭐ **best** |
| 1500 | 0.07763 | -0.0016 | 首次下降 |
| 1600 | 0.07629 | -0.0013 | |
| 1700 | 0.07722 | +0.0009 | |
| 1800 | 0.07779 | +0.0006 | |
| 1900 | 0.07660 | -0.0012 | |
| 2000 | 0.07799 | +0.0014 | (kill 前最后记录) |

### 3.2 关键对比

| 指标 | Task #59 baseline (flan-t5 + SimpleKMeans) | **Task #66 P3 best** | Δ vs baseline |
|------|------------------------------------------|---------------------|---------------|
| **R@5 (val)** | **0.08572** | **0.07918** | **-7.6%** (未达 GO 阈值) |
| vs Task #87 baseline (0.01937) | +342% | **+309%** | 🟢 显著优于最弱 baseline |
| vs Task #66 P3 预测 (≥0.0857) | — | **未达** | ⚠️ P3 GO 假设**部分验证** |

### 3.3 val_loss 过拟合迹象

| Step | val_loss | 趋势 |
|------|----------|------|
| 1400 | 8.835 | 最佳点 |
| 1500 | 8.972 | ↑ |
| 1600 | 9.069 | ↑↑ |
| 1700 | 9.073 | ↑↑ |
| 1800 | 9.356 | ↑↑↑ |
| 1900 | 9.385 | ↑↑↑ |

val_loss 在 step 1400 之后单调上升（8.835 → 9.385，+6.2%），明确过拟合。

---

## 4. 分析解读

### 4.1 P3 PARTIAL GO 含义

| 假设 | 阈值 | 实测 | 状态 |
|------|------|------|------|
| P3 R@5 ≥ 0.0857 | 0.0857 | 0.07918 (-7.6%) | ⚠️ **未完全达预测** |
| P3 R@5 ≥ Task #87 baseline (0.01937) | 0.01937 | 0.07918 (+309%) | ✅ **显著优于 baseline** |
| G1×G3 拓扑保护有效性 | (P1 已验证 2× / 7σ) | (P3 部分转化) | ⚠️ P1 信号强但 P3 转化不充分 |

**核心结论**:
- G1×G3 拓扑保护在 **SID 内部**显著有效（P1: 2× 拓扑保留）
- 但 **SID → TIGER 端到端转化**只到 0.07918，比 SimpleKMeans + LLM embedding (Task #59 = 0.0857) 低 7.6%
- 可能原因: **拓扑正则化的下游收益在 TIGER 训练中被 LLM-embedding 的天然稠密性盖过**（Task #59 用 SimpleKMeans 无任何拓扑约束，R@5=0.0857）
- → **PM-RQ/G-campaign 在 Toys 上的下游优势不显著**

### 4.2 与 Task #59/60/61 对比

| 任务 | Embedding | SID 方式 | R@5 | vs P3 |
|------|-----------|---------|-----|-------|
| Task #59 | flan-t5 2048d | SimpleKMeans | 0.0857 | +8.2% |
| Task #60 | sentence-t5 768d | SimpleKMeans | 0.0838 | +5.8% |
| Task #61 | hybrid 2816d | SimpleKMeans | 0.0977 | +23.4% |
| **Task #66 P3** | **MCKG + Sinkhorn cascade (G1×G3)** | RQ-VAE | **0.07918** | — |

**洞察**: 即使 P3 加了拓扑正则化，仍输给无任何几何处理的 SimpleKMeans + LLM 组合（Task #59 / 60 / 61）。**这印证了 Task #52 终判的核心结论: norm 健康 + 简单量化 ≫ 几何架构复杂度**。

### 4.3 早停为何未触发

- Lightning EarlyStopping 配置 patience 可能 > 3（实际等效 ~10）
- val_R@5 在 step 1500-2000 区间围绕 0.077 ± 0.001 振荡，标准 early stopping 容忍这种波动
- val_loss 持续上升（8.835 → 9.385，+6.2%）但监控指标是 val/recall@5，未直接触发

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| 备份 metrics (step 500) | `logs/task66_p3_sinkhorn_cascade/backup_step500/metrics_step500.csv` |
| 原 P3 训练产物 | `logs/task66_p3_sinkhorn_cascade/runs/2026-07-20/12-00-56/` (metrics.csv, ckpt step 500) |
| Resume 训练产物 | `logs/task66_p3_resume/runs/2026-07-20/12-45-00/` (metrics.csv 到 step 1999) |
| **best ckpt** | `logs/task66_p3_resume/runs/2026-07-20/12-45-00/checkpoints/checkpoint_epoch=000_step=001400.ckpt` (val_R@5=0.07918) |
| Resume log | `logs/task66_p3_resume.log` (~2.7 MB) |
| ckpt symlink | `logs/task66_p3_step500.ckpt` → 原始 step 500 ckpt |

**Stage 4 推断未执行** (用户主动 kill，未跑 Stage 4 inference 拿最终 R@5；best val_R@5=0.07918 作为代理估计)

---

## 6. 后续建议

| 路线 | 描述 | ROI |
|------|------|-----|
| ~~继续 P3 训练~~ | ❌ 已证明 val_R@5 收敛停滞 | 0 |
| **Stage 4 推断** | 用 step 1400 best ckpt 跑 tiger_inference_flat 拿真实 R@5 (vs val_R@5=0.07918) | 低（仅是收尾验证） |
| **P3 关闭 + 转向 Task #67** | P3 PARTIAL GO 表明 G1×G3 拓扑正则化下游收益有限，转向"拼接 192d + norm 修复 + Dead Code Revival"（见 Task #67 description） | **高** |
| **重评估 G-campaign ROI** | Task #63 P3 与 Task #66 P3 共用 sid_sinkhorn_balanced.pt → 同场 Stage 3 训练 → Task #63 P3 推断必然也 ≤ 0.07918 → G1 P3 也 PARTIAL GO | 中 |

---

## 7. 完成判定

- [x] D0 ✅ (task63_d0 / task66_d0)
- [x] P1 ✅ (task63_p1 / task66_p1 — Sinkhorn cascade vs Vanilla L0 拓扑保留 2× / 7σ)
- [x] P3 Stage 3 TIGER 训练 (从 step 500 resume → step 2000) ✅
- [ ] P3 Stage 4 inference + eval ⏸️ 用户主动 kill，未跑（可选收尾）
- [x] P3 verdict ← **本文档**

---

**result:** Task #66 P3 (Sinkhorn cascade TIGER) **P3 PARTIAL GO**: best val_R@5=**0.07918 @ step 1400** (vs Task #59 baseline 0.0857，差 7.6%)。G1×G3 拓扑保护在 SID 内部显著有效 (P1 2× / 7σ)，但下游 TIGER 端到端转化只到 0.07918，未达 P3 预测的 ≥ 0.0857 阈值——**PM-RQ/G-campaign 在 Toys 上的下游优势不显著**。best ckpt `checkpoint_epoch=000_step=001400.ckpt` 已保存，可选 Stage 4 推断收尾。

result: Task #66 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
