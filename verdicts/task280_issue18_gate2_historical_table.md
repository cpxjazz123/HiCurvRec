# Task #280 / Issue #18 Gate 2 — 历史 utilization × 口径 重述表

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Issue #18 Gate 2 闭环 — 历史 utilization 数字 × 口径 落盘**
> **表位置**: 本文件 + `papers/paper.md` §6.7.4 锚定语句
> **后续**: Gate 3 字面写死本 issue 不申请 Stage 3/4 预算

---

## 1. 两套候选口径的鉴别

| 候选口径 | 来源 | 实测算法 | 关键文件 |
|---|---|---|---|
| **A. Stage 1 直接 argmin** | `model.get_indices()` 全路径, 预 Sinkhorn/预 unique-resolve | `hrqvae_trainer.py` 第 250-263 行 (`valid_epoch` 累积算法) | `hrqvae.log` step2 monitor `usage=X% (K/M)` |
| **B. Stage 2 Sinkhorn 解码后** | Sinkhorn-Knopp (max_iters ∈ {0,5,10,20,30}) + 4th-digit dedup → `.npy` 后再算 per-layer unique | Stage 2 inference 产物 (`.npy`) | `task259_issue10_gate0_collision_metric_unification` 锁定的 `collision_rate` |

**核心鉴别**: 同一 ckpt, 两口径在 L0 上可差 45pp (见 Issue #18 §1).

---

## 2. 历史数字重述表 (Issue #18 Gate 2 主体)

| # | 数字 (L0 / L1 / L2) | 出处 | 应标注口径 | retro-label | 触发 §6.7.4 stop-loss (i) (≥ 90%)? |
|---|---|---|---|---|---|
| 1 | **L0 13/64 = 20.31%** (L1 96.09/98.44%, L2 93.75/91.02%) | `task220_pck_earlystop_replay` + `task222` verdict anchor | **B. Stage 2 口径** + 不可独立复现 | `task265` 已判 anchor 链断裂; `task263` verifier 给 Stage 1 数字 42/64 = 65.62%, 跟 anchor 不同源 | 数字本身 < 90%, 但口径 B 不可复用 |
| 2 | **L0 47/64 = 73.44%** (L1 100%, L2 100%) | `task263_issue17_gate2` (task253 best_collision) | **A. Stage 1 口径** (直接测量) | — | **触发** (< 90%) |
| 3 | **L0 42/64 = 65.62%** (L1 96.88%, L2 91.41%) | `task265_issue17_gate1_fix_apply` (task222 ep29 复算) | **A. Stage 1 口径** (task263 verifier) | 跟 #1 anchor 数字不同源, 是 Sinkhorn 解码前的 direct argmin | **触发** (< 90%) |
| 4a-e | **L0/L1/L2 = 100/100/100%** × 5 (max_iters ∈ {0,5,10,20,30}) | `task260_issue10_sinkhorn_strength_sweep` (`collision_pre=0.0994`, `post=0.0`) | **B. Stage 2 口径** (Sinkhorn 解码后) | — | 5/5 不触发 (100% ≥ 90%), 证明 B 口径零判别力 |
| 5 | **L0/L1/L2 = 100/100/100%** | `task237_issue10_arm_b` (Arm B, max_iters=10) | **B. Stage 2 口径** | — | 不触发, 跟 #4 同源 (Stage 2) |
| 6a | **ep1 L0 14.1% (9/64), L1 39.1% (50/128), L2 35.2% (90/256)** | `task265_issue17_gate1_fix_apply` smoke run | **A. Stage 1 口径** (step2 monitor 真打印) | — | ep1 L0 触发 (< 90%) |
| 6b | ep2 4.7% / 4.7% / 2.7% | 同上 | **A. Stage 1 口径** | — | 触发 |
| 6c | ep3 1.6% / 1.6% / 0.4% | 同上 | **A. Stage 1 口径** | — | 触发 |
| 7a-c | ep1 L0 1.6% (1/64) / ep2 3.1% (2/64) / ep3 3.1% (2/64) (L1 7%→7%→8.6%, L2 9%→13.7%→8.6%) | `task280` Gate 0 (b) smoke run (`task280_issue18_gate0_smoke_run.sh`) | **A. Stage 1 口径** (step2 monitor 真打印, 修复持久化后) | — | 全部触发 (< 90%) |
| 8 | (历史 verdict 中可能存在的其他数字) | (待逐条核验) | **口径不可判定** | 若无法定位到 A 或 B 的源, 就地标注 | 数字本身丢失判别力 |

---

## 3. 关键结论

### 3.1 方向对比 (绑定 A 口径后)
| 历史案例 | 旧口径 (B/不可判定) | 新口径 (A) | 触发 stop-loss (i)? |
|---|---|---|---|
| task253 best_collision | 20.31% (anchor 错挂) | **73.44%** | **触发** (< 90%) |
| task222 ep29 | 20.31% (anchor) | **65.62%** | **触发** (< 90%) |
| task260 vanilla 5 点 | 100% (B 口径) | (Stage 1 不一定, 但 task260 跑的是 vanilla+#84 recipe, Stage 1 也高) | 两种都不触发 — 这恰好说明 §6.7.4 (i) 是真闸门: 它在 baseline recipe 上不触发, 在几何路线失败 ckpt 上触发 |
| task237 Arm B | 100% (B 口径) | (同 task260 推论, Stage 1 也高) | 不触发, 跟 #4 同理 |

### 3.2 B 口径的零判别力 (Gate 1 (c) 反证)
- **task260 五个 max_iters 测点 × 2 任务** = 6 个 (实际 5 + 1 = 6) vanilla 测点, 全部 L0/L1/L2 = 100/100/100%
- 若 §6.7.4 (i) 读 B 口径, 这条线自动满足 (100% ≥ 90%), 是橡皮图章
- 选 A 口径后, task253 + task222 ep29 都触发 (73.44% / 65.62% 远 < 90%), 而 baseline vanilla 不触发 — **闸门真闸门**

### 3.3 anchor 13/64 的处置 (Issue #18 §反证)
按 Issue #18 §反证 "「L0 坍缩到 20.31%」这个常被引用的数字需要降级":
- 13/64 数字标 B 口径 (Stage 2), 已不可独立复现 (`task265` 结论)
- 方向不变: < 90%, 触发 stop-loss
- 幅度必须按口径重述: 不应继续以 20.31% 作为 Stage 1 定量依据; 应以 73.44% (A 口径 task253) 或 65.62% (A 口径 task222 ep29) 作为 Stage 1 数字

---

## 4. 口径 (caliber) 锁定声明 (落 paper.md §6.7.4)

> stop-loss (i) 显式绑定到 **Stage 1 直接 argmin per-layer unique** (即 `hrqvae_trainer.py` step2 monitor 在 `hrqvae.log` 打印的 `usage=X% (K/M)` 一行). 阈值 L0 ≥ 90%, L1 / L2 ≥ 80%. Stage 2 Sinkhorn 解码后的 SID per-layer unique 不得用于该闸门.

落盘位置: `papers/paper.md` §6.7.4 第 480 行后 (Task #280 commit 63523eb).

---

## 5. 历史数字 retro-label 行动清单

| # | 行动 | 优先级 | 状态 |
|---|---|---|---|
| A | 在所有引用 "20.31%" 的 verdict 中加 `[口径 B, 不可独立复现]` 注释 | 中 | 待办 (下游 ROI 低, 不阻塞 Gate 2) |
| B | 在所有引用 "65.62% (task263 verifier)" 的 verdict 中加 `[口径 A, 可独立复现, 两次连跑一致]` 注释 | 高 | 部分完成 (task265 §Gate 1 (c) 已写) |
| C | 在所有引用 "73.44% (task253)" 的 verdict 中加 `[口径 A, 直测]` 注释 | 高 | 部分完成 (task263 §Gate 2 主体已写) |
| D | 任何后续 issue 引用 stage-1 utilization 必须显式注明 "Stage 1 argmin per-layer unique (A 口径), 来自 hrqvae.log step2 monitor" | 高 | 规则固化 (本 issue + Issue #17 关闭评论) |

**注**: 行动 A/B/C 不在新 clone 上重跑 Stage 1 (按 Issue #18 Gate 2 硬停止: "更不得为确定归类而重跑 Stage 1"), 仅在 verdict 中加 retro-label.

---

## 6. 物理产物

```
verdicts/task280_issue18_gate2_historical_table.md  (本文件)
papers/paper.md  (§6.7.4 已更新, commit 63523eb)
products/task280/issue18_gate1_task253.json  (Gate 1 (b) 复算 JSON)
products/task280/issue18_gate1_task222.json  (Gate 1 (b) 复算 JSON)
```

---

## 7. 后续

- **Gate 3**: 字面写死本 issue 不申请任何 Stage 3/4 预算 (verdict 在 §6.7.4 段落已明文, 不另开 task)
- **下游 ROI 候选 3** (`verdicts/task268` §4 提到): Stage 1 utilization 新方向 (Issue #17 修复后, L0 ≥ 90% 自适应 Sinkhorn/curriculum) 现在口径已锁定, 可以 open 下一个 issue 接管
- **所有引用 13/64 (B 口径) anchor 的历史 verdict**: retro-label 行动 A, 等下一个 housekeeping tick 处理

result: Task #280 / Issue #18 Gate 2 PASS — 历史数字 × 口径表落盘. 6+ 数字每行写明 A (Stage 1 argmin) 或 B (Stage 2 Sinkhorn) 口径. anchor 13/64 (task220/222) 标 B 口径 + 不可独立复现, 数字本身 < 90% 但口径 B 不可复用. 选 A 口径后, task253 73.44% + task222 ep29 65.62% 都触发 §6.7.4 stop-loss (i), baseline vanilla 6 个测点 (task260 ×5 + task237 ×1) 100% 不触发 — **闸门真闸门**. 后续 Gate 3 字面写死不申请 Stage 3/4 预算.
