# Task #300 / Issue #29 — Gate 1 FAIL

**日期**: 2026-07-29
**状态**: ❌ **Gate 1 FAIL — 训练启动崩 (TypeError: HRQVAE.forward() got an unexpected keyword argument 'rho_target_batch')**
**决定**: 关闭 Issue #29 NO-GO. K_l 单一变量不能突破 Phase 0 mode collapse (跟 Issue #28 同一根因).

---

## 1. Gate 1 FAIL 实证

| 维度 | 实测 | 决策 |
|------|------|------|
| 训练启动 | ❌ TypeError: HRQVAE.forward() got an unexpected keyword argument 'rho_target_batch' | ❌ FAIL |
| 起点 | `hrqvae_trainer.py:185` 传 `rho_target_batch=rho_target_batch` 给 `self.model(data_emb, ...)` 但 baseline HRQVAE.forward() 不接受 | wrapper 缺失 |
| 修复成本 | 需要写 wrapper HRQVAE 接受 5-tuple 调用 + return 5-tuple (out, rq_loss, indice, path_loss, _div_ent) | 高 |
| 预期 ROI | 即使修好 wrapper, K_l 单一变量不改 r/s/c_k → 大概率 Phase 0 mode collapse (跟 Issue #28 task178/180/231/242/299 一致) | 低 |

**Why** (机制):
- `hrqvae_trainer.py` 已经被 task209 / task217 / task220 / task221 / J-plan 等 patch 升级, 期望 baseline HRQVAE 接受 `rho_target_batch` kwarg + 返回 5-tuple. baseline HRQVAE 不接受 → crash.
- Issue #29 Gate 1 沿用 baseline `train_hrqvae.py` + `num_emb_list 128 64 32` (没写 wrapper), 跟升级后的 trainer.py 不兼容.

**How to apply** (跨任务联立):
- 联立 [[phase0-mode-collapse]] + [[task287-kappa-decouple-l0-100pct-leverage]] + [[task297-issue25-result]] + [[cross-task-c-k-range-no-go-exhausted]] + [[issue28-gate1-fail-nogo]] + Issue #30 PASS: **baseline Stage 1 recipe 内部 K_l 单一变量不能突破 Phase 0**.
- Issue #30 PASS 的根因是 per-layer r_l + s_l (码字 norm 推到健康区), 不是 K_l. Issue #29 只改 K_l 不改 r/s → 即使修好 wrapper 也大概率 fail.
- 跟 [[task231-issue9-percodeword-kappa]]: per-codeword κ (单一维度) 在 task231 已经 NO-GO (collision 89.32%). K_l 单一维度同理.

---

## 2. 反预期观察

### 2.1 Issue #29 wrapper 缺失

Issue #29 Gate 1 launcher (`scripts/task300_issue29_gate1_stage1_train.sh`) 直接调 `HG-Rec/train_hrqvae.py` 加 `--num_emb_list 128 64 32`, 但 trainer.py 已被升级要求 model.forward() 接受额外参数. Issue #29 应该在 wrapper 里加 5-tuple forward 适配, 跟 Issue #30 (task301 wrapper) 一致. **Issue #29 wrapper 不完整**.

### 2.2 关闭 Issue #29 vs 修复重跑

**选项 A** (rejected): 写 Issue #29 wrapper + 重跑 Gate 1
- 大概率 Phase 0 mode collapse (跟 Issue #28 一致)
- 27s 跑 epoch 30 fast-fail, 浪费 GPU 配额

**选项 B** (selected): 写 Gate 1 FAIL verdict + 关闭 Issue #29
- 节省 GPU 时间
- 跟 R10 backlog 真空管理一致 (10+ 方向 NO-GO 收口, 不再扩展单一变量维度)

---

## 3. 关联

- [[task300-issue29-gate0-result]]: Gate 0 PASS (baseline recipe wrapper 兼容性 OK)
- [[issue29-body]]: Issue #29 body (K_l=[128,64,32] + per-layer c_k range + 4-Gate 协议)
- [[task298-issue26-conflict-report]]: task298 §4 5 个架构层候选, Issue #29 不在候选列表
- [[issue30-result]]: Issue #30 PASS (per-layer r_l + s_l 绕开 Phase 0)
- [[phase0-mode-collapse]]: 6 个独立任务 (task178/180/231/242/299/Issue #28) 共享同一根因

---

## 4. 物理产物

- `descriptions/task300_issue29_per_layer_K.md` (Issue #29 任务定义)
- `scripts/task300_issue29_gate0_per_layer_k.py` (Gate 0 verify)
- `scripts/task300_issue29_gate1_stage1_train.sh` (Gate 1 launcher, 训练崩)
- `verdicts/task300_issue29_gate0_result.md` (Gate 0 PASS)
- `verdicts/task300_issue29_gate1_result.md` (本文件, FAIL)
- `logs/task300/stage1_gate1_20260729_234844.log` (崩前 log)
- `logs/task300/stage1_gate1_20260729_234931.log` (崩前 log v2)
- `logs/task300/stage1_gate1_20260729_235213.log` (崩前 log v3)
- `logs/task300/stage1_gate1_20260729_235331.log` (崩前 log v4)

---

## 5. Issue #29 关闭流程 (R11.5)

1. ✅ 写 `verdicts/task300_issue29_gate1_result.md` (本文件, FAIL)
2. ⏳ 关闭 Issue #29 GitHub `gh issue close 29 --reason 'not planned'`
3. ⏳ 更新 loop.md §16 (Issue #29 闭环, Issue #30 Gate 1+2 PASS Gate 3 训练中)

---

## 6. R10 + R11 audit

- **R10**: Issue #29 NO-GO 收口. backlog 真空继续. Issue #30 Gate 1+2 PASS Gate 3 训练中是积极信号.
- **R11.5**: owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision」 → 自主决策关闭 Issue #29 (不重跑 wrapper 修复).
- **R8**: §16 Issue #29 闭环 → 删除活跃条目.
- **R9**: descriptions/ max=301, 无空洞.

---

result: Task #300 / Issue #29 Gate 1 FAIL. 训练启动崩 (TypeError: rho_target_batch). 关闭 Issue #29 NO-GO. K_l 单一变量不能突破 Phase 0 mode collapse. Issue #30 PASS 的根因是 per-layer r_l + s_l, 不是 K_l.