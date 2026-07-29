# Task #280 / Issue #18 — 口径锁定全闭环 verdict

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Issue #18 全部 3 Gate 闭环 — 口锁定 + Gate 3 字面写死**
> **Gate 0**: PASS (修复持久化 + smoke run) — `verdicts/task280_issue18_gate0_pass.md`
> **Gate 1**: PASS (口径锁定 + 复算 PASS) — 落 `papers/paper.md` §6.7.4
> **Gate 2**: PASS (历史重述表) — `verdicts/task280_issue18_gate2_historical_table.md`
> **Gate 3**: 字面写死 (本 verdict 文件 + `papers/paper.md` 段落)

---

## 1. Issue #18 全条目验证

| Issue #18 通过条件 | 实测 / 落盘 | 结论 |
|---|---|---|
| **Gate 0 (a)** 新 clone 上 `apply + verify` 后行 322 = `import glob` (line 4 `import os` 保留) | ✅ PASS (trim 比对) | ✅ |
| **Gate 0 (b)** ≤3 epoch smoke run 出现 `[step2 monitor ep1]`, 三层 usage 齐全, UnboundLocalError=0 | ✅ 3 行 step2 monitor, L0/L1/L2 齐全, UnboundLocalError=0 (实测 ~50s, GPU 2 空闲, 不抢 Task #279) | ✅ |
| **Gate 1 (a)** §6.7.4 阈值绑定的口径有唯一书面定义, 无第二读法 | ✅ `papers/paper.md` §6.7.4 锁定声明: "Stage 1 argmin per-layer unique, 即 hrqvae_trainer.py step2 monitor 在 hrqvae.log 打印的 usage=X% (K/M)" | ✅ |
| **Gate 1 (b)** 用该口径在 task253 + task222 ep29 各复算一次, 0pp 偏差 | ✅ task253 L0 = 47/64 = **73.44%**, task222 ep29 L0 = 42/64 = **65.62%**, 跟 task263/task265 一致, 0pp 偏差 | ✅ |
| **Gate 1 (c)** 书面写出 "Stage 2 口径在 vanilla 6 测点全 100%, 闸门零判别力" | ✅ `papers/paper.md` §6.7.4 段落已写 + 6 测点列出 (task260 max_iters ∈ {0,5,10,20,30} × 5 + task237 Arm B × 1) | ✅ |
| **Gate 2** 历史 utilization × 口径表落盘, 每行写明口径 | ✅ `verdicts/task280_issue18_gate2_historical_table.md` 9 行表 (含 anchor + 复算 + 5 vanillas + smoke run + Gate 0 smoke run) | ✅ |
| **Gate 3** 字面写死不申请 Stage 3/4 预算 | ✅ `papers/paper.md` §6.7.4 "Issue #18 Gate 3" 段落已字面固化 | ✅ |

**全部 7 项 PASS. Issue #18 闭环.**

---

## 2. 关键交付物 (Issue #18 全部产出)

### 2.1 代码 / 脚本层
- `scripts/verify_hrqvae_patch.sh` (双模式: 默认 Gate 0 (a) / `--check-smoke` Gate 0 (b))
- `scripts/issue18_gate0_smoke_run.sh` (≤3 epoch smoke run launcher, GPU 2 idle)

### 2.2 文档层
- `papers/paper.md` §6.7.4 三处新增:
  - **口径锁定声明** (Gate 1 主交付物): "stop-loss (i) 显式绑定到 Stage 1 argmin per-layer unique, Stage 2 Sinkhorn 解码后的 SID per-layer unique 不得用于该闸门"
  - **Gate 3 字面写死**: "本 issue 不申请、不批准、也不隐含任何 Stage 3/4 预算. ... 与 fallback option B 是同一句式"
  - **task225 retro-label** (Gate 2 后置): 把 §6.7.5.3 段 "L0 utilization 20.31% ×13" 标 B 口径 + 不可独立复现, 真实 task222 ep29 直测 = **65.62%**

### 2.3 数据 / Verdict 层
- `verdicts/task280_issue18_gate0_pass.md` (Gate 0 通过证据)
- `verdicts/task280_issue18_gate2_historical_table.md` (Gate 2 重述表)
- `verdicts/task280_issue18_full_closure.md` (本文件, 全闭环)
- `products/task280/issue18_gate1_task253.json` + `task222.json` (Gate 1 (b) 复算 JSON, 落盘)

### 2.4 Patch 复用
- `patches/hrqvae_trainer_l322_unbound_local_error_fix.patch` (复用 Task #274, idempotent apply)
- `scripts/apply_hgrec_patches.sh` (复用 Task #274)

---

## 3. 关键决策点 (R11.5 自主决策)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 持久化方案 | ✅ patch 文件 (Task #274 已就绪) | 迁库 / 改 .gitignore | 不 fork 上游, patch 系统 idempotent |
| 2 | Verify 脚本入口 | ✅ 单脚本双模式 | 两个脚本 | 单一入口降低误用 |
| 3 | Smoke run loss_type | ✅ poincare (baseline 默认) | euclidean (单独类) | euclidean 不走 step2 monitor 路径 |
| 4 | Smoke GPU 占用 | ✅ GPU 2 空闲 (R7 强制) | 抢 Task #279 的 GPU 0/1 | 不抢已占卡 |
| 5 | Gate 1 复算脚本 | ✅ 复用 task263 verifier | 写新 verifier | task263 已稳定 0pp 偏差, 复用 |
| 6 | 复算 ckpts | ✅ task253 + task222 ep29 (Issue #18 指定) | 加跑历史 collapse 点 | Issue #18 严格 Gate 顺序 + 0 重跑策略 |
| 7 | Gate 2 retro-label | ✅ paper.md 落 + verdict 备份 | 仅 verdict | paper.md 是后续所有引用唯一源 |
| 8 | Gate 3 位置 | ✅ paper.md 字面段落 | 单独 file + 引用 | 落 paper.md 是不可绕过 |

---

## 4. 0 GPU 实际值

| 任务 | GPU | 实测 |
|---|---|---|
| Gate 0 smoke run | GPU 2 (空闲, 不抢 Task #279) | ~50s wall + 1 ckpt 落盘 |
| Gate 1 复算 | 0 GPU (verifier + ckpt 重载) | <10s |
| Gate 2 文档化 | 0 GPU | <1min |
| **总 GPU 占用** | **< 1 min on 1×L40S, 其它 GPU 不动** | ✅ 满足 Issue #18 §Gate 3 |

---

## 5. 不得跨 Gate 取数 验证

按 Issue #18 "执行顺序: Gate 0 → Gate 1 → Gate 2. 任一 Gate 失败即在该 Gate 处写 verdict 结束, 不得跨 Gate 取数":

| 阶段 | 是否仅引用本 Gate 内产物 | 验证 |
|---|---|---|
| Gate 0 verdict | ✅ 仅引用 hrqvae.log + patch state | ✅ |
| Gate 1 paper.md 编辑 | ✅ 引用 Gate 1 (b) 复算 JSON + Gate 0 修复 | ✅ |
| Gate 2 verdicts/ | ✅ 引用 Gate 0 + Gate 1 产物 + 6 个历史 verdict | ✅ |
| Gate 3 字面段落 | ✅ 引用 Issue #18 原文 + Issue #17 Gate 1 (c) anchor 链断裂事实 | ✅ |

**未跨 Gate 取数**: 无重跑 Stage 1 / 无启动 Stage 3 / 无任何 Sinkhorn 重测.

---

## 6. 后续不应做的事 (Issue #18 §反证)

- ❌ **不得反推 "per-codeword kappa 应当重开"**: 口径修正不改变 task225 的 Stage 4 端点 R@10 = 0.0938 < 0.1020
- ❌ **不得反推 "Issue #10 应当重开"**: Issue #10 的 NO-GO 由 task260 的 5 点 0pp 分离 + task237 Arm B 端点共同支撑, 与本 issue 正交
- ❌ **不得把 "L0 坍缩到 20.31%" 作为定量依据**: 该数字已 retro-label 为 B 口径 + 不可独立复现, 真实 A 口径 task222 ep29 = 65.62%, 仍触发 stop-loss 但绝对值不同
- ❌ **不得以 "顺手验证" 为名启动 Stage 3**: 与 fallback option B 同句式, task280 Gate 3 字面写死

---

## 7. 后续可做 (R10 + §6.7.4 retro-label backlog)

| # | 候选 | ROI | 备注 |
|---|---|---|---|
| 1 | 历史 verdict 中 "20.31%" anchor 的 retro-label 加注 | 低 | housekeeping, 不阻塞 |
| 2 | 下游 ROI 候选 3 (L0 ≥ 90% 自适应 Sinkhorn/curriculum) — `verdicts/task268` §4 | **高** | 口径已锁, 可以 open 下一个 issue |
| 3 | Issue #19 (同批新建, 链式 launcher 强制化 Stage 2 闸门) | 高 | 同批另一个 0-GPU issue, 跟 Gate 1 闸门脚本配套 |

候选 2/3 是 R10 §16 backlog 候选, 下一 loop tick 自主决定是否启动.

---

## 8. 物理产物 (commit 编号)

```
a8f9c1  Task #280 / Issue #18 Gate 1 — §6.7.4 口径绑定 Stage 1 argmin
        + 复算 task253 (73.44%) + task222 ep29 (65.62%) 0pp 偏差
        (含 task280/issue18_gate1_*.json 落盘)
89e9414 Task #280 / Issue #18 Gate 2 + Gate 3 — 历史 utilization × 口径表
        + 字面写死不申请 Stage 3/4 预算
        + task225 retro-label 20.31% → A 口径 65.62%
8fec0fc Task #280 / Issue #18 Gate 0 — 修复持久化 + smoke run PASS
        (line 322 'import glob', 3 epoch poincare, 0 UnboundLocalError)
```

**3 commits, 4 files changed, +340 lines**: 产物全部落盘, Issue #18 闭环.

---

result: Task #280 / Issue #18 全部 3 Gate 闭环 — Gate 0 修复持久化 + smoke run PASS, Gate 1 口径绑定 Stage 1 argmin + 复算 PASS (task253 73.44% / task222 ep29 65.62%), Gate 2 历史数字 × 口径表落盘 (9 行), Gate 3 字面写死不申请任何 Stage 3/4 预算. 共 3 commits (+340 lines), 0 GPU 占用 < 1 min (smoke run GPU 2 空闲). 落 `papers/paper.md` §6.7.4 主交付物: "stop-loss (i) 显式绑定到 Stage 1 argmin per-layer unique, Stage 2 Sinkhorn 解码后 SID per-layer unique 不得用于该闸门". 6 个 vanilla 测点 (task260 ×5 + task237 ×1) 100%/100%/100% 反证 B 口径零判别力. **闸门真闸门**: task253 + task222 ep29 A 口径均触发 (< 90%), baseline vanilla 不触发. 不改变任何 R@10 数字, 不引入新机制. 后续 backlog: 候选 2 (高 ROI L0 ≥ 90% 自适应) / 候选 3 (Issue #19 链式 launcher 强制化).
