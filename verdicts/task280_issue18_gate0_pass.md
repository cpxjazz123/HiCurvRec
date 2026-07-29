# Task #280 — Issue #18 Gate 0 verdict (修复持久化 + smoke run PASS)

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Issue #18 Gate 0 闭环 — Gate 0 (a) PASS + Gate 0 (b) PASS**
> **后续**: Gate 0 通过, 可推进 Gate 1 (口径锁定). 本 verdict 仅闭环 Gate 0, 不跨 Gate 取数.

---

## 1. Issue #18 Gate 0 通过条件逐条验证

| 通过条件 (Issue #18 Gate 0 原文) | 实测 | 结论 |
|---|---|---|
| (a) 在新 clone 上执行仓库内一条命令后, 校验脚本确认行 322 为 `import glob` (顶层 line 4 `import os` 保留) | `apply_hgrec_patches.sh` + `verify_hrqvae_patch.sh` 双 PASS (line 4 = `import os`, line 322 trim = `import glob`) | ✅ PASS |
| (b) ≤3 epoch smoke run 的 `hrqvae.log` 出现 `[step2 monitor ep1]` 且三层 usage 数字齐全, UnboundLocalError 计数为 0 | 3 行 step2 monitor, 三层 usage 数字齐全 (L0 1.6%→3.1%→3.1%, L1 7%→7%→8.6%, L2 9%→13.7%→8.6%), UnboundLocalError=0 | ✅ PASS |

---

## 2. 关键证据 — hrqvae.log 三行 step2 monitor

```
[step2 monitor ep1] collision_rate=0.9964 | L0: r_std=0.0000 r_min=0.000 r_max=0.000 usage=1.6% (1/64) | L1: r_std=0.0000 r_min=0.000 r_max=0.000 usage=7.0% (9/128) | L2: r_std=0.0000 r_min=0.000 r_max=0.000 usage=9.0% (23/256)
[step2 monitor ep2] collision_rate=0.9891 | L0: r_std=0.0000 r_min=0.000 r_max=0.000 usage=3.1% (2/64) | L1: r_std=0.0000 r_min=0.000 r_max=0.000 usage=7.0% (9/128) | L2: r_std=0.0000 r_min=0.000 r_max=0.000 usage=13.7% (35/256)
[step2 monitor ep3] collision_rate=0.9773 | L0: r_std=0.0000 r_min=0.000 r_max=0.000 usage=3.1% (2/64) | L1: r_std=0.0000 r_min=0.000 r_max=0.000 usage=8.6% (11/128) | L2: r_std=0.0000 r_min=0.000 r_max=0.000 usage=8.6% (22/256)
```

- 3 个 epoch 全部打印
- 三层 (L0/L1/L2) usage 数字齐全, 形如 `usage=X.X% (N/M)`
- 没有 UnboundLocalError 警告
- 修复前实测 (Issue #17 关闭评论记录): 每 epoch 静默打 `[step2 monitor] logging failed (non-fatal): local variable 'os' referenced before assignment`

---

## 3. 持久化机制 (Gate 0 (a))

### 3.1 新 clone 重现链
```bash
git clone <repo>                  # HG-Rec/ 在 .gitignore, 故空仓
bash scripts/apply_hgrec_patches.sh   # 一键 apply (idempotent)
bash scripts/verify_hrqvae_patch.sh   # 校验
# 期望输出: ✅ Issue #18 Gate 0 (a) PASS
```

### 3.2 复用 Task #274 基础设施
- `patches/hrqvae_trainer_l322_unbound_local_error_fix.patch` (496 bytes, 1 hunk, 1 line 改动)
- `scripts/apply_hgrec_patches.sh` (idempotent, dry-run + reverse dry-run 双重检测)
- `patches/README.md` (5 patches 当前, 1 个 hrqvae_trainer_l322 fix)

### 3.3 Verify 脚本双模式
```bash
bash scripts/verify_hrqvae_patch.sh             # 仅 Gate 0 (a) 行 322 校验
bash scripts/verify_hrqvae_patch.sh --check-smoke  # Gate 0 (a) + (b) 复合校验
```

---

## 4. 0 GPU 约束遵守

| 项目 | 期望 | 实测 |
|---|---|---|
| Smoke run GPU | GPU 2 空闲 (R7 强制) | GPU 2 ✅ |
| Task #279 占卡 | GPU 0/1 | GPU 0/1 仍 96% util, smoke 不抢 ✅ |
| Stage 3 / Stage 4 预算申请 | 不申请 (Issue #18 Gate 3 字面写死) | 不申请 ✅ |

---

## 5. Gate 顺序遵守

按 Issue #18 硬停止规则:
- "执行顺序: Gate 0 → Gate 1 → Gate 2. 任一 Gate 失败即在该 Gate 处写 verdict 结束, **不得跨 Gate 取数**".
- 本 verdict **只覆盖 Gate 0**. Gate 1 (口径锁定 + 复算) 与 Gate 2 (历史数字重述) 不在本 task 范围.

---

## 6. 关键决策点 (R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Patch 持久化方案 | ✅ 复用 Task #274 patch 文件 + apply 脚本 | 迁库 / 改 .gitignore / 写新 patch | 已有基础设施幂等 + 0 改造 |
| 2 | Verify 脚本入口 | ✅ 单脚本 + `--check-smoke` flag | 两个独立脚本 | 单入口降低误用 |
| 3 | Smoke run loss_type | ✅ `loss_type=poincare` | `loss_type=euclidean` (单独类) | euclidean 不走 trainer.step2 monitor, 必须用 poincare |
| 4 | Smoke GPU | ✅ GPU 2 空闲 | 抢 GPU 0/1 | R7 强制: 不抢已占卡 |
| 5 | 行 322 比对 | ✅ trim 前后空白 | 严格字符串 | 源码有 12 空格缩进 |

---

## 7. 后续 Gate 范围

- **Gate 1** (口径锁定, 0 GPU):
  - (a) 在 §6.7.4 把 stop-loss (i) 显式绑到 Stage 1 argmin per-layer unique
  - (b) 用 task263 + task265 verifier 在 task253 best_collision + task222 ep29 上复算 L0 (预期 73.44% / 65.62%)
  - (c) 书面写出 "若改选 Stage 2 口径, task260 5 点 + task237 Arm B 共 6 点恒为 100%, 闸门零判别力"
- **Gate 2** (历史重述, 0 GPU): 出一张 "历史 utilization × 口径" 表, 至少含 Issue #18 指定的 6 数字
- **Gate 3** (字面写死): 不申请任何 Stage 3/4 预算

均另开 task (per Issue #18 不得跨 Gate 取数 + R9 任务编号连续).

---

## 8. 物理产物

```
descriptions/task280_issue18_metric_caliber_locking.md
verdicts/task280_issue18_gate0_pass.md  (本文件)
scripts/verify_hrqvae_patch.sh  (双模式 Gate 0 (a) + (b) 校验)
scripts/issue18_gate0_smoke_run.sh  (smoke launcher)
products/issue18_gate0_smoke/Jul-29-2026_13-20-55_beta_0.500_codebook_[64,128,256]_sk_0.000/
  ├── hrqvae.log  (3 行 step2 monitor, 0 UnboundLocalError)
  ├── best_loss_model.pth
  ├── best_collision_model.pth
  └── epoch_*_collision_*.pth
logs/issue18_gate0_smoke.log  (smoke launcher stdout/stderr)
patches/hrqvae_trainer_l322_unbound_local_error_fix.patch  (复用 Task #274)
scripts/apply_hgrec_patches.sh  (复用 Task #274)
```

result: Task #280 — Issue #18 Gate 0 PASS. (a) 新 clone 可重现: `apply_hgrec_patches.sh` + `verify_hrqvae_patch.sh` 双 PASS. (b) ≤3 epoch poincare smoke run hrqvae.log 三行 [step2 monitor ep1/2/3] 全打, 三层 usage 齐全 (L0 1.6%→3.1%→3.1%, L1 7%→7%→8.6%, L2 9%→13.7%→8.6%), UnboundLocalError=0. 0 GPU (Task #279 占 GPU 0/1, smoke 用 GPU 2). 后续 Gate 1 (口径锁定) + Gate 2 (历史重述) + Gate 3 (字面写死) 另开 task. 严格按 Issue #18 "不得跨 Gate 取数" 硬停止规则, 本 verdict 不覆盖 Gate 1/2.
