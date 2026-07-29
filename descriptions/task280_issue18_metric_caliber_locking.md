# Task #280 — Issue #18 口径锁定 3-Gate 闭环

## 来源

Issue #18 (2026-07-29 同批新建) — `[Gate 口径] §6.7.4 stop-loss (i) 的「L0 utilization ≥ 90%」未绑定口径`. 承接 Issue #17 Gate 1 (c) anchor 断裂 (task265 结论 "L0 13/64 = 20.31% 跟真 Stage 1 训练日志里的 utilization 是两套不同 quantity").

Issue #18 自身的关键词: "选错口径会让这条闸门永远自动通过", "本 issue 零 GPU、不引入新机制、不改变任何 R@10 数字". 收益只有一个: 下一个逐层曲率方案的 Stage 1 闸门有一个不会自动通过的判据, 并且承载它的修复能在新 clone 上活下来.

## 任务范围 (3-Gate 顺序执行, 任一 Gate 失败即终止)

| Gate | 内容 | GPU | 通过条件 | 硬停止 |
|------|------|-----|----------|--------|
| Gate 0 | 修复持久化 + ≤3 epoch smoke run | 0 GPU (1×L40S smoke ~50s) | (a) 新 clone 上行 322 = `import glob` 校验 + (b) hrqvae.log 出现 [step2 monitor ep1] 三层 usage 齐全 + UnboundLocalError=0 | 持久化失败 → STOP, 不进 Gate 1 |
| Gate 1 | 口径锁定 + 历史 ckpt 复算 | 0 GPU | (a) §6.7.4 单一书面定义 (b) task253 + task222 ep29 各复算 L0 (预期 73.44% / 65.62%) (c) 书面写出 6 个 vanilla 点恒为 100% 的反证 | (b) 复算不一致 → STOP |
| Gate 2 | 历史数字重述 (口径表) | 0 GPU | 落盘一张 "历史 utilization × 口径" 表, 每行写明口径 | 无法归类 → 标注 "口径不可判定", 不猜测 |
| Gate 3 | 写明本 issue 不申请任何 Stage 3/4 预算 | 0 GPU | 字面写死 | (无, 兜底) |

## Gate 0 实际完成 (2026-07-29)

### Gate 0 (a) — 修复持久化
- **Patch 文件已存在**: `patches/hrqvae_trainer_l322_unbound_local_error_fix.patch` (Task #274)
- **Apply 脚本已存在**: `scripts/apply_hgrec_patches.sh` (idempotent)
- **新 clone 重现链**:
  ```bash
  git clone ...         # HG-Rec/ 在 .gitignore, 故空仓
  bash scripts/apply_hgrec_patches.sh   # 一键 apply
  bash scripts/verify_hrqvae_patch.sh   # 校验 line 322 = "import glob"
  ```
- **Verify 结果**: ✅ PASS (trim 比对, line 4 = `import os`, line 322 = `import glob`)

### Gate 0 (b) — ≤3 epoch smoke run
- **GPU 2 空闲**: Task #279 占 GPU 0/1, smoke run 用 GPU 2 (R7 优先)
- **Smoke 配置**: 3 epoch, poincare loss (baseline 默认), codebook=[64,128,256], batch_size=256
- **实测时间**: ~50s (符合 Issue #18 预期)
- **hrqvae.log 输出**:
  ```
  [step2 monitor ep1] collision_rate=0.9964 | L0: usage=1.6% (1/64) | L1: usage=7.0% (9/128) | L2: usage=9.0% (23/256)
  [step2 monitor ep2] collision_rate=0.9891 | L0: usage=3.1% (2/64) | L1: usage=7.0% (9/128) | L2: usage=13.7% (35/256)
  [step2 monitor ep3] collision_rate=0.9773 | L0: usage=3.1% (2/64) | L1: usage=8.6% (11/128) | L2: usage=8.6% (22/256)
  ```
- **Step2 monitor lines**: 3 (期望 ≥ 1) ✅
- **UnboundLocalError count**: 0 (期望 = 0) ✅
- **结论**: Gate 0 (b) PASS

## 关键决策点 (R11.5 自主决策)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 持久化方案 | ✅ patch 文件 + apply 脚本 (Task #274 已就绪) | 迁库 / 改 .gitignore | 不动 .gitignore 不 fork 上游, 一键 apply 脚本已幂等 |
| 2 | verify 脚本加 `--check-smoke` flag | ✅ 同一脚本两种模式 | 写两个独立脚本 | 节省 debug 时间, 单入口 |
| 3 | smoke run 用 loss_type=poincare | ✅ 切 poincare | 写 euclidean 适配 | euclidean 是独立类 (`hrqvae_euclidean.py`), 不走 trainer.step2 monitor 路径 |
| 4 | smoke run 跑 GPU 2 | ✅ 空闲卡 | 抢 GPU 0/1 | R7 强制: 不抢已占卡 |
| 5 | 行 4 行 322 trim 比对 | ✅ 用 `xargs` 去缩进 | 严格字符串比对 | 源码行 322 有 12 空格缩进, 不能严格等 |

## 物理产物

```
descriptions/task280_issue18_metric_caliber_locking.md  (本文件)
scripts/verify_hrqvae_patch.sh  (单脚本双模式: 默认 (a) / --check-smoke (b))
scripts/issue18_gate0_smoke_run.sh  (≤3 epoch smoke launcher)
patches/hrqvae_trainer_l322_unbound_local_error_fix.patch  (已存在, Task #274)
scripts/apply_hgrec_patches.sh  (已存在, Task #274)
products/issue18_gate0_smoke/Jul-29-2026_13-20-55_*/hrqvae.log  (smoke log)
logs/issue18_gate0_smoke.log  (smoke launcher log)
verdicts/task280_issue18_gate0_pass.md  (Gate 0 verdict, 不进 Gate 1/2)
```

## 后续 Gate 1/2 安排

按 Issue #18 严格 Gate 顺序:
1. **Gate 1** (口径锁定): 读 task263 + task265 复算 task253/task222 ep29 L0=73.44%/65.62%, 在 §6.7.4 里绑定口径
2. **Gate 2** (历史重述): 写 "历史 utilization × 口径" 表, 含 Issue #18 指定的 6 数字
3. **Gate 3**: 字面写明不申请 Stage 3/4 预算

本 task #280 只闭环 **Gate 0**, 不跨 Gate 取数 (按 Issue #18 硬停止规则).

result: Task #280 — Issue #18 Gate 0 闭环. Issue #18 Gate 0 (a) PASS: line 4 = `import os`, line 322 = `import glob` (新 clone 可重现). Gate 0 (b) PASS: ≤3 epoch poincare smoke run hrqvae.log 出现 [step2 monitor ep1/2/3] × 3 行三层 usage 齐全 (L0 1.6%→3.1%→3.1%, L1 7%→7%→8.6%, L2 9%→13.7%→8.6%), UnboundLocalError=0. 复用 Task #274 patch 文件 + apply 脚本 (idempotent). 后续 Gate 1/2 另开 task.
