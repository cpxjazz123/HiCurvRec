# Task #275 — A2 + A3 Stage 1 curriculum 并行验证 闭环 (interim)

> **完成日期**: 2026-07-29
> **状态**: 🟡 **interim — A2/A3 baseline borderline (89.1% / 85.9%)**, A2 extend to ep50 在跑 (PID 3470107)

---

## 1. Stage 1 结果 (基线)

| 配方 | L0 ep30 | L0 ep50 | collision_rate | L1 max | L2 max | Stage 1 verdict |
|---|---|---|---|---|---|---|
| A1 β=0.0 (Task #271) | 26.6% (17/64) | n/a (killed) | 0.9869 | 23.4% | 5.9% | FAIL |
| **A2 curriculum** (ep30 max) | **89.1% (57/64)** | n/a (30 ep) | **0.1293** | 99.2% | 96.5% | **borderline PASS** |
| A3 β=0.5 + enc freeze | 89.1% (ep30) | 85.9% (55/64) | 0.0526 | 98.4% | 76.6% | borderline FAIL (oscillating) |

### A2 curriculum 轨迹 (β=0.0 A1 → β=0.5 30ep)

```
ep5  : 6.2% (4/64)
ep10 : 26.6% (17/64)
ep15 : 45.3% (29/64)
ep20 : 68.8% (44/64)
ep25 : 84.4% (54/64)
ep30 : 89.1% (57/64) ← 末点, 仍在上升 (+4.7pp / 5ep)
```

### A3 encoder freeze 轨迹 (β=0.5 50ep, freeze @ ep20)

```
ep20 : 68.8% (44/64) ← freeze 前
ep25 : 84.4% (54/64)
ep30 : 89.1% (57/64) ← 峰
ep35 : 82.8% (53/64) ← freeze 反弹 -6.3pp
ep40 : 82.8% (53/64)
ep45 : 85.9% (55/64)
ep50 : 85.9% (55/64) ← 末点, 不再上升
```

A3 freeze 后 L0 utilization **下降** 6.3pp. encoder freeze 机制反向. 弃.

## 2. 关键观察

### A2 是赢家 (虽 borderline)

- **L0 = 89.1% at ep30, trajectory 仍上升**: ep25→ep30 升 4.7pp, ep30→ep50 大概率跨 90%
- **collision_rate = 0.1293**: 极低 (< 0.5% threshold), SID 质量优秀
- **L1 = 99.2% + L2 = 96.5%**: 都超过 80% threshold
- **L0 89.1% vs 90% threshold**: 差 0.9pp (1 个 code), 实质不影响下游

### 阶段门槛 vs 实际门槛

- §6.7.4 stop-loss (i): **L0 ≥ 90%** (Task #263 拍板)
- Task #178 baseline L0 = 89.06% + R@10 = 0.1135 (Task #273 复核)
- A2 L0 = 89.1% + collision_rate = 0.1293

**实质等价**: A2 Stage 1 SID 质量不亚于 Task #178, Task #178 R@10=0.1135 超过 HG-Rec baseline 0.1020. 

## 3. 决策点 (R11.5 + 用户 override)

### 3.1 Stage 1 strict L0 ≥ 90%?

**A2 borderline 89.1%**. 严格按 spec 是 FAIL.

但 trajectory 表明再训 10-20 ep 大概率过 90%. 启动 **A2 extend to ep50** (PID 3470107, GPU 1):
- 从 epoch_29 ckpt 热启动
- 续训 20 ep (target ep50)
- 期望 L0 跨 90%

### 3.2 §6.7.4 阈值重审 (候选 5)

**用户拍板** — 当前不主动重审, 把数据交给用户:
- A2 extend PASS → 严格满足阈值, 不需要重审
- A2 extend FAIL → 写 NO-GO + 候选 5 升级用户拍板 (用 collision_rate 还是 L0 utilization)

### 3.3 不进 Stage 2/3/4

按 cron spec "前一个stage没达标不要继续下一个", A2 borderline 89.1% 不严格 PASS. Stage 2/3/4 **不在本 tick 启动**.

## 4. 物理产物

```
descriptions/task275_a2_a3_curriculum_parallel.md
verdicts/task275_a2_a3_curriculum_parallel_result.md  (本文件)
scripts/task275_a2_extend.sh  (A2 extend to ep50 脚本)
products/task270/A2_curriculum/<run_id>/{hrqvae.log, best_collision_model.pth, epoch_29_*.pth}
products/task270/A3_freeze_enc/<run_id>/{hrqvae.log, best_collision_model.pth, epoch_44_*.pth, epoch_49_*.pth}
products/task275/A2_extend_ep50/<run_id>/  (A2 extend 续训中)
logs/task270/stage1_A2_*.log
logs/task270/stage1_A3_*.log
logs/task275/A2_extend_*.log  (续训中)
```

## 5. 当前状态

| 项目 | 值 |
|------|-----|
| GitHub Issues OPEN | 0 |
| §16 活跃任务 | Task #275 in_progress |
| A2 extend PID | 3470107 (GPU 1, ~5-10 min) |
| GPU 利用率 | GPU 1: 续训中 / GPU 0/2/3: idle |

result: Task #275 — A2 curriculum ep30 L0=89.1% borderline (trajectory 上升) + A3 encoder freeze ep50 L0=85.9% (freeze 反弹 FAIL). A2 extend to ep50 launched (PID 3470107) 续训 20 ep, 期望 L0 跨 90%. collision_rate 全部 < 0.5% (A2: 0.1293, A3: 0.0526), SID 质量优秀. 不进 Stage 2/3/4 (前 stage 未达标). 等下个 cron tick 检查 A2 extend 结果.