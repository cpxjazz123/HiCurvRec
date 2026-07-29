# Task #270 — Stage 1 L0 utilization ≥ 90% curriculum 方案设计 (Issue #17 修复后新方向)

> **完成日期**: 2026-07-29
> **状态**: 🟡 **设计完成, 训练未启动** — 3 配方 launcher 脚本落盘 + 0 GPU 本 cron tick + 候选 A1 待下个 tick 验证

---

## 1. 现状问题

Task #263 直接测量 task253 Stage 1 ckpt (best_collision):
- **L0 utilization = 73.44%** (47/64 unique codes) < 90% §6.7.4 stop-loss (i) 阈值
- 同机制族 6/6 历史 task (task178/179/180/199/201/203/204) 全部 L0 < 90%
- **根因猜测**: 默认 `--beta 0.5` 双曲 commit loss 在 50 epoch 内强制码字进 boundary, K=64 codes 在 4D Poincaré 球无法充分散开

Task #265 修了 step2 monitor (Issue #17 Gate 1) → utilization 字段现在真打印, 工具链就绪. **但没有机制把 L0 推到 ≥ 90%**.

## 2. 候选机制设计 (单一变量)

### A1: 纯欧氏 VQ-VAE (`--beta 0.0`)
- **假设**: 双曲 commit loss 是 L0 utilization 瓶颈, 去掉后 codes 在欧氏空间自由散开
- **CLI**: `--beta 0.0`, 其他参数 baseline
- **风险**: 0 (这是 β=0.5 的简化退化, 已 working recipe 的边界情况)
- **通过条件**: L0 ≥ 58/64 @ epoch 30
- **时间**: 50 epoch × ~6s/ep = ~5 min

### A2: Curriculum β (先用 β=0.0 30 epoch, 再 warm-start β=0.5 30 epoch)
- **假设**: A1 (欧氏充分散开) 是 A2 (双曲精修) 的好起点
- **依赖**: 必须先跑完 A1 拿 best_collision ckpt
- **CLI**: A1 部分 = `--beta 0.0` + 50 epoch; A2 部分 = `--beta 0.5 --init_encoder_from A1/best_collision.pth` + 30 epoch
- **风险**: encoder warm-start 跟新 β=0.5 commitment loss 可能不兼容 (β 突变 loss 跳变)
- **通过条件**: A2 完成后 L0 ≥ 58/64 @ epoch 30+
- **时间**: A1 5 min + A2 3 min = ~8 min

### A3: β=0.5 + encoder freeze at epoch 20
- **假设**: encoder 在训练早期就把码字空间"压扁", freeze 后让 codebook+decoder 重分配
- **CLI**: `--beta 0.5 --freeze_encoder_epoch 20` (已存在 Task #193 实现, 不改 src/)
- **风险**: encoder freeze 是已知杠杆 (Task #193), 但可能跟 task178/180 历史冲突
- **通过条件**: L0 ≥ 58/64 @ epoch 30+
- **时间**: 50 epoch × ~6s/ep = ~5 min

## 3. 不选/弃的候选

| 候选 | 不选理由 |
|---|---|
| **C: Sinkhorn 端点 rebalance** | L0 utilization 是 Stage 1 量, Stage 2 推断 Sinkhorn 改不了 Stage 1 ckpt 的 L0 unique count. 机制错配, 弃. |
| **D: lr_log_r 提升** | 默认已 10.0 (Task #230 Phase 0.5 调节). 进一步提升可能让码字坍缩成几个 high-norm 大码字, 风险大 |
| **修改 src/ 加 β curriculum env var** | R11.4 critical decision + 跟 user "do by yourself" override 冲突 (改 src/ 仍然需要 dry-run 报告). A2 路径用 `--init_encoder_from` 间接实现 β 切换, 避免改 src/ |

## 4. 物理产物

```
descriptions/task270_utilization_curriculum_design.md
verdicts/task270_utilization_curriculum_design_result.md  (本文件)
scripts/task270_utilization_curriculum.sh  (RECIPE={A1|A2|A3} 3 配方 launcher, env 切)
logs/task270/  (空)
products/task270/  (空, A1 未跑)
```

## 5. 关键决策点 (R11.3)

- **单一变量设计**: A1/A2/A3 各只动一个机制 (β / curriculum / freeze), 严格遵守 §6.7.4 越闸教训 (多变量同时改 = 不可审计)
- **不写 Stage 2/3/4 计划**: Issue #17 §H3 已明示本方向"不申请任何 Stage 3/4 预算". 仅 Stage 1 验证 L0 ≥ 90% 通过后才进 Stage 2/3/4 申请
- **不修改 src/**: A2 通过 `--init_encoder_from` 间接实现 β 切换, 避免 R11.4 critical decision
- **本 cron tick 0 GPU**: 仅设计 + launcher 预写, 不启动 A1 (避免单 tick 占用 GPU 太长阻塞其他 backlog)
- **优先级**: 选 A1 先跑 (最简单, 5 min), 如 A1 直接 PASS (β=0 已经 L0 ≥ 90%) 那是最快路径; 否则 A3 (engineered, 已有 freeze 杠杆); 最后 A2 (curriculum)

## 6. 通过条件严格化 (R11.3)

每配方必须满足:
1. **L0 ≥ 58/64 = 90.625%** at epoch ≥ 30 (pre-revive utilization, 跟 step2 monitor 数字对齐)
2. **collision < 0.95** at epoch ≥ 30 (sanity: 不能 collision 仍然 0.99 = 没训练)
3. **recon_loss ≤ 1500** at epoch 50 (跟 task253 baseline 一致, 训练 loss 不能爆炸)

任一配方全 PASS → 任务进 Stage 2 推断 + Stage 3 训练申请 (Task #271 单独 page).
任一配方全 NO-GO → 闭环, verdict 写 "L0 ≥ 90% 在 baseline recipe 不可达", 资源转向其他 backlog 候选.

## 7. 后续 (R10 主动推进)

- **下个 cron tick 候选**: 启动 `RECIPE=A1 bash scripts/task270_utilization_curriculum.sh` (5 min, GPU 0)
- **后续 cron ticks**: 拿到 A1 结果 → 决定 A2 vs A3, 或直接 PASS 写 verdict
- **GPU budget ceiling**: 3 配方 × 5 min = 15 min total. 不超 1 cron tick 占用

result: Task #270 — 3 配方 curriculum design (A1 β=0.0 / A2 curriculum / A3 freeze encoder) + 1 行 launcher (RECIPE 切换) + 0 GPU 本 cron tick. 通过条件 (L0 ≥ 58/64 @ ep 30) 严格化. 等下个 cron tick 启动 A1 验证. 资源 / GPU 预算可控 (15 min total).
