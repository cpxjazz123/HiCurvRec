# Task #288 — Issue #20 L0 utilization ≥ 90% 三配方验证 (A1 β=0.0 / A2 curriculum / A3 encoder freeze)

## 来源

GitHub Issue #20: "[Curriculum Stage 1] L0 utilization ≥ 90% 三配方验证 (A1 β=0.0 / A2 curriculum warm-start / A3 encoder freeze) — §6.7.4 stop-loss (i) 从恒触发推到可触发"

承接:
- Issue #17 (closed): step2 monitor nested-scope bug 修了, utilization 真打印
- Issue #18 (closed, task280): utilization 阈值口径锁定到 Stage 1 直接 argmin
- Issue #19 (closed, task281): scripts/issue19_gate_template.sh 闸门模板
- verdicts/task270_utilization_curriculum_design_result.md: 三配方 launcher 预写

## 现状

- Task #263 直接测 task253 L0 utilization = **73.44%** (47/64 unique) < §6.7.4 stop-loss (i) 90% 阈值
- 同机制族 6/6 历史 task (task178/179/180/199/201/203/204) 全部 L0 < 90%
- task270 已预写 scripts/task270_utilization_curriculum.sh 三配方 launcher (RECIPE={A1|A2|A3} 切换)

## 任务范围 (严格按 Issue #20 Gate 0 → 1 → 2 → 3)

### Gate 0 (零 GPU, 仓库改动)
- **目标**: scripts/issue19_gate_template.sh 模板能正确求值 task270 三配方 Stage 1 产物的 utilization / collision / recon_loss
- **通过条件**: (a) task253 ckpt 回放 → 模板非零退出 (L0 < 90%); (b) task222 ep29 ckpt 回放 → 同样非零退出; (c) 伪造 90%/94%/1499 日志 → 模板正常放行
- **硬停止**: 任一不满足 → 不得启动 Gate 1
- **做法**: scripts/task288_issue20_gate0.sh, 把 issue19_gate_template.sh 模板套到 task270 launcher 输出

### Gate 1 (A1 β=0.0 纯欧氏 VQ-VAE, ~5 min, 1 GPU)
- **CLI**: bash scripts/task270_utilization_curriculum.sh RECIPE=A1 GPU=<空闲卡>
- **通过条件 (三条同时)**:
  1. L0 ≥ 58/64 = 90.625% at epoch ≥ 30
  2. collision < 0.95 at epoch ≥ 30
  3. recon_loss ≤ 1500 at epoch 50
- **硬停止**: 任一不满足 → **STOP, 不进 Gate 2, 不开 Gate 4**

### Gate 2 (A2 curriculum warm-start, ~3 min, 仅 Gate 1 通过)
- **CLI**: bash scripts/task270_utilization_curriculum.sh RECIPE=A2 GPU=<空闲卡>
- **前置**: Gate 1 best_collision ckpt 存在
- **通过条件**: 同 Gate 1
- **硬停止**: 同 Gate 1. `recon_loss` 跳变到 >1500 即 β 跳变假设不成立

### Gate 3 (A3 encoder freeze, ~5 min, 仅 Gate 1 与 Gate 2 全失败)
- **CLI**: bash scripts/task270_utilization_curriculum.sh RECIPE=A3 GPU=<空闲卡>
- **通过条件**: 同 Gate 1
- **硬停止**: 同 Gate 1. Gate 3 失败不进入 Gate 4

### Gate 4 (明确不在本任务范围, 写死)
- 本任务**不申请、不批准、不隐含任何 Stage 2 推断 / Stage 3 T5 训练 / Stage 4 端到端评估的预算**.
- 任一配方通过 Gate 1/2/3 全部三条判据**不等于**可以跑 Stage 3 — 下一轮要走 Gate 0/1/2/3 全套 + 独立 issue.
- 任何以"Stage 1 都过了, 顺手跑一下 Stage 3 / 整 batch 一起跑"为名的提议一律拒收.

## 通过条件 (Issue #20 严格化)

每配方同时满足:
1. **L0 ≥ 58/64 = 90.625%** at epoch ≥ 30 (与 §6.7.4 stop-loss (i) 同义一致)
2. **collision < 0.95** at epoch ≥ 30 (sanity, 避免"利用率高但码本未训练")
3. **recon_loss ≤ 1500** at epoch 50 (避免 β 跳变导致 loss 爆炸)

## 关键决策点 (R11.3)

- **执行顺序**: Gate 0 → Gate 1 → Gate 2 → Gate 3. 任一 Gate 失败即在该 Gate 处写 verdict 结束
- **GPU 分配**: R7 + R11.5 自主决策. 当前 4×L40S, 启动前必须 nvidia-smi 确认空闲卡
- **跨 Gate 取数禁止**: 任一 Gate 失败 = 立即 STOP, 不累积证据
- **总 GPU 上限**: ≤ 13 min (5 + 3 + 5, 按 task270 §5 R10 GPU budget ceiling)

## 物理产物

```
descriptions/task288_issue20_l0_utilization_3recipe.md  (本文件)
scripts/task288_issue20_gate0.sh  (Gate 0 闸门验证脚本)
products/task288/A1_euclidean/  (Gate 1 ckpt, 若 PASS)
products/task288/A2_curriculum/  (Gate 2 ckpt, 若 Gate 1 PASS)
products/task288/A3_freeze_enc/  (Gate 3 ckpt, 若 Gate 1/2 FAIL)
verdicts/task288_issue20_l0_utilization_3recipe_result.md  (最终 verdict)
logs/task288/stage1_A1_*.log
logs/task288/stage1_A2_*.log
logs/task288/stage1_A3_*.log
```

## R9-Enforce 备注

2026-07-29 audit: descriptions/ 存在 9 个历史空洞 (230, 255, 257-260, 264, 266, 282). 新任务用 task288 (跳过历史占位 ID).

## 反证 / 压力测试 (Issue #20 §反证)

- β=0 可能让 collision 爆炸到 >0.95 → Gate 1 (2) 是 sanity, 必须同时满足
- --freeze_encoder_epoch 冻结的是 encoder, codebook 仍可学. K=64 L0 codebook 不动, 码字也无法散开. Gate 3 必须如实记录 codebook 端有没有 step
- 三个配方全失败 = "L0 ≥ 90% 在 baseline recipe 内部无解". 闭环 NO-GO verdict, 资源转向 task268 §4 候选 2 (m-arm free curvature continuation, Task #227 v8 NO-GO 后的 v9+ κ-Stereographic path)

## 进展意义限定 (正面写)

Stage 1 utilization 到 90% 与 Stage 4 R@10 ≥ 0.1020 不是同一件事 — 本任务只产出前者. 任一配方通过只意味着 "§6.7.4 stop-loss (i) 从恒触发降到可触发", 下一轮 R@10 端点判定仍需 Issue #18 / #19 闸门 + 独立 Stage 4 评估. 不得反推 per-codeword kappa / Gromov / 任何最终目标方向应当重开.

result: Task #288 — Issue #20 L0 utilization ≥ 90% 三配方验证 (A1/A2/A3). Gate 0 → Gate 1 → Gate 2 → Gate 3 顺序执行, 任一 Gate 失败即 STOP. 总 GPU ≤ 13 min. 不申请 Stage 2/3/4 预算. 承接 task270 三配方 launcher.