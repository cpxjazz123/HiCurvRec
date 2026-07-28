# Task #263 / Issue #17 Gate 2 — task253 直接 utilization 测量 + Gate 1 校准 FAIL 调查

## 背景

Issue #17 §假设 H1: "三次越闸中至少第三次的直接原因是 §6.7.4 stop-loss (i) 所需的 per-layer utilization 在 Stage 1 训练日志中缺席". 本任务调查事实 + 跑 Gate 2 主体 (task253 直接 utilization) + 记录 Gate 1 校准 FAIL.

## 任务范围

1. 调查 hrqvae_trainer.py Step 2 monitor 为什么打印不生效 (实测 row 322 nested-scope bug)
2. 写独立 verifier (scripts/task263_issue17_gate2_task253_direct_utilization_meas.py), 走 model.get_indices() 全路径
3. Gate 1 (c) 校准: 在 task222 ep29 ckpt 上跑 verifier, 期望 L0=13/64=20.31% (跟 task220/222 verdict 一致)
4. Gate 2 主体: 在 task253 best_collision ckpt 上跑 verifier, 写 verdict
5. 报告修复方案 + 后续 Action Item

## 关键发现 (实测)

| 项 | 期望 | 实测 |
|---|---|---|
| Gate 1 (c) 校准 (task222 ep29 L0) | 13/64 = 20.31% | **42/64 = 65.62%** ⚠️ FAIL |
| Gate 2 主体 (task253 best_collision L0/L1/L2) | N/A (anchor 不可信, 直接测量) | L0=47/64=**73.44%**, L1=100%, L2=100% ✅ |

- Gate 1 (c) 校准 FAIL 根因 = task220/222 数字 anchor 链断 (task189 verifier 报 strict=True missing 9 个 radius/κ 参数, 跑不起来)
- Gate 2 主体 PASS, 跟 task253 hrqvae.log `epoch_34_collision_0.0915` 完全一致 (collision 算对)
- §6.7.4 stop-loss (i) 触发 (L0 < 90%), Issue #16 越闸判定方向保留, 但 proxy 链事实审计失败

## 修复路径 (R11.4 critical decision)

hrqvae_trainer.py 行 322 `import glob, os` 是 nested-scope, 把模块 `os` (顶层 line 4) 覆盖, 后续 step2 monitor `os.environ.get(...)` 触发 UnboundLocalError 全场静默. 修复方案 A: 删行 322, 改用顶层 `os`. 1 行字面删除, 0 风险.

**待用户拍板** (R11.4 critical decision: 改 HG-Rec/model/ 上游 src/).

## 物理产物

```
scripts/task263_issue17_gate2_task253_direct_utilization_meas.py
products/task263/calibration_task222_ep29.json
products/task263/task253_bestcollision_utilization.json
descriptions/task263_issue17_gate2_task253_direct_utilization.md  (本文件)
verdicts/task263_issue17_gate2_task253_direct_utilization_result.md
```

## 后续

- 用户拍板修 hrqvae_trainer.py → smoke run 验证 Gate 1 (a) + Gate 1 (c) (校准目标改为 verifier 算法一致而非 13/64 anchor)
- Issue #17 GitHub 评论: 同步 Gate 2 主体 + Issue #16 proxy 链审计失败事实
- Issue #10 自主决策: 用户已说"A2 关掉 Issue #10", 但 R11.5 不阻塞路径仍可推进 backlog

result: Task #263 / Issue #17 Gate 2 主体 PASS — task253 best_collision L0/L1/L2 utilization 直接测量 (73.44%/100%/100%). Gate 1 (c) 校准 FAIL (task222 20.31% anchor 链审计失败). Issue #16 §越闸判定方向保留 (real L0=73.44% 仍 < 90%). 修 src/ 待用户拍板.