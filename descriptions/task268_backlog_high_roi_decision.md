# Task #268 — 推进 backlog 高 ROI 候选 + Issue #17 闭环 + Task #269 NO-OP 识别

## 背景

Task #267 关闭 Issue #10 后,按 verdict §后续列出 3 个 backlog 高 ROI 候选:

1. paper-aligned LETTER/S3Rec/Caser/Fdsa 重跑 R@10
2. m-arm free curvature continuation (Task #227 v8 collision NO-GO 后)
3. Stage 1 utilization-targeted 新方向 (Issue #17 主线已 PASS)

候选 1 在本任务被选为下一推进方向,并展开为 Task #269。

## 任务范围

1. 关闭 Issue #17 (gh CLI, 已 executed 2026-07-28T23:38Z, state=CLOSED stateReason=COMPLETED)
2. 核对 backlog 候选 1 的实际状态: paper-aligned baselines 已完成度
3. 写出 Task #269 verdict 反映 NO-OP 性质
4. commit + push (R8 sub-rule)
5. 资源转向候选 2 (m-arm free curvature) 或候选 3 (utilization-targeted 新方向)

## 关键决策点 (R11.3)

- **Issue #17 关闭路径**: GitHub issue 主体在 Task #265 comment + Task #262 commit 已写明 3-Gate 全 PASS。`gh issue close 17 --reason "completed"` 是 issue 状态机最后一步,不需新决策。
- **Task #269 NO-OP 关键发现**: Task #150 LETTER paper-aligned **早就在 Musical_Instruments 上完成** (R@10=0.0509, 2026-07-24 commit)。Task #143 FDSA paper-aligned NO-GO 撤回, 沿用 Task #85 RecBole default R@10=0.0594。**"重测"动作实际没有 GPU 工作需要**。
- **资源转向**: 候选 1 实际上 = "已存档到 Task #246 v3 ranking"。下一个真正需要 GPU 的 backlog 是候选 2 或 3。

## 物理产物

```
descriptions/task268_backlog_high_roi_decision.md  (本文件)
verdicts/task268_backlog_high_roi_decision_result.md
descriptions/task269_letter_fdsa_paper_aligned_retest.md
verdicts/task269_letter_fdsa_paper_aligned_retest_result.md
gh issue close 17 --repo WENYULIANG123/GeneRec --reason "completed"  (executed 2026-07-28T23:38Z)
git commit + push  (本任务末尾)
```

## 后续 (R10 主动推进)

- 候选 2 (m-arm free curvature continuation): Task #227 v8 collision NO-GO 是 product_manifold 上限, 但 k-Stereographic 候选 v9+ 是另一条路径, 未尝试
- 候选 3 (Stage 1 utilization 新方向): Issue #17 修复后 step2 monitor 真打印 utilization, 可以设计 L0 ≥ 90% 自适应 Sinkhorn / curriculum 方案

result: Task #268 — backlog 候选 1 (paper-aligned) 推进: 关闭 Issue #17 (Gate 0/1/2 全 PASS) + 写出 Task #269 NO-OP 识别 (LETTER/FDSA/Caser paper-aligned 早由 #150/#143/#141 完成). 资源转候选 2 或 3.
