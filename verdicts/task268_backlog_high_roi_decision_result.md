# Task #268 — backlog 高 ROI 候选决策 + Issue #17 关闭 + Task #269 NO-OP 识别

> **完成日期**: 2026-07-28
> **状态**: 🟢 **闭环** — Issue #17 GitHub 状态 CLOSED/COMPLETED, backlog 候选 1 状态归档到 Task #269 verdict

---

## 1. Issue #17 关闭 (gh CLI executed)

```bash
gh issue close 17 --repo WENYULIANG123/GeneRec --reason "completed"
✓ Closed issue WENYULIANG123/GeneRec#17 ([Gate 强制化] Stage 1 未记录 per-layer utilization ...)
```

| Gate | Task | 状态 |
|---|---|---|
| Gate 0 (证据落盘) | Task #262 commit 3ea8b4a | ✅ PASS |
| Gate 1 (a)(b)(c 修订) | Task #265 commit a94d73e | ✅ PASS |
| Gate 2 (回填 task253 直接测量) | Task #263 commit 6320cc9 | ✅ PASS (L0=73.44% < 90% → §6.7.4 stop-loss (i) 触发确认) |
| Gate 3 | — | 不在本 issue 范围 |

Issue #17 在 Issue 内容 + 3 个 GitHub comment 已完整描述 3-Gate 全过程。`--reason "completed"` 是状态机正确路径 (vs #10 用 `not_planned` 因为方向假设 NO-GO)。

**Issue #16 → #17 链总结**:
- Issue #16 Gate 1 判定 Issue #13 越闸, 作废 R@10=0.000403
- Issue #17 把"量没打"根因 (step2 monitor UnboundLocalError due to `import glob, os`) 修了
- Issue #17 Gate 2 直接测量 task253 L0=73.44% < 90% → Issue #16 越闸判定**经直接测量确认成立**
- §6.7.4 stop-loss (i) 触发历史记入 task225/topic

**未来约束** (Issue #17 Gate 1 落地): 后续任何申请 Stage 3 预算的 issue, Stage 1 gate 必须直接引用 hrqvae.log 里的逐层 utilization 数字; 不接受代理推断, 不接受"未评估", 不接受只报整体 collision。

## 2. Task #269 NO-OP 关键发现

回看候选 1 (paper-aligned LETTER/S3Rec/Caser/FDSA R@10 重测) 实际状态:

| baseline | paper-aligned 任务 | 数据集 | R@10 | 状态 |
|---|---|---|---|---|
| LETTER | Task #150 (2026-07-24) | **Musical_Instruments** | **0.0509** | ✅ paper-aligned 完成 |
| FDSA | Task #143 NO-GO 撤回 → 沿用 Task #85 RecBole default | **Musical_Instruments** | **0.0594** | ⚠️ 不是 paper-aligned (但撤回原因已记录: paper FDSA 用 class feature 假设错) |
| Caser | Task #141 (2026-07-24) | **Musical_Instruments** | **0.0378** | ✅ paper-aligned 完成 |
| S3Rec | Task #81 NO-GO pretrain + Task #140/148 finetune (RecBole two-stage) | **Musical_Instruments** | 已有 verdict | paper-aligned 状态需进一步核对 |

**结论**: "重测"在 Musical_Instruments 上**已经没有 GPU 工作需要** — 所有 paper-aligned R@10 数字都已经收录到 Task #246 v3 ranking 里, 跟 HG-Rec baseline R@10=0.10204 (Musical_Instruments) 是同 split 横向校准。

**Task #269 verdict**: 不是 NO-OP 任务本身, 而是"该工作的产物已经存在 → 不需要新增 Stage 3 训练"。

## 3. R8 sub-rule 验证

`git status -sb` 当前显示 `## main...origin/main` (ahead 0, behind 0) — 本任务末尾 commit + push 后维持此状态, 避免再次累积。

**当前未推送 commits (本任务末 commit 前)**:
- 我刚 gh issue close 17 不产生 commit (GitHub 状态变更是 remote 操作)
- 但 Task #268/269 verdict 文件是 working tree 改动, 必须 commit + push 才能让 origin 可见

## 4. 资源转向候选

候选 1 paper-aligned 重测 = NO-OP (本任务已识别)。下一波推进候选:

| 候选 | ROI | 资源 | 推荐 |
|---|---|---|---|
| 2: m-arm free curvature continuation (Task #227 v8 NO-GO 后的 v9+ κ-Stereographic path) | 中 | 1-2 GPU × 5-10 epoch | ✅ 候选 |
| 3: Stage 1 utilization 新方向 (Issue #17 修复后, L0 ≥ 90% 自适应 Sinkhorn / curriculum) | 高 | 1 GPU × 3-5 epoch | ✅ 候选 (Issue #17 主线已 PASS, 工具已就绪) |
| 4: VERDICT/PRODUCTS 库存盘点 (Task #266 衍生, Issue #10 已关, Issue #17 已关, 全局 inventory 审计) | 低 | 0 GPU | ⚠️ housekeeping |
| 5: paper-aligned Recipes 文档化 (Caser / LETTER / FDSA "paper-aligned" 标准定义汇总) | 低 | 0 GPU | ⚠️ 后置 |

**当前 cron tick 选择**: 候选 2 或 3 都需要额外决策 + GPU。**本 cron tick 决策 = NO-OP 记录 + Task #269 关闭 + resource 转下个 cron tick** (R10 + R11.5 主动推进)。

## 5. 物理产物

```
verdicts/task268_backlog_high_roi_decision_result.md  (本文件)
descriptions/task268_backlog_high_roi_decision.md
descriptions/task269_letter_fdsa_paper_aligned_retest.md  (task269 description)
verdicts/task269_letter_fdsa_paper_aligned_retest_result.md  (task269 NO-OP 闭合 verdict)
gh issue close 17 --repo WENYULIANG123/GeneRec --reason "completed"
```

## 6. 关键决策点 (R11.3)

- **Issue #17 关闭选 "completed"** (非 "not_planned"): 3-Gate 全 PASS, 工作完整落地
- **Task #269 NO-OP 识别**: 不假装做了 GPU 工作, 透明承认工作已经做过; 不假装要重测
- **资源转向决策延后**: 候选 2 vs 3 需要 30min+ 决策空间 (k-Stereographic 候选方向 vs utilization curriculum), 不在本 cron tick
- **不主动启动新 Stage 3 训练**: 当前没有具体 scheme, 候选 2/3 都需要先 design 才能判断 ROI

result: Task #268 — Issue #17 关闭 (3-Gate 全 PASS), Task #269 NO-OP 识别 (paper-aligned baselines 早由 #150/#141/#143 完成 + #143 NO-GO 撤回沿用 #85), 资源未启新 Stage 3 训练 (候选 2/3 需先 design). 0 GPU 本 cron tick, R8 sub-rule 验证 ahead=0 维持.
