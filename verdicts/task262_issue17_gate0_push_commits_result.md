# Task #262 — git push 14 commits + Issue #17 Gate 0 PASS + Issue #16 Gate 0 错判纠正

## 关键发现 (严重)

**本地 main 领先 origin/main 14 个 commits**, 全部 Task #248-#261 都没 push. Issue #17 §3 用非递归 `gh api contents/verdicts` 核查准确识别了这个事实, **比 Issue #16 §4 的 `recursive=1` 核查更准确**.

Issue #16 关闭评论里我写的"7 个 verdict 都在 main 分支"是**错的**: 它们在本地 main (5b52c13) 但**不在 origin/main (564ab59)**. Issue #16 Gate 0 的反驳"§4 全树缺失是核查方法论错误"**不成立**, 实际上 #16 自己也处于"数字不可审计"状态 (跟它对 #13 的批评一致).

**Issue #17 §3 的核查方法 (非递归 contents API) 是更可靠的核查手段**. 我之前误信了本地 commit history 等于远程 main, 这是 cron tick 一直 commit 但从未 push 的累积问题.

## Gate 0 操作

按 Issue #17 §Gate 0 通过条件:

> 上述文件在 main 分支可通过 `gh api repos/WENYULIANG123/GeneRec/contents/verdicts/` 单文件直接取到 (非递归命中)

执行: `git push origin main` → 564ab59..5b52c13 main -> main

推送后重新核查 (commit 不变, 是同一 14 个 commits 推到 remote):

```
verdicts/ 目录: 479 条 → 497 条
✅ task248: task248_issue13_gate0_residual_operator_result.md
✅ task249: task249_gate1_consistency.json (注: 是 .json 不是 .md, Issue #17 §3 误判)
✅ task250: task250_issue14_gate0_forge_metric_def_result.md
✅ task251: task251_band_unification.md (注: 是中间文件不是 _result.md, Issue #17 §3 误判)
✅ task252: task252_issue15_gate1_predictive_validity_result.md
✅ task253: task253_issue13_gate2_mobius_residual_result.md
✅ task254: task254_gate3_logmap_argmin.json (注: 是 .json 不是 _result.md, Issue #17 §3 误判)
✅ task257: task257_issue16_gate0_evidence_recovery_result.md
✅ task258: task258_issue16_gate1_stop_loss_audit_result.md
✅ task259: task259_issue10_gate0_collision_metric_unification_result.md
✅ task260: task260_issue10_sinkhorn_strength_sweep.json (注: 是 .json 不是 _result.md, Issue #17 §3 误判)
✅ task261: task261_loop_md_sync_status_result.md
```

**Issue #17 §3 的核查方法有个小错**: 它以"taskXXX_*_result.md"格式匹配, 但部分 verdict 实际命名带 .json / 临时 _result.md (e.g. task251 是 `task251_band_unification.md` 不是 `task251_*_result.md`). 用 prefix 匹配 (task249, task250, ...) 而不是全名匹配, 12 个 verdict 全部命中.

**Gate 0 PASS** —— 12 个 verdict 在 origin/main 全部可取到.

## 错判纠正: Issue #16 Gate 0 反推不成立

Issue #16 §4 写 "verdict 文件在仓库中全部不存在", 我在 Task #257 (commit 9eddbde) 里反驳 "§4 全树缺失是核查方法论错误 (recursive=1 在 GeneRec 大仓库截断)".

**真相**: Issue #16 §4 写的"verdict 缺失"**部分成立** — 那时本地确实没 push, GitHub 端确实没这些 verdict. 我用本地 `ls verdicts/` + `git log --oneline -- verdicts/task<N>` 看到的是**本地 commit history**, 跟 origin/main 状态不同步. Issue #16 §4 的 `recursive=1` 核查**方法本身没错**, 只是它看到的是 origin/main 的实际状态 (那时确实没这些文件, 因为我没 push). 我的反驳把"本地有"跟"origin/main 有"混淆了.

**修正**: Issue #16 Gate 0 的 §4 反驳不成立, 但 **Issue #16 Gate 1 的 §6.7.4 stop-loss (i) 越闸判定仍然成立** (那个判定用 task222 L0 = 20.31% 做同机制族 proxy, 不依赖本地 push 状态). R@10 = 0.000403 作废的结论保留, 几何路线 NO-GO 退回到 task225 0.0938 的结论保留.

## Issue #17 Gate 1 / Gate 2 / Gate 3 (本任务范围外)

按 Issue #17 §阶段闸门:

- **Gate 0**: PASS (本任务)
- **Gate 1**: Stage 1 逐层 utilization 打印 (零 GPU, 代码改动 + ≤3 epoch smoke run)
- **Gate 2**: 回填 task253 的直接测量 (廉价, 仅前向统计)
- **Gate 3**: 明确不在本 issue 范围 (不申请任何 Stage 3/4 预算)

**本任务只闭环 Gate 0**. Gate 1/2 留待后续推进. 但这是**正向持续任务** (R10 主动推进), 下个 cron tick 可以尝试 Gate 1 (代码改动).

## 关键决策点 (R11.3)

- **立即 push 不等用户决策**: R11.4 不可逆 + R2 禁止 fallback. push 不破坏任何东西, 反过来"知道有 14 commits 没 push 还继续 commit 不 push"才是真违反 R8 (R8 要求保持 origin/main 跟本地 main 同步 — 之前已偏离 14 commits). 立即执行.
- **GitHub API 优先于本地 ls**: 以后核查 "在 main 分支"必须用 `gh api contents/...` 或 `git ls-tree origin/main`, 不能再用本地 git log. R8 检查项加一条: cron tick 必须先 `git fetch` + `git status -sb` 检查 ahead/behind.
- **不重写 Issue #16 关闭评论**: Issue #16 Gate 1 越闸判定成立, 改 Issue #16 评论会越改越乱. 在本 verdict (task262) 记录错判事实即可. Issue #17 GitHub 评论里说明 + 在 Task #257 verdict 里追加 §note (本地 commit ≠ origin/main 状态).
- **追加 memory**: 本轮教训 = "AI 自动 commit 后必须立即 push, 不能等到 cron tick 结束才 batch push". R8 加 sub-rule: 每个 cron tick commit 后**必须立即** push (除非 push 会跟其他并行任务冲突).

## 物理产物

```
git push origin main  (把 14 commits 从本地 main 推到 origin/main)
verdicts/task262_issue17_gate0_push_commits_result.md  (本文件)
descriptions/task262_issue17_gate0_push_commits.md  (新增)
```

## 后续

- Issue #17 GitHub 评论: Gate 0 PASS, 12 个 verdict 可取到
- Issue #17 Gate 1 (代码改动): 下个 cron tick 推进 (Stage 1 训练循环加 per-layer utilization 打印)
- R8 修订: 加 sub-rule "commit 后立即 push, 不能 batch 累积"

result: Task #262 — git push 14 commits 成功 (564ab59..5b52c13 main -> main). Issue #17 Gate 0 PASS: 12 个 verdict (task248-#261) 在 origin/main 全部可取到 (gh api contents/verdicts 非递归核查). 错判纠正: Issue #16 §4 "verdict 缺失"当时确实成立 (本地没 push), 我 Task #257 反驳不成立, 但 Issue #16 Gate 1 越闸判定仍成立 (proxy 链独立). R8 加 sub-rule: cron tick commit 后立即 push, 不 batch 累积.