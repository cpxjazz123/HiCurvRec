# Task #124 — README.md + TASKS_INDEX.md Stale Number Refresh v2 (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: README.md line 58 "descriptions (114 entries)" → "(125 entries)" (实际 127 总, 125 编号 + 2 参考). TASKS_INDEX.md 同步更新 descriptions/ count 123 → 127 + verdicts/ 253 → 255 + Coverage matrix 加 Task #122-#123 + Recent closed tasks 表加 #122/#123. dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| README.md line 58 updated | `grep "descriptions/" README.md \| head -3` | ✅ "125 entries" |
| TASKS_INDEX.md descriptions count | `grep "125 files" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX.md verdicts count | `grep "255 files" TASKS_INDEX.md` | ✅ |
| Coverage matrix Task #122-#123 added | `grep "Task #122 - Task #123" TASKS_INDEX.md` | ✅ |
| Recent closed tasks #122/#123 added | `grep "#122\|#123" TASKS_INDEX.md` | ✅ |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |

## 2. Counts after Task #124

| Metric | Value | Source |
|--------|-------|--------|
| descriptions/ tracked | 127 | `git ls-files descriptions/ \| wc -l` |
| descriptions/ numbered | 125 | 123 unique IDs + 2 duplicates (task27 + task67) |
| descriptions/ unique IDs | 123 | Task #1 - Task #123 contiguous |
| descriptions/ reference | 2 | README.md + _general_pipeline.md |
| scripts/ tracked | 483 | `git ls-files scripts/ \| wc -l` |
| verdicts/ tracked | 255 | `git ls-files verdicts/ \| wc -l` |

## 3. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| descriptions/ 数字 (114 → ?) | ✅ 125 entries (Task #1-#123 + 2 duplicate 修订) | ❌ 123 entries (TASKS_INDEX 之前错算) |
| 是否清理 task27/task67 duplicate | ❌ 不清理 (历史修订文件, 删除会丢失信息) | ✅ rm duplicates (但 R9 不强制 unique, 修订历史有价值) |
| TASKS_INDEX 是否同步更新 | ✅ 同步 (cross-validation, 两个文档数字一致) | ❌ 仅 README (但两个文档数字不一致 reader 困惑) |
| verdicts/ count | ✅ 255 (含 task124 verdict + 历史 data files) | ❌ 254 (漏 task124) |

## 4. R9 重复 audit 发现

`git ls-files descriptions/ | grep -oE "task[0-9]+_" | sort | uniq -d` 返回:
- `task27_` 重复 (task27_neighborhood_quality_recall_correlation.md + task27_plan_bcd_sample_expansion.md)
- `task67_` 重复 (task67_concat_192d_rqvae_norm_fix.md + task67_v6_norm_fix_no_revival.md)

**这是 renumbering 残留** (Task #27 早期有 v1 + v2 修订; Task #67 有 norm fix + revival 修订). 修订历史有价值, **不清理**, 但 TASKS_INDEX 标注 "Task #27 and Task #67 have 2 revision files each".

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| README line 58 updated | `grep "descriptions/" README.md` | ✅ "125 entries" |
| TASKS_INDEX descriptions count | `grep -A1 "descriptions/:" TASKS_INDEX.md` | ✅ "127 files total — 125 numbered" |
| TASKS_INDEX verdicts count | `grep -A2 "verdicts/:" TASKS_INDEX.md` | ✅ "255 files" |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| tracked files 增量 | `git ls-files \| wc -l` | ✅ 915 → 917 (+TASKS_INDEX.md + README.md edits) |

## 6. 关联

- 前置: Task #113 (README stale number refresh v1) + #122 (TASKS_INDEX refresh)
- 后置: 无 (documentation drift 闭环)

---

result: Task #124 — README + TASKS_INDEX stale number refresh v2 闭环. README line 58 "114 entries" → "125 entries". TASKS_INDEX descriptions 123 → 127, verdicts 253 → 255. Coverage matrix + Recent closed tasks 同步加 Task #122/#123. 发现 R9 duplicates (task27 + task67 各 2 修订文件), 标 "renumbering 历史" 不清理. dispatcher 5/5 PASS 不破坏.