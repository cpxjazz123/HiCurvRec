# Task #131 — Empty Orphan Dir Cleanup + Explicit Gitignore Hygiene

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: (a) 删除空 orphan dir `https:/github.com/zar123123` (0 文件, 用户 git clone 失败残留); (b) `.gitignore` append 显式 rules: `https:/` + `log/` (singular) + `logs_curv_*/`. 当前 `*.log` 已覆盖大多数 transient 内容, 加显式 rule 让 reviewer 一眼看懂. dispatcher 7/7 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| 删除前 verify 0 files | `find https: -type f \| wc -l` | ✅ 0 |
| `rm -rf https:` 成功 | `ls https:` (should fail) | ✅ |
| .gitignore append 3 rules | `grep -cE "^(https:/\|log/\|logs_curv)" .gitignore` | ✅ 3 |
| git status 仍只 3 untracked (configs/src/tools) | `git status --short \| wc -l` | ✅ 5 (含 task131 desc + 2 edits) |
| dispatcher 7/7 PASS | `python3 scripts/all_audits.py` | ✅ |

---

## 2. 删除 dry-run (R11.4)

```bash
$ find /home/wlia0047/ar57/wenyu/GeneRec/https: -type f | wc -l
0
```

确认 0 文件 → 安全删除 (无 R2 fallback 风险).

---

## 3. .gitignore 新增 rules

```gitignore
# === Orphan cleanup (Task #131) ===
# Empty orphan dir from failed git clone attempt (https://github.com/zar123123)
https:/
# Singular 'log/' baseline experiment logs (HGN/SASRec, already *.log covered)
log/
# Per-curvature transient experiment logs (Task #88)
logs_curv_*/
```

### 为什么显式 gitignore (虽然 *.log 已覆盖)?

| 收益 | 说明 |
|------|------|
| Reviewer 友好 | `git status --short` 显示的 untracked dir 名字立刻说明意图 |
| Documentation as code | .gitignore 文件本身就是 reviewer-facing docs |
| 防御性 | 即使未来 *.log 规则被改, dir 仍被显式 ignore |
| 避免歧义 | `log/` (singular) vs `logs/` (plural) pattern 显式区分 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否删除 https:/ | ✅ 删除 (0 文件, 纯 junk) | ❌ 不删 (永久留垃圾) |
| 是否显式 gitignore log/ | ✅ 加 (虽然 *.log 已覆盖) | ❌ 不加 (但 reviewer 困惑) |
| 是否显式 gitignore logs_curv_*/ | ✅ 加 (per-curvature pattern) | ❌ 不加 |
| 是否删 log/ 和 logs_curv_*/ 内容 | ❌ 不删 (transient logs, R2 no fallback) | ✅ 全删 (但 R2 禁止) |
| 是否 gitignore results/ | ❌ 已存在 (line 228, Task #119 加) | n/a |

---

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| https:/ 删除前 0 files | `find https: -type f \| wc -l` | ✅ 0 |
| https:/ 删除后不存在 | `ls https:` (fail) | ✅ |
| .gitignore 3 rules 加 | `grep -cE "^(https:/\|log/\|logs_curv)" .gitignore` | ✅ 3 |
| git status | `git status --short` | ✅ 5 entries (configs/src/tools + task131 desc + .gitignore + loop.md) |
| dispatcher 7/7 PASS | `python3 scripts/all_audits.py` | ✅ 7/7 |

---

## 6. 关联

- 前置: Task #130 (final state closure, project 真正 closure)
- 后置: 真正 cleanup done. reviewer-facing workspace 最小化 (only 3 GRID upstream dirs untracked)

---

result: Task #131 — empty orphan dir cleanup + gitignore hygiene 闭环. 删除空 orphan `https:/github.com/zar123123` (0 文件). .gitignore append 显式 rules `https:/` + `log/` + `logs_curv_*/`. git status 仍只 3 untracked (configs/src/tools GRID upstream). dispatcher 7/7 PASS 不破坏. reviewer-facing workspace 最小化.
