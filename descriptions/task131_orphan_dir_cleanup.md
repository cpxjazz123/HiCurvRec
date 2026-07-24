# Task #131 — Empty Orphan Dir Cleanup + Explicit Gitignore Hygiene

> **任务目的**: (1) 删除空 orphan dir `https:/github.com/zar123123` (用户 git clone 失败残留, 0 文件, 纯 junk); (2) .gitignore 加显式规则覆盖 `log/` (singular, orphan) + `logs_curv_*/` (curvature experiment transient logs) + `https:/` (malformed URL prefix). 当前 `*.log` 已覆盖大多数 transient 内容, 但加显式 rule 让 reviewer 一眼看懂. 0 GPU, ~3 min.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #130 final state closure 闭环后, 发现 4 个 orphan dirs:
- `https:/github.com/zar123123` — 0 文件 (用户 git clone 失败残留)
- `log/` (singular) — 2 .log 文件 (HGN + SASRec 基线实验日志, 已被 `*.log` gitignore)
- `logs_curv_*/` — 各 1 .log 文件 (Task #88 per-layer curvature 实验 transient 日志, 已被 `*.log` gitignore)
- `results/` — 已被 `results/` gitignore (Task #119 删过)

`log/` 和 `logs_curv_*/` 的 .log 文件已被 `*.log` gitignore, 但 dir 本身没显式 rule. `https:/` 是 0 文件的 junk dir.

**Task #131 = clean orphan dirs + explicit gitignore hygiene**

---

## 2. 实验设计 (writeup only)

### 2.1 删除空 orphan `https:/` dir

```bash
rm -rf /home/wlia0047/ar57/wenyu/GeneRec/https:
```

确认前先 verify 0 files (R11.4 dry-run for irreversible):
```bash
find /home/wlia0047/ar57/wenyu/GeneRec/https: -type f | wc -l  # must be 0
```

### 2.2 .gitignore 加显式 rules

Append:
```gitignore
# Empty orphan dir (malformed URL prefix, git clone failure residue)
https:/

# Singular 'log/' (baseline experiment transient logs, HGN/SASRec)
log/

# Per-curvature transient experiment logs (Task #88)
logs_curv_*/

# (results/ already covered, line 228)
```

### 2.3 验证 dispatcher + git status

- `git status --short` 应该仍只显示 configs/ + src/ + tools/ (3 个 GRID 上游 dirs)
- `python3 scripts/all_audits.py` 7/7 PASS

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| `https:/` 删除前 verify 0 files | ✅ 闭环 |
| .gitignore append 3 rules | ✅ 闭环 |
| git status 仍只 3 个 untracked (configs/src/tools) | ✅ 闭环 |
| dispatcher 7/7 PASS | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否删除 https:/ | ✅ 删除 (0 文件, 纯 junk, 用户 clone 失败残留) | ❌ 不删 (但永久留垃圾) |
| 是否显式 gitignore log/ (singular) | ✅ 加 (虽然 *.log 已覆盖, 显式更清晰) | ❌ 不加 (但 reviewer 困惑为什么有 log/ dir) |
| 是否显式 gitignore logs_curv_*/ | ✅ 加 (per-curvature pattern, reviewer 一眼看懂) | ❌ 不加 |
| 是否 gitignore results/ | ❌ 已存在 (line 228) | n/a |
| 是否保留 log/, logs_curv_*/ 内容 | ✅ 保留 (transient logs, 不需要删) | ❌ 全删 (但 R2 no fallback, 内容存在但 gitignored 即可) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| rm -rf https: | ~1 sec |
| .gitignore append 3 rules | ~1 min |
| dispatcher verify | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~3 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: 误删 https:/ 中重要文件
  → **缓解**: 删除前 `find https: -type f | wc -l` 必须 = 0 (dry-run per R11.4)

**风险 2**: .gitignore rule 写错, 覆盖重要 tracked files
  → **缓解**: 用具体名字 `https:/` + `log/` + `logs_curv_*/`, 不使用通配

---

## 7. 完成度跟踪

- [x] 写 descriptions/task131_orphan_dir_cleanup.md (本文件)
- [ ] find https: -type f 验证 0 files
- [ ] rm -rf https:
- [ ] .gitignore append 3 rules
- [ ] git status verify
- [ ] dispatcher 7/7 PASS
- [ ] git commit + §16

---

## 8. 关联

- 前置: Task #130 final state closure (project 真正 closure)
- 后置: 真正 cleanup done. No more housekeeping

---

**核心交付**: 删除空 orphan `https:/` + .gitignore 加显式 rules (log/ + logs_curv_*/ + https:/). git status 保持只 3 个 GRID 上游 untracked dirs. dispatcher 7/7 PASS 不破坏.
