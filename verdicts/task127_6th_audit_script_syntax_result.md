# Task #127 — TASKS_INDEX+CHANGELOG drift v3 + 6th audit (script syntax)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: (a) `scripts/task127_script_syntax_audit.py` 创建, 用 `ast.parse()` 验证所有 `scripts/*.py` (320 scripts baseline 0 errors); (b) `scripts/all_audits.py` dispatcher 升级 5/5 → 6/6 PASS; (c) TASKS_INDEX.md Recent closed tasks 标题 (Task #101 - Task #125) → (Task #101 - Task #127) + #126/#127 2 行; coverage matrix + #126 行; (d) CHANGELOG.md [Unreleased] Notes 段追加 #126+#127 housekeeping 续集; (e) Verified 行 dispatcher 5/5 → 6/6.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| `scripts/task127_script_syntax_audit.py` 创建 | `ls scripts/task127_script_syntax_audit.py` | ✅ |
| audit script 通过 py_compile | `python3 -m py_compile scripts/task127_script_syntax_audit.py` | ✅ |
| 320 scripts/*.py 0 syntax errors | `python3 scripts/task127_script_syntax_audit.py` | ✅ 320 scripts, 0 errors |
| dispatcher 6/6 PASS | `python3 scripts/all_audits.py` | ✅ |
| TASKS_INDEX 加 #126 + 标题扩 #127 | `grep "Task #101 - Task #127" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX 2 行新加 | `grep -cE "^\| #(126|127) \|" TASKS_INDEX.md` | ✅ 2 |
| CHANGELOG Notes 段追加 | `grep "#126+#127" CHANGELOG.md` | ✅ |
| CHANGELOG Verified 5/5 → 6/6 | `grep "6/6 PASS" CHANGELOG.md` | ✅ |
| audit script header 注释更新 (4 → 6) | `grep "all 6 paper defense" scripts/all_audits.py` | ✅ |

---

## 2. 6th audit 设计

### scripts/task127_script_syntax_audit.py

- 用 `ast.parse()` 遍历所有 `scripts/*.py`
- 不用 `py_compile` (写盘污染 repo)
- 不用 import (避免环境依赖)
- Exit 0 iff 0 errors
- Baseline: 320 scripts (含 task127 自身), 0 errors

### 为什么需要这个 audit?

| 风险 | 现有 dispatcher 是否 catch | task127 是否 catch |
|------|---------|---------|
| Python script syntax error | ❌ (5 个 audit 都不验 scripts/*.py) | ✅ |
| IndentationError / SyntaxError | ❌ | ✅ |
| Future reviewer 改 script 时引入 typo | ❌ | ✅ |

### Dispatcher 升级

`scripts/all_audits.py` AUDITS 列表:
```python
AUDITS: list[tuple[str, str]] = [
    ("task101", "python3 scripts/task101_verify_env.py --skip-data --skip-packages"),
    ("task103", "python3 scripts/task103_paper_claims_audit.py"),
    ("task105", "python3 scripts/task105_ckpt_integrity.py --ci-mode"),
    ("task106", "python3 scripts/task106_audits.py"),
    ("task114", "python3 scripts/task114_verdict_integrity.py"),
    ("task127", "python3 scripts/task127_script_syntax_audit.py"),  # NEW
]
```

dispatcher docstring header: "all 4 paper defense" → "all 6 paper defense"; "Equivalent to running each in sequence" 加 task127 行.

CI workflow `.github/workflows/audits.yml` **不需要改** — Task #125 已用 dispatcher pattern, future audits 自动 CI 覆盖.

---

## 3. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否加 6th audit | ✅ 加 (320 scripts 0 errors baseline, 自动 catch future regressions) | ❌ 不加 (但 reviewer 需手动 py_compile) |
| audit name | ✅ task127 (跟 task ID 一致, dispatcher order 跟 task ID 对齐) | ❌ task128+ (但 AUDITS 顺序应跟 ID 对齐) |
| ast.parse vs py_compile | ✅ ast.parse (in-memory, 不污染 repo) | ❌ py_compile (写盘 .pyc) |
| ast.parse vs import | ✅ ast.parse (不需环境依赖) | ❌ import (依赖 conda env + repo deps) |
| TASKS_INDEX 补 #126 | ✅ 补 + 标题扩 #127 | ❌ 只补 #127 (drift v2 状态丢失) |

---

## 4. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| audit script compiles | `python3 -m py_compile scripts/task127_script_syntax_audit.py` | ✅ |
| audit script runs | `python3 scripts/task127_script_syntax_audit.py` | ✅ 320/320 |
| dispatcher 6/6 PASS | `python3 scripts/all_audits.py` | ✅ 6/6 |
| dispatcher 6th entry visible | `grep "task127" scripts/all_audits.py` | ✅ |
| TASKS_INDEX 标题扩 #127 | `grep "Task #101 - Task #127" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX #126/#127 行 | `grep -cE "^\| #(126|127) \|" TASKS_INDEX.md` | ✅ 2 |
| CHANGELOG Notes #126+#127 | `grep "#126+#127" CHANGELOG.md` | ✅ |
| CHANGELOG Verified 6/6 | `grep "6/6 PASS" CHANGELOG.md` | ✅ |

---

## 5. 关联

- 前置: Task #126 (TASKS_INDEX + CHANGELOG docs drift v2) + baseline scripts/*.py 0 errors
- 后置: 无 (drift v3 闭环 + 6th audit 加 dispatcher 自动化)

---

result: Task #127 — TASKS_INDEX+CHANGELOG drift v3 + 6th audit (script syntax check via ast.parse) 闭环. dispatcher 5/5 → 6/6 PASS (320 scripts baseline 0 errors). reviewer-facing docs cross-validation 闭合到 #127.
