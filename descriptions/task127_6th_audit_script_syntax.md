# Task #127 — TASKS_INDEX+CHANGELOG drift v3 + 6th audit (script syntax)

> **任务目的**: (1) 修复 post-#126 出现的 docs drift v3: TASKS_INDEX.md Recent closed tasks 表加 #126 行 + 标题扩到 "Task #101 - Task #126"; CHANGELOG [Unreleased] Notes 追加 #126. (2) Add 6th audit to dispatcher: `scripts/task127_script_syntax_audit.py` 用 `ast.parse()` 验证所有 `scripts/*.py` 通过 syntax 检查 (319 scripts 0 errors 当前 baseline). dispatcher 升级 5/6 → 6/6 PASS.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #126 TASKS_INDEX + CHANGELOG docs drift v2 闭环时只补到 #125, #126 自身没加进 TASKS_INDEX/CHANGELOG. 跟 Task #122-#125 重复同样的 pattern (drift 在新 task 闭环时立即出现). 

同时发现: 319 个 `scripts/*.py` 当前都过 syntax 检查 (baseline 0 errors), 但这个 check 没被 dispatcher 自动化 — 依赖 reviewer 手动跑 `python3 -m py_compile` 或 `ast.parse`. 加 6th audit 让 dispatcher 强制执行, future script 改动会被自动 catch.

**Task #127 = drift v3 fix + 6th audit (additive value)**

---

## 2. 实验设计 (writeup only)

### 2.1 TASKS_INDEX.md drift v3 fix

#### 2.1.1 Recent closed tasks 标题扩
`(Task #101 - Task #125)` → `(Task #101 - Task #126)`

#### 2.1.2 Recent closed tasks 表加 #126 行
```
| #126 | TASKS_INDEX + CHANGELOG docs drift v2 (3 places synced) | ✅ | `verdicts/task126_docs_drift_v2_result.md` |
```

#### 2.1.3 Coverage matrix 加 #126 行
```
| Task #126 | ✓ closed | TASKS_INDEX + CHANGELOG docs drift v2 |
```

### 2.2 CHANGELOG [Unreleased] drift v3 fix

Append to Notes section:
```
- Continued housekeeping: #126 TASKS_INDEX + CHANGELOG docs drift v2 +
  #127 drift v3 + 6th audit (script syntax check).
```

### 2.3 6th audit — `scripts/task127_script_syntax_audit.py`

创建脚本: 用 `ast.parse()` 对所有 `scripts/*.py` 跑 syntax check. PASS iff 0 errors.

设计:
- 遍历 `scripts/*.py` (319 files baseline)
- 每个 file 用 `ast.parse()` 验证
- 输出: total scripts / syntax errors / list of failing files
- Exit 0 iff 0 errors

### 2.4 dispatcher 升级 5/6 → 6/6

`scripts/all_audits.py` 添加 task127 到 AUDITS 列表:
```python
("task127", "scripts/task127_script_syntax_audit.py"),
```

### 2.5 CI workflow 同步 (auto via dispatcher pattern)

`.github/workflows/audits.yml` 不需要改 — Task #125 已用 dispatcher pattern, future audits 自动 CI 覆盖.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| TASKS_INDEX 加 #126 行 + 标题 | ✅ 闭环 |
| CHANGELOG Notes 追加 #126+#127 | ✅ 闭环 |
| task127_script_syntax_audit.py 创建 | ✅ 闭环 |
| dispatcher 5/5 → 6/6 PASS | ✅ 闭环 |
| scripts/*.py 0 errors baseline | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否加 6th audit (script syntax) | ✅ 加 (319 scripts baseline 0 errors, 自动 catch future regressions) | ❌ 不加 (但 reviewer 需手动跑 py_compile) |
| audit name | ✅ task127 (跟 task ID 一致) | ❌ task128+ (但 dispatcher order 应跟 task ID 对齐) |
| audit 内容 | ✅ ast.parse() + 列表 + exit code | ❌ py_compile (额外写盘开销) |
| TASKS_INDEX 补 #126 | ✅ 补 (cross-validation) | ❌ 不补 (drift 持续扩大) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 task127 audit script (~30 行) | ~3 min |
| Edit TASKS_INDEX + CHANGELOG | ~2 min |
| dispatcher 升级 + verify | ~1 min |
| git commit + §16 | ~2 min |
| **总计** | **~8 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: 新加 audit 失败 (319 scripts 中某个 import 失败)
  → **缓解**: baseline 已确认 0 errors; audit 只做 syntax check 不做 import check, 不会因为缺依赖 fail

**风险 2**: TASKS_INDEX 表格加 #126 后与 CHANGELOG Notes 数字不一致
  → **缓解**: 两个文档用同一 git source of truth (`git log --oneline`)

---

## 7. 完成度跟踪

- [x] 写 descriptions/task127_6th_audit_script_syntax.md (本文件)
- [ ] 写 scripts/task127_script_syntax_audit.py
- [ ] TASKS_INDEX Recent closed tasks 加 #126
- [ ] TASKS_INDEX Coverage matrix 加 #126
- [ ] CHANGELOG [Unreleased] Notes 追加
- [ ] scripts/all_audits.py 加 task127 entry
- [ ] dispatcher 6/6 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #126 (TASKS_INDEX + CHANGELOG docs drift v2) + baseline scripts/*.py 0 errors
- 后置: 无 (drift v3 闭环 + 6th audit 加 dispatcher)

---

**核心交付**: TASKS_INDEX + CHANGELOG drift v3 闭环 (#126 自加) + 6th audit (script syntax check). dispatcher 升级 5/6 → 6/6 PASS. 319 scripts 0 errors baseline 锁定.
