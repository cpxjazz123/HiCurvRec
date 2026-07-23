# Task #114 — Verdict Integrity Audit (5th Audit for Dispatcher)

> **任务目的**: 写 `scripts/task114_verdict_integrity.py` 审计所有 232 verdict 文件的格式合规性 — 检查 (a) 每个 verdict 含 `result:` 行 (R8 + loop.md §9.3 强制); (b) verdict 文件名 task ID 与 descriptions/ task ID 一致; (c) 无 orphan verdict (verdict 不存在对应 description); (d) R9 contiguous (descriptions/ 1-N 无空洞). 然后扩展 `scripts/all_audits.py` 加入这第 5 个 audit, 形成 5/5 PASS dispatcher.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #110 创建了 `scripts/all_audits.py` (4 audit dispatcher). Task #112 把它纳入 CHANGELOG [1.0.0] 段 + Verified 段. 但 dispatcher 当前只审计 4 个 paper-defense 类 (env / paper claims / ckpt / paper audits), 缺少 **项目元数据完整性** 审计.

具体缺口:
- 没有脚本验证所有 verdict 文件含 `result:` 行 (R8 + §9.3 强制)
- 没有脚本验证 verdict task ID 与 description task ID 配对
- 没有脚本验证 R9 contiguous (descriptions/ 1-N 无空洞)

历史上:
- Task #112 verdict 验证 "R9 contiguous 1-112 无空洞" 用临时 bash 脚本, 没落盘
- Task #113 verdict 验证 "R9 1-112" 用临时命令, 没落盘
- 每次需要手写 `ls descriptions/ | grep -oE 'task[0-9]+' | ... | uniq | wc -l`, 重复劳动

**Task #114 = 元数据完整性审计落盘**:
- 写 `scripts/task114_verdict_integrity.py` (audit 5 类检查)
- 扩展 `scripts/all_audits.py` 加入这第 5 audit
- dispatcher 升级到 5/5 PASS
- CI workflow (Task #107) 自动包含新 audit

---

## 2. 实验设计 (writeup only)

### 2.1 scripts/task114_verdict_integrity.py 设计

```python
AUDITS = [
    ("verdict_result_lines", ...),       # 全部 verdict 含 `result:` 行
    ("task_id_pairing", ...),            # verdict task ID ∈ description task ID
    ("r9_descriptions_contiguous", ...), # descriptions/ 1-max 无空洞
    ("r9_verdicts_contiguous", ...),     # verdicts/ 1-max 无空洞 (best-effort)
    ("dispatcher_self_check", ...),      # all_audits.py 含本 audit
]

def check_verdict_result_lines():
    """每个 *_result.md verdict 含独立行的 `result:` 开头"""
    
def check_task_id_pairing():
    """每个 verdict 文件的 task ID 在 descriptions/ 中存在对应 description"""
    
def check_r9_descriptions_contiguous():
    """descriptions/ task1, task2, ..., taskN 无空洞 (R9 mandate)"""
    
def check_r9_verdicts_contiguous():
    """verdicts/ verdict task IDs 1..max contiguous (best-effort, 允许 gaps for aux tasks)"""
    
def check_dispatcher_self_check():
    """all_audits.py AUDITS list 含 task114"""
```

### 2.2 scripts/all_audits.py 扩展

```diff
 AUDITS: list[tuple[str, str]] = [
     ("task101", "python3 scripts/task101_verify_env.py --skip-data --skip-packages"),
     ("task103", "python3 scripts/task103_paper_claims_audit.py"),
     ("task105", "python3 scripts/task105_ckpt_integrity.py --ci-mode"),
     ("task106", "python3 scripts/task106_audits.py"),
+    ("task114", "python3 scripts/task114_verdict_integrity.py"),
 ]
```

### 2.3 预期输出

```
=== Task #114 — Verdict Integrity Audit ===
Running 5 sub-audits in sequence

  A1 verdict_result_lines:        ✅ PASS (232/232 verdicts have `result:` line)
  A2 task_id_pairing:             ✅ PASS (232/232 verdicts have matching description)
  A3 r9_descriptions_contiguous:  ✅ PASS (1-112 contiguous, no gaps)
  A4 r9_verdicts_contiguous:      ✅ PASS (1-111 contiguous, gaps #113-#115 are aux)
  A5 dispatcher_self_check:       ✅ PASS (all_audits.py contains task114)

Verdict: ✅ 5/5 PASS
```

exit 0 = all pass, exit 1 = any fail.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| `scripts/task114_verdict_integrity.py` 落盘 + py_compile | ✅ 闭环 |
| 5 sub-audit 全部 PASS | ✅ 闭环 |
| `scripts/all_audits.py` 加入 task114 | ✅ 闭环 |
| dispatcher 跑出 5/5 PASS | ✅ 闭环 |
| CI workflow (Task #107) 自动含新 audit | ✅ 闭环 (无需改 yaml, dispatcher 自动调用) |
| README 7 badges 不动 | ✅ 闭环 (新 audit 不进 badge, 是 meta-audit) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 `scripts/task114_verdict_integrity.py` (~150 行 stdlib-only) | ~10 min |
| py_compile 验证 (Rule 10) | 3 sec |
| 单 audit 跑通 | ~2 sec |
| 扩展 `scripts/all_audits.py` (1 行) | ~30 sec |
| dispatcher 跑出 5/5 PASS | ~3 sec |
| 写 verdict + commit + §16 | ~5 min |
| **总计** | **~18 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: 5th audit 发现真实问题 (R9 漏洞 / verdict 缺 result:) → 修复成本高
  → **缓解**: 不预先修, 只 audit + 报告. 修复留给独立 task (例如 Task #115), 不污染 Task #114 scope

**风险 2**: CI workflow 不自动跑新 audit (因 yaml 写死 9 steps 不含 dispatcher)
  → **缓解**: Task #107 verdict 已记录 yaml 9 steps, dispatcher 是 reviewer-facing single entry, CI 不必调用 dispatcher. CI 9 steps 各自独立足够.

**风险 3**: 5th audit 与 4 audit 重复 (e.g., task103 内部已 check R9)
  → **缓解**: task103 只 check paper claims, 不 check verdict format. task114 是 meta-audit, 唯一职责.

**风险 4**: audit script 误报 (e.g., aux verdicts like task116_#134 没 description)
  → **缓解**: task_id_pairing 用 best-effort (allow aux verdicts), R9 仅检查 descriptions/ (不检查 verdicts/)

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task114_verdict_integrity_audit.md` (本文件)
- [ ] 写 `scripts/task114_verdict_integrity.py`
- [ ] py_compile 验证 (Rule 10)
- [ ] 单独跑 audit 验证 5/5 PASS
- [ ] 扩展 `scripts/all_audits.py` 加入 task114
- [ ] 跑 dispatcher 验证 5/5 PASS
- [ ] 写 `verdicts/task114_verdict_integrity_result.md`
- [ ] git commit
- [ ] loop.md §16 更新

---

## 7. 关联

- 前置: Task #110 (4 audit dispatcher) + Task #112 (release artifacts)
- 后置: Task #115+ (auditing 发现的问题修复, e.g., orphan verdicts)

---

**核心交付**: `scripts/task114_verdict_integrity.py` (5 sub-audit) + `scripts/all_audits.py` 扩展到 5/5 PASS. dispatcher 升级到 5 audit, 增加元数据完整性验证.