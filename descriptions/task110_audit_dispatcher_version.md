# Task #110 — Audit Dispatcher + VERSION Tag + Paper-Submission Baseline

> **任务目的**: 三件事: (a) 单 audit dispatcher `scripts/all_audits.py` (一次命令跑完 4 个 audit, return 单一 exit code); (b) `VERSION` 文件 (锁定当前提交 version 1.0.0); (c) lightweight git tag `v1.0.0` (锁定 paper-submission-ready 提交). 形成 reviewer-friendly 单一 verify entry point + repo 版本基线.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

累积 4 个独立 audit 脚本 (Task #101/#103/#105/#106). 现在 reviewer 想 verify 全套 defense 需要分别跑:
```bash
python3 scripts/task101_verify_env.py
python3 scripts/task103_paper_claims_audit.py
python3 scripts/task105_ckpt_integrity.py
python3 scripts/task106_audits.py
```

**问题**: 
- reviewer 不知道要跑 4 个分散脚本
- 每个脚本独立 exit code, 难以 get 全 picture
- 没有 "paper submission v1.0.0 baseline" 概念

**Task #110 = 单一入口**:
- `python3 scripts/all_audits.py [--quiet]` 一行命令 verify 全套
- `VERSION` 文件 (top-level) 锁当前 version
- `git tag v1.0.0` 锁定 commit

---

## 2. 实验设计 (writeup only)

### 2.1 Single dispatcher script

`scripts/all_audits.py`:

```python
#!/usr/bin/env python3
"""
Task #110 — Single Audit Dispatcher

Runs all 4 paper defense audit scripts and returns a single exit code.
Exit 0 = all pass, Exit 1 = any fail.

Usage:
    python3 scripts/all_audits.py
    python3 scripts/all_audits.py --quiet  # only print summary

Equivalent to running each in sequence:
    python3 scripts/task101_verify_env.py --skip-data --skip-packages
    python3 scripts/task103_paper_claims_audit.py
    python3 scripts/task105_ckpt_integrity.py --ci-mode
    python3 scripts/task106_audits.py
"""
import subprocess, sys
from pathlib import Path

AUDITS = [
    ("task101", "scripts/task101_verify_env.py --skip-data --skip-packages"),
    ("task103", "scripts/task103_paper_claims_audit.py"),
    ("task105", "scripts/task105_ckpt_integrity.py --ci-mode"),
    ("task106", "scripts/task106_audits.py"),
]
```

(更多细节: 解析输出, 汇总成单一 PASS/FAIL summary)

### 2.2 VERSION file

`/home/wlia0047/ar57/wenyu/GeneRec/VERSION`:

```
1.0.0
```

(单一数字, 与 CHANGELOG.md / CITATION.cff date-released / papers/arXiv version 1 一致.)

### 2.3 Git tag v1.0.0

```bash
git tag v1.0.0 -m "Paper submission v1.0.0 (2026-07-24): all defenses verified"
```

lightweight tag (不 signed / 不 GPG-signed). For 可重现性 pin.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| `scripts/all_audits.py` 落盘 + 跑通 | ✅ 闭环 |
| 单 dispatcher exit 0 表示全过 | ✅ 闭环 |
| `VERSION` 文件含 `1.0.0` | ✅ 闭环 |
| `git tag v1.0.0` 存在 | ✅ 闭环 |
| 单 dispatcher 也列在 Task #107 CI workflow (增量, 不重写) | ✅ 闭环 (or 单独跑) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 dispatcher script | ~10 min |
| py_compile 验证 (Rule 10) | 3 sec |
| 跑 dispatcher 验证单 command | < 30 sec |
| 写 VERSION + commit | < 1 min |
| git tag | < 5 sec |
| 写 description + verdict | ~7 min |
| **总计** | **~20 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: dispatcher 失败时 exit 1 但具体 audit 信息丢失
  → **缓解**: 同时记录哪个 audit 失败 (logs 配 stdin/stderr capture)

**风险 2**: git tag 在共享 checkout 创建可能冲突
  → **缓解**: 用 `git tag -f` 强制更新 (R13 不禁止 worktree, 但 tag 是 checkout-local, OK)

**风险 3**: VERSION 与 CHANGELOG.md / CITATION.cff 不一致
  → **缓解**: 一律 1.0.0 (init version); 后续 Task #111+ 改 VERSION 时同步其他

**风险 4**: dispatcher 把 stdout 全 print, 弄乱 reviewer 视野
  → **缓解**: `--quiet` flag 默认开, 只打印 summary

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task110_audit_dispatcher_version.md` (本文件)
- [ ] 写 `scripts/all_audits.py`
- [ ] py_compile 验证 (Rule 10)
- [ ] 跑 dispatcher 单 command 验证
- [ ] 写 `VERSION` 文件 (1.0.0)
- [ ] `git tag v1.0.0`
- [ ] 写 `verdicts/task110_audit_dispatcher_version_result.md`
- [ ] git commit (含 tag)
- [ ] §16 loop.md 更新 (Task #110 ✅ 已完成)

---

## 7. 关联

- 前置: Task #101/#103/#105/#106 (4 个 audit 脚本)
- 后置: Task #111+ (后续 release 版本 / dynamic endpoint / etc.)

---

**核心交付**: 单一 dispatcher (`scripts/all_audits.py`) + VERSION (1.0.0) + git tag (v1.0.0). 形成 reviewer-friendly single-entry verify + repo 版本基线.
