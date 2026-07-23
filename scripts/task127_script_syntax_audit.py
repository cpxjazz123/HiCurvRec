#!/usr/bin/env python3
"""Task #127 — Script syntax audit (6th paper-defense audit).

Walks every `scripts/*.py`, parses each with `ast.parse()`, and reports the
total + any syntax errors. Returns exit code 0 iff 0 errors. This is the
6th paper-defense audit added on top of task101/103/105/106/114.

Why ast.parse and not py_compile?
- py_compile writes .pyc to disk → pollutes repo state.
- ast.parse is in-memory and identical to compile() at the parser stage.
- We do NOT import the modules (would need full environment + repo
  dependencies); syntax check is sufficient for catching regressions in
  reviewer-facing scripts.

Baseline (2026-07-24, Task #127 closure): 319 scripts/*.py, 0 errors.

Usage:
    python3 scripts/task127_script_syntax_audit.py
"""
from __future__ import annotations

import ast
import glob
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    scripts = sorted(glob.glob(str(repo_root / "scripts" / "*.py")))

    if not scripts:
        print("⚠️  No scripts/*.py found — dispatcher invocations would always pass")
        return 0

    errors: list[tuple[str, str]] = []
    for f in scripts:
        try:
            ast.parse(open(f, encoding="utf-8").read(), filename=f)
        except SyntaxError as e:
            errors.append((f, f"line {e.lineno}: {e.msg}"))

    print(f"Scanned {len(scripts)} scripts/*.py files")
    if errors:
        print(f"❌ {len(errors)} syntax error(s) found:")
        for f, msg in errors:
            print(f"  {f}: {msg}")
        return 1
    print(f"✅ All {len(scripts)} scripts pass syntax check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
