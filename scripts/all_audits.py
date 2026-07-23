#!/usr/bin/env python3
"""
Task #110 — Single Audit Dispatcher

Runs all 6 paper defense audit scripts in sequence and returns a single
exit code (0 = all pass, 1 = any fail). Provides reviewer-friendly
single-entry point for "verify all paper defenses".

Usage:
    python3 scripts/all_audits.py           # default: summary mode
    python3 scripts/all_audits.py --verbose # show each audit's stdout
    python3 scripts/all_audits.py --quiet   # only PASS/FAIL summary

Equivalent to running each in sequence:
    python3 scripts/task101_verify_env.py --skip-data --skip-packages
    python3 scripts/task103_paper_claims_audit.py
    python3 scripts/task105_ckpt_integrity.py --ci-mode
    python3 scripts/task106_audits.py
    python3 scripts/task114_verdict_integrity.py
    python3 scripts/task127_script_syntax_audit.py

No network. No fallback (R2). Stdlib only.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SCRIPTS = REPO / "scripts"

# Tuple of (audit name, command string). All scripts are invoked via
# `python3 <path>` because the audit scripts are not chmod +x.
AUDITS: list[tuple[str, str]] = [
    ("task101", "python3 scripts/task101_verify_env.py --skip-data --skip-packages"),
    ("task103", "python3 scripts/task103_paper_claims_audit.py"),
    ("task105", "python3 scripts/task105_ckpt_integrity.py --ci-mode"),
    ("task106", "python3 scripts/task106_audits.py"),
    ("task114", "python3 scripts/task114_verdict_integrity.py"),
    ("task127", "python3 scripts/task127_script_syntax_audit.py"),
]


def run_audit(name: str, cmd: str, verbose: bool) -> tuple[bool, float, str]:
    """Run a single audit script via subprocess.run.

    Returns:
        (passed, elapsed_sec, combined_output)
    """
    start = time.time()
    try:
        result = subprocess.run(
            cmd.split(),
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min per audit
        )
        elapsed = time.time() - start
        passed = result.returncode == 0
        output = (result.stdout or "") + (result.stderr or "")
        if verbose:
            print(f"\n--- {name} {'✅' if passed else '❌'} ({elapsed:.2f}s) ---")
            print(output)
        return passed, elapsed, output
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, elapsed, f"Timeout (>120s) running {name}"
    except FileNotFoundError as e:
        elapsed = time.time() - start
        return False, elapsed, f"Script not found: {e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print each audit's stdout/stderr")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print final PASS/FAIL summary")
    args = parser.parse_args()

    if args.verbose and args.quiet:
        print("Cannot pass both --verbose and --quiet")
        return 1

    print("=== Task #110 — Paper Defense Audit Dispatcher ===")
    print(f"Running {len(AUDITS)} audits in sequence\n")

    results: list[tuple[str, bool, float]] = []
    for name, cmd in AUDITS:
        passed, elapsed, _ = run_audit(name, cmd, args.verbose)
        results.append((name, passed, elapsed))
        if not args.quiet:
            verdict = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name}: {verdict} ({elapsed:.2f}s)")

    print("\n=== Audit Summary ===")
    n_passed = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"  Passed: {n_passed}/{n_total}")
    for name, p, e in results:
        print(f"    {'✅' if p else '❌'} {name} ({e:.2f}s)")

    if n_passed == n_total:
        print("\n🎉 All paper defenses verified at Task #110 dispatcher.")
        return 0
    else:
        failed = [n for n, p, _ in results if not p]
        print(f"\n❌ {len(failed)} audit(s) failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
