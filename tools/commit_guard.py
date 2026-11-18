#!/usr/bin/env python3
"""Fail a commit preflight when Git would have to diff a large worktree file.

`commit.draft()` does not look only at the files a later `take()` selects: it first builds the
complete ``HEAD -> worktree`` diff, including every untracked file that is not ignored.  One
generated 300 MB table therefore makes *every* draft expensive even when the commit is three
Python files.  This check only stats paths from Git's cheap ``ls-files`` / ``name-only`` output;
it never reads file contents.

Usage
    python3 tools/commit_guard.py
    python3 tools/commit_guard.py --max-mb 2
    python3 tools/commit_guard.py --repo /path/to/repo

Exit status
    0  no changed path is at or above the limit
    1  at least one changed path is too large (paths are printed)
    2  not a Git worktree / Git is unavailable

A generated path should normally be covered by ``.gitignore``.  For a deliberately tracked large
evidence file, either remove it from the worktree diff before drafting or pass its exact path with
``--allow``; the allow list is intentionally explicit.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

MIB = 1024 * 1024

#: A draft builds a hunk model for every changed line and prints every changed path.
#: Past these sizes the *tooling* is the bottleneck (a 1.26 MB single-file SID table is
#: 147k lines), so the guard refuses before the draft is built rather than after it hangs.
MAX_PATHS = 150
MAX_LINES = 30_000


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _nul_paths(root: Path, *args: str) -> list[str]:
    result = _git(root, *args)
    if result.returncode != 0:
        return []
    return [
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def changed_paths(root: Path) -> list[str]:
    """Return worktree paths that a draft would scan, without reading file contents."""
    paths: list[str] = []
    paths.extend(_nul_paths(root, "ls-files", "--others", "--exclude-standard", "-z"))
    paths.extend(_nul_paths(root, "diff", "--name-only", "-z"))
    paths.extend(_nul_paths(root, "diff", "--cached", "--name-only", "-z"))
    return sorted(set(paths))


def draft_budget(root: Path, paths: list[str]) -> tuple[int, list[tuple[int, str]]]:
    """Changed lines a draft would model, plus the worst offenders.

    Tracked files count their added+deleted lines from `git diff --numstat`; untracked
    files count all their lines, which is what the draft shows.  Nothing above the byte
    limit is read (it is already reported by the size check).
    """
    total = 0
    offenders: list[tuple[int, str]] = []
    numstat = _git(root, "diff", "--numstat", "--cached", "HEAD")
    for line in numstat.stdout.decode("utf-8", "surrogateescape").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, name = parts
        count = (int(added) if added.isdigit() else 0) + (int(deleted) if deleted.isdigit() else 0)
        total += count
        offenders.append((count, name))
    untracked = _nul_paths(root, "ls-files", "--others", "--exclude-standard", "-z")
    for relative in untracked:
        path = root / relative
        try:
            size = path.lstat().st_size
        except OSError:
            continue
        if size >= 2 * MIB:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        count = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
        total += count
        offenders.append((count, relative))
    offenders.sort(reverse=True)
    return total, offenders


def _allowed(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="Git worktree to inspect (default: current directory)")
    parser.add_argument("--max-mb", type=float, default=2.0,
                        help="flag files at or above this many MiB (default: 2.0)")
    parser.add_argument("--allow", action="append", default=[], metavar="GLOB",
                        help="explicit path/glob allowed to exceed the limit; repeatable")
    parser.add_argument("--max-paths", type=int, default=MAX_PATHS,
                        help=f"refuse a draft over this many changed paths (default {MAX_PATHS})")
    parser.add_argument("--max-lines", type=int, default=MAX_LINES,
                        help=f"refuse a draft over this many changed lines (default {MAX_LINES})")
    args = parser.parse_args(argv)

    root = args.repo.resolve()
    probe = _git(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != b"true":
        print(f"commit_guard: not a Git worktree: {root}", file=sys.stderr)
        return 2
    if args.max_mb <= 0:
        print("commit_guard: --max-mb must be positive", file=sys.stderr)
        return 2

    threshold = int(args.max_mb * MIB)
    paths = changed_paths(root)
    large: list[tuple[int, str]] = []
    for relative in paths:
        if _allowed(relative, args.allow):
            continue
        try:
            size = (root / relative).lstat().st_size
        except OSError:
            continue
        if size >= threshold:
            large.append((size, relative))

    lines, offenders = draft_budget(root, paths)
    over_paths = len(paths) > args.max_paths
    over_lines = lines > args.max_lines
    if large or over_paths or over_lines:
        for size, relative in large[:10]:
            print(f"  {size / MIB:9.2f} MiB  {relative}", file=sys.stderr)
        if over_paths or over_lines:
            print(
                f"commit_guard: draft budget exceeded -- {len(paths)} path(s) "
                f"(limit {args.max_paths}), {lines} line(s) (limit {args.max_lines}).",
                file=sys.stderr,
            )
            for count, name in offenders[:8]:
                print(f"  {count:>9d} lines  {name}", file=sys.stderr)
            print(
                "A draft models every changed line, so split the commit or ignore the generated "
                "files.  Pretty-printed SID tables (147k lines each) and generated source trees "
                "belong in a digest/manifest, not in the diff.",
                file=sys.stderr,
            )
            print(
                "Override only deliberately: --max-paths N --max-lines N.",
                file=sys.stderr,
            )
        return 1

    print(f"commit_guard: ok ({len(paths)} changed path(s), {lines} line(s), "
          f"none >= {args.max_mb:g} MiB)")
    return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
