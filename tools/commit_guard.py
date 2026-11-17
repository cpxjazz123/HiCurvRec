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

    if not large:
        print(f"commit_guard: ok ({len(paths)} changed path(s), none >= {args.max_mb:g} MiB)")
        return 0

    print(
        f"commit_guard: {len(large)} changed path(s) >= {args.max_mb:g} MiB; "
        "do NOT call commit.draft() until these are ignored, removed from the worktree, "
        "or explicitly allowed:",
        file=sys.stderr,
    )
    for size, relative in large:
        print(f"  {size / MIB:9.2f} MiB  {relative}", file=sys.stderr)
    print(
        "Move generated data under an ignored results/research path, or pass --allow GLOB "
        "only for deliberate evidence.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
