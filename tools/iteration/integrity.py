"""Infra checkpoint: hash the protected trees and report drift.

Only additions/removals/changes are reported; generated outputs (``outputs``,
``logs``, caches) are excluded by the caller. No file is ever rewritten except
the snapshot file written by :func:`write`.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.iteration.depgraph import file_sha

REPO = Path(__file__).resolve().parents[2]
SCHEMA = 'genrec.issue259.infra_snapshot/1'
PROTECTED = ('tools/iteration', 'tools/research/stage2', 'tools/research/verification',
             'tools/graph_infra', 'tools/issue252', 'tools/issue253',
             'assets/manifest.json', 'stage3_T5Train/train_HG-Rec.py',
             'CLAUDE.md', '.agent/skills/complete-open-issue/SKILL.md')
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.git', '.mypy_cache', '.ruff_cache'}


def _rel(path, repo):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def snapshot(paths, *, repo=REPO, exclude=()):
    repo = Path(repo).resolve()
    excluded = {str(item).strip('/') for item in exclude}
    files, missing = {}, []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = repo / path
        path = path.resolve()
        if path.is_file():
            files[_rel(path, repo)] = file_sha(path)
        elif path.is_dir():
            for child in sorted(path.rglob('*')):
                if not child.is_file():
                    continue
                rel = _rel(child, repo)
                parts = set(Path(rel).parts)
                if parts & SKIP_PARTS:
                    continue
                if any(item in parts for item in excluded):
                    continue
                if any(rel == item or rel.startswith(item + '/') for item in excluded):
                    continue
                files[rel] = file_sha(child)
        else:
            missing.append(_rel(path, repo))
    return {'schema': SCHEMA, 'files': dict(sorted(files.items())), 'missing': sorted(missing)}


def compare(before, after):
    before_files = (before or {}).get('files') or {}
    after_files = (after or {}).get('files') or {}
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    changed = sorted(key for key in set(before_files) & set(after_files)
                     if before_files[key] != after_files[key])
    return {'added': added, 'removed': removed, 'changed': changed,
            'ok': not (added or removed or changed)}


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
