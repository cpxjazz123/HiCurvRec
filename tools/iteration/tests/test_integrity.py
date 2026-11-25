"""Infra integrity snapshot: protected paths are hashed, compared, never rewritten."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import integrity
from tools.iteration.tests.support import write_tree


def test_snapshot_and_compare(tmp_path):
    write_tree(tmp_path, {'tools/a.py': 'A = 1\n', 'tools/sub/b.py': 'B = 2\n',
                          'tools/logs/out.txt': 'noise\n'})
    before = integrity.snapshot([tmp_path / 'tools'], repo=tmp_path, exclude=('logs',))
    names = set(before['files'])
    assert names == {'tools/a.py', 'tools/sub/b.py'}
    (tmp_path / 'tools/a.py').write_text('A = 9\n')
    (tmp_path / 'tools/sub/b.py').unlink()
    (tmp_path / 'tools/new.py').write_text('N = 3\n')
    after = integrity.snapshot([tmp_path / 'tools'], repo=tmp_path, exclude=('logs',))
    diff = integrity.compare(before, after)
    assert diff['changed'] == ['tools/a.py']
    assert diff['removed'] == ['tools/sub/b.py']
    assert diff['added'] == ['tools/new.py']
    assert diff['ok'] is False


def test_unchanged_snapshot_ok(tmp_path):
    write_tree(tmp_path, {'tools/a.py': 'A = 1\n'})
    before = integrity.snapshot([tmp_path / 'tools'], repo=tmp_path)
    after = integrity.snapshot([tmp_path / 'tools'], repo=tmp_path)
    assert integrity.compare(before, after) == {'added': [], 'removed': [], 'changed': [], 'ok': True}


def test_write_read_roundtrip(tmp_path):
    write_tree(tmp_path, {'tools/a.py': 'A = 1\n'})
    snap = integrity.snapshot([tmp_path / 'tools'], repo=tmp_path)
    path = integrity.write(tmp_path / 'INFRA.json', snap)
    assert integrity.read(path)['files'] == snap['files']


def test_missing_path_is_recorded_not_crashed(tmp_path):
    snap = integrity.snapshot([tmp_path / 'absent'], repo=tmp_path)
    assert snap['files'] == {}
    assert snap['missing'] == ['absent']
