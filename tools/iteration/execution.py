"""Durable execution of one validation command.

Every command writes its metadata *before* the process starts and its combined
stdout/stderr to a persisted log, then records the exit code, end time and
duration. A timeout is ``INCOMPLETE`` and is never mistaken for a pass.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.iteration.depgraph import file_sha

SCHEMA = 'genrec.pipeline.execution/1'


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    unit_id: str
    status: str
    exit_code: object
    command: str
    cwd: str
    log: str
    record: str
    started_at: str
    ended_at: str
    duration_s: float
    note: str = ''

    def evidence(self):
        log = Path(self.log)
        return {'command': self.command, 'exit_code': self.exit_code,
                'log': str(log), 'output': str(log),
                'output_sha256': file_sha(log),
                'duration_s': self.duration_s, 'note': self.note}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe(unit_id):
    return str(unit_id).replace('/', '__').replace('..', '__')


def _write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def run(unit_id, argv, *, out_dir, cwd=None, env=None, timeout=None, note=''):
    if isinstance(argv, str):
        argv = shlex.split(argv)
    argv = [str(part) for part in argv]
    if not argv:
        raise ExecutionError('empty command')
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ExecutionError('out_dir is not a directory: %s' % out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record = out_dir / 'checks' / _safe(unit_id) / ('%s_%d_%s' % (stamp, os.getpid(), uuid.uuid4().hex[:8]))
        record.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ExecutionError('cannot prepare the execution directory: %s' % exc)
    log = record / 'run.log'
    command = shlex.join(argv)
    cwd = str(Path(cwd).resolve()) if cwd is not None else os.getcwd()
    metadata = {'schema': SCHEMA, 'unit_id': str(unit_id), 'command': command,
                'argv': argv, 'cwd': cwd, 'status': 'RUNNING', 'exit_code': None,
                'started_at': _now(), 'ended_at': None, 'duration_s': None,
                'log': str(log), 'note': str(note),
                'versions': {'python': platform.python_version(), 'platform': platform.platform(),
                             'interpreter': sys.executable}}
    _write_json(record / 'command.json', metadata)
    started = time.monotonic()
    status = 'FAIL'
    exit_code = None
    with log.open('w', encoding='utf-8') as handle:
        handle.write('# command: %s\n# cwd: %s\n# started: %s\n' % (command, cwd, metadata['started_at']))
        handle.flush()
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            handle.write('# spawn error: %s\n' % exc)
            status = 'INCOMPLETE'
        else:
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                handle.write('# timeout after %ss\n' % timeout)
                status = 'INCOMPLETE'
            else:
                status = 'PASS' if exit_code == 0 else 'FAIL'
    ended = time.monotonic()
    metadata.update({'status': status, 'exit_code': exit_code, 'ended_at': _now(),
                     'duration_s': round(ended - started, 3)})
    _write_json(record / 'command.json', metadata)
    return ExecutionResult(unit_id=str(unit_id), status=status, exit_code=exit_code,
                           command=command, cwd=cwd, log=str(log), record=str(record),
                           started_at=metadata['started_at'], ended_at=metadata['ended_at'],
                           duration_s=metadata['duration_s'], note=str(note))


def run_and_record(cache, spec, argv, *, out_dir, repo=None, cwd=None, env=None,
                   timeout=None, note=''):
    """Run one command and persist a receipt bound to the spec fingerprint."""
    from tools.iteration import depgraph
    repo = Path(repo) if repo is not None else depgraph.REPO
    fingerprint = spec.fingerprint(repo=repo)
    result = run(spec.test_id, argv, out_dir=out_dir, cwd=cwd, env=env,
                 timeout=timeout, note=note)
    cache.record(spec.test_id, fingerprint['digest'], result.status,
                 evidence=result.evidence(), declared=spec.declared(),
                 fingerprint=fingerprint)
    return result
