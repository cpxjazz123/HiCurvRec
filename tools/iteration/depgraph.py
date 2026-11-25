"""Dependency-aware validation: reuse a PASS only for the same real dependencies.

A receipt is bound to a *fingerprint*: the content hashes of the test's complete
import closure (code), the assets and configuration it reads, and the runtime
identity (interpreter, torch, visible devices).  A changed module anywhere in
that closure changes the fingerprint, so the old PASS cannot be reused; an
unrelated report edit does not.

Repo-local imports are followed through the source, not guessed from names, so a
module that starts importing a new file invalidates every test that reaches it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
from pathlib import Path

SCHEMA = 'genrec.pipeline.depgraph/1'
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_sha(path):
    return sha256_bytes(Path(path).read_bytes())


def _module_candidates(name, repo, roots=()):
    parts = name.split('.')
    stem = Path(*parts)
    for base in (*roots, repo, repo / 'tools', repo / 'stage2_RQ-VAE'):
        candidate = base / (str(stem) + '.py')
        if candidate.is_file():
            return candidate
        package = base / stem / '__init__.py'
        if package.is_file():
            return package
    return None


def _imports(path, repo, roots=()):
    try:
        tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
            names += [node.module + '.' + alias.name for alias in node.names if alias.name != '*']
    found = []
    for name in names:
        candidate = _module_candidates(name, repo, roots)
        if candidate is not None:
            found.append(candidate.resolve())
    rel = Path(path).resolve().parent
    source = Path(path).read_text(encoding='utf-8')
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.level and node.module is None:
            for alias in node.names:
                sibling = (rel / (alias.name + '.py'))
                if sibling.is_file():
                    found.append(sibling.resolve())
    return found


def module_closure(paths, repo=REPO, roots=()):
    """Every repo-local module the given files import, transitively."""
    seen = set()
    queue = [Path(p).resolve() for p in paths]
    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        queue += _imports(path, repo, roots)
    return sorted(seen)


def _key(path, repo):
    """Every dependency is named the same way in the fingerprint and in `changed`."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(repo).resolve()))
    except ValueError:
        return str(path)


def runtime_identity():
    import sys
    torch_version = 'absent'
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        pass
    devices = ''
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30)
        devices = ';'.join(sorted(set(line.strip() for line in out.stdout.splitlines() if line.strip())))
    except Exception:
        devices = 'unavailable'
    return {'python': platform.python_version(), 'implementation': sys.implementation.name,
            'torch': torch_version, 'devices': devices}


class Spec:
    """One validation unit and the real dependencies its result depends on."""

    def __init__(self, test_id, *, code=(), assets=(), configs=(), tests=(), runtime=True, roots=()):
        self.test_id = str(test_id)
        self.code = [Path(p) for p in code]
        self.assets = [Path(p) for p in assets]
        self.configs = [Path(p) for p in configs]
        self.tests = [Path(p) for p in tests]
        self.runtime = bool(runtime)
        self.roots = [Path(r) for r in roots]

    def declared(self):
        return {'code': [str(p) for p in self.code], 'assets': [str(p) for p in self.assets],
                'configs': [str(p) for p in self.configs], 'tests': [str(p) for p in self.tests]}

    def fingerprint(self, repo=REPO):
        closure = module_closure([*self.tests, *self.code, *self.configs], repo=repo, roots=self.roots)
        payload = {'schema': 'genrec.pipeline.dep_fingerprint/1',
                   'test_id': self.test_id,
                   'code': {_key(p, repo): file_sha(p) for p in closure if p.is_file()},
                   'assets': {_key(p, repo): file_sha(p) for p in self.assets if p.is_file()},
                   'configs': {_key(p, repo): file_sha(p) for p in self.configs if p.is_file()},
                   'tests': {_key(p, repo): file_sha(p) for p in self.tests if p.is_file()},
                   'runtime': runtime_identity() if self.runtime else None}
        payload['digest'] = sha256_bytes(json.dumps(payload, sort_keys=True).encode())
        return payload


class Cache:
    """PASS/FAIL receipts keyed by (test id, fingerprint digest)."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, test_id, digest):
        safe = test_id.replace('/', '__').replace(':', '_')
        return self.root / safe / (digest + '.json')

    def record(self, test_id, digest, status, *, evidence=None, declared=None):
        path = self._path(test_id, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema': 'genrec.pipeline.receipt/1', 'test_id': test_id,
                                    'digest': digest, 'status': status,
                                    'evidence': evidence or {}, 'declared': declared or {}},
                                   indent=2, sort_keys=True) + '\n')
        return path

    def lookup(self, test_id, digest):
        path = self._path(test_id, digest)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding='utf-8'))


def plan(specs, *, cache, changed=(), repo=REPO):
    """Split the validation set into reusable PASSes and required reruns."""
    normalised = set()
    for path in changed:
        resolved = Path(path).resolve()
        try:
            normalised.add(str(resolved.relative_to(Path(repo).resolve())))
        except ValueError:
            normalised.add(str(resolved))
    changed = normalised
    reuse, rerun = [], []
    for spec in specs:
        fingerprint = spec.fingerprint(repo=repo)
        closure = set(fingerprint['code']) | set(fingerprint['assets']) | \
            set(fingerprint['configs']) | set(fingerprint['tests'])
        touched = sorted(closure & changed)
        receipt = cache.lookup(spec.test_id, fingerprint['digest'])
        if touched:
            rerun.append({'test_id': spec.test_id, 'digest': fingerprint['digest'],
                          'reason': 'changed dependency ' + touched[0]})
        elif receipt is None:
            rerun.append({'test_id': spec.test_id, 'digest': fingerprint['digest'],
                          'reason': 'no receipt for this fingerprint'})
        elif receipt.get('status') != 'PASS':
            rerun.append({'test_id': spec.test_id, 'digest': fingerprint['digest'],
                          'reason': 'previous status ' + str(receipt.get('status'))})
        else:
            reuse.append({'test_id': spec.test_id, 'digest': fingerprint['digest'],
                          'receipt': str(cache._path(spec.test_id, fingerprint['digest']))})
    return {'schema': 'genrec.pipeline.plan/1', 'reuse': reuse, 'rerun': rerun,
            'changed': sorted(changed)}
