"""Dependency-aware validation with machine-checked evidence.

A unit is reusable only when (a) the fingerprint of its real inputs is identical,
(b) the stored receipt passes the evidence checks, and (c) no deleted/renamed
input from the previous closure needs a one-time re-evaluation.

Import closure covers absolute imports, multi-level relative imports, package
``__init__`` chains and literal ``importlib.import_module``/``__import__`` calls.
Non-literal dynamic imports are marked unresolved, which widens validation
instead of silently skipping them.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STANDARD_ROOTS = (REPO / 'tools', REPO / 'stage2_RQ-VAE')
RUNTIME_KEYS = ('interpreter', 'implementation', 'python', 'torch', 'cuda', 'devices', 'platform')
SCHEMA = 'genrec.pipeline.receipt/1'
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.git', '.mypy_cache', '.ruff_cache'}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def _runtime_identity():
    identity = {'interpreter': sys.executable,
                'implementation': platform.python_implementation(),
                'python': '%d.%d' % sys.version_info[:2],
                'platform': platform.platform()}
    try:
        import torch
        identity['torch'] = torch.__version__
        identity['cuda'] = torch.version.cuda
        identity['devices'] = int(torch.cuda.device_count())
    except Exception:
        identity.update({'torch': None, 'cuda': None, 'devices': 0})
    return identity


def runtime_identity():
    """Runtime identity; scoped per unit through ``Spec(runtime_keys=...)``."""
    return dict(_runtime_identity())


def _key(path, repo):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _expand(paths, repo):
    files, missing = [], []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = Path(repo) / path
        path = path.resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_file() and not (SKIP_PARTS & set(child.relative_to(path).parts)):
                    files.append(child.resolve())
        else:
            missing.append(path)
    return files, missing


def _bases(repo, roots):
    out = []
    for base in (Path(repo), *[Path(r) for r in roots], *STANDARD_ROOTS, REPO):
        base = base.resolve()
        if base not in out:
            out.append(base)
    return out


def _resolve_module(name, repo, roots=()):
    parts = [part for part in str(name).split('.') if part]
    if not parts:
        return None
    for base in _bases(repo, roots):
        target = base.joinpath(*parts)
        for candidate in (target.with_suffix('.py'), target / '__init__.py'):
            if candidate.is_file():
                return candidate.resolve(), base
    return None


def _base_for(path, repo, roots=()):
    path = Path(path).resolve()
    best = None
    for base in _bases(repo, roots):
        try:
            path.relative_to(base)
        except ValueError:
            continue
        if best is None or len(base.parts) > len(best.parts):
            best = base
    return best


def _package_parts(path, base):
    rel = Path(path).resolve().relative_to(base)
    parts = list(rel.with_suffix('').parts)
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return parts


def _package_chain(path, base):
    path = Path(path).resolve()
    chain = []
    directory = path.parent
    base = Path(base).resolve()
    try:
        directory.relative_to(base)
    except ValueError:
        return chain
    while True:
        init = directory / '__init__.py'
        if init.is_file():
            chain.append(init.resolve())
        if directory == base:
            break
        directory = directory.parent
    return chain


def _add_under(base, parts, files):
    target = Path(base).joinpath(*parts)
    for candidate in (target.with_suffix('.py'), target / '__init__.py'):
        if candidate.is_file():
            resolved = candidate.resolve()
            files.add(resolved)
            files.update(_package_chain(resolved, base))
            return resolved
    return None


def _imports(path, repo, roots=()):
    files, unresolved = set(), []
    try:
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeDecodeError):
        return files, unresolved
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _resolve_module(alias.name, repo, roots)
                if hit:
                    resolved, base = hit
                    files.add(resolved)
                    files.update(_package_chain(resolved, base))
        elif isinstance(node, ast.ImportFrom):
            base = _base_for(path, repo, roots)
            if base is None:
                unresolved.append('%s:%d:relative import outside repository' % (path, node.lineno))
                continue
            if node.level:
                parts = _package_parts(path, base)
                package = parts if Path(path).name == '__init__.py' else parts[:-1]
                up = int(node.level) - 1
                if up > len(package):
                    unresolved.append('%s:%d:relative import above root' % (path, node.lineno))
                    continue
                prefix = package[:len(package) - up] if up else package
                module = [part for part in (node.module or '').split('.') if part]
                target = prefix + module
                _add_under(base, target, files)
                for alias in node.names:
                    if alias.name != '*':
                        _add_under(base, target + [alias.name], files)
            else:
                if node.module:
                    hit = _resolve_module(node.module, repo, roots)
                    if hit:
                        resolved, base = hit
                        files.add(resolved)
                        files.update(_package_chain(resolved, base))
                    for alias in node.names:
                        if alias.name != '*':
                            hit = _resolve_module(node.module + '.' + alias.name, repo, roots)
                            if hit:
                                resolved, base = hit
                                files.add(resolved)
                                files.update(_package_chain(resolved, base))
        elif isinstance(node, ast.Call):
            name = ''
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name not in ('import_module', '__import__'):
                continue
            literal = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                literal = node.args[0].value
            else:
                for keyword in node.keywords:
                    if keyword.arg == 'name' and isinstance(keyword.value, ast.Constant) \
                            and isinstance(keyword.value.value, str):
                        literal = keyword.value.value
            if literal is None:
                unresolved.append('%s:%d:dynamic import' % (path, node.lineno))
            else:
                hit = _resolve_module(literal, repo, roots)
                if hit:
                    resolved, base = hit
                    files.add(resolved)
                    files.update(_package_chain(resolved, base))
    return files, unresolved


def closure_report(paths, repo=REPO, roots=()):
    """Resolve the transitive import closure of the seed paths."""
    repo = Path(repo).resolve()
    seeds, missing = _expand(paths, repo)
    files, unresolved = set(seeds), []
    queue = list(seeds)
    while queue:
        path = queue.pop()
        if path.suffix != '.py':
            continue
        found, problems = _imports(path, repo, roots)
        unresolved.extend(problems)
        for found_path in found:
            if found_path not in files:
                files.add(found_path)
                queue.append(found_path)
    return {'files': sorted(files),
            'unresolved': sorted(set(unresolved)),
            'missing': sorted(str(p) for p in missing)}


def module_closure(paths, repo=REPO, roots=()):
    return closure_report(paths, repo=repo, roots=roots)['files']


class Spec:
    """One validation unit: declared inputs + the import closure they reach."""

    def __init__(self, test_id, *, code=(), tests=(), assets=(), configs=(), extra=(),
                 runtime=False, runtime_keys=None, roots=(), cacheable=True):
        self.test_id = str(test_id)
        self.code = [Path(p) for p in code]
        self.tests = [Path(p) for p in tests]
        self.assets = [Path(p) for p in assets]
        self.configs = [Path(p) for p in configs]
        self.extra = [Path(p) for p in extra]
        self.runtime = bool(runtime)
        self.runtime_keys = tuple(runtime_keys) if runtime_keys else RUNTIME_KEYS
        self.roots = tuple(Path(p) for p in roots)
        self.cacheable = bool(cacheable)

    def declared(self):
        return {name: [str(Path(p).resolve()) for p in getattr(self, name)]
                for name in ('code', 'tests', 'assets', 'configs', 'extra')}

    def fingerprint(self, repo=REPO):
        repo = Path(repo).resolve()
        closure = closure_report(self.code + self.tests, repo=repo, roots=self.roots)
        maps, missing = {}, list(closure['missing'])
        for name in ('code', 'tests', 'assets', 'configs', 'extra'):
            files, absent = _expand(getattr(self, name), repo)
            missing.extend(str(p) for p in absent)
            maps[name] = {_key(p, repo): file_sha(p) for p in sorted(files)}
        closure_files, absent = _expand(closure['files'], repo)
        missing.extend(str(p) for p in absent)
        maps['dependencies'] = {_key(p, repo): file_sha(p) for p in sorted(closure_files)}
        runtime = {}
        if self.runtime:
            runtime = {key: runtime_identity().get(key) for key in self.runtime_keys}
        payload = {'schema': 'genrec.pipeline.fingerprint/2', 'test_id': self.test_id,
                   'code': maps['code'], 'tests': maps['tests'], 'assets': maps['assets'],
                   'configs': maps['configs'], 'extra': maps['extra'],
                   'dependencies': maps['dependencies'], 'runtime': runtime,
                   'runtime_keys': list(self.runtime_keys) if self.runtime else [],
                   'roots': sorted(str(p.resolve()) for p in self.roots),
                   'unresolved': list(closure['unresolved'])}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
        return {'schema': 'genrec.pipeline.fingerprint/2', 'test_id': self.test_id,
                'digest': digest, 'closure': sorted(maps['dependencies']),
                'dependencies': maps['dependencies'], 'code': maps['code'],
                'tests': maps['tests'], 'assets': maps['assets'], 'configs': maps['configs'],
                'extra': maps['extra'], 'runtime': runtime,
                'unresolved': payload['unresolved'], 'missing': sorted(set(missing)),
                'declared': self.declared()}


def validate_receipt(receipt, *, digest=None, repo=REPO):
    """Return None when reusable, else the reason it is not."""
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA:
        return 'receipt schema mismatch'
    if digest is not None and receipt.get('digest') != digest:
        return 'receipt digest mismatch'
    if receipt.get('status') != 'PASS':
        return 'previous status %s' % receipt.get('status')
    fingerprint = receipt.get('fingerprint')
    if not isinstance(fingerprint, dict) or not fingerprint.get('digest'):
        return 'missing fingerprint'
    if fingerprint['digest'] != receipt.get('digest'):
        return 'fingerprint digest mismatch'
    evidence = receipt.get('evidence')
    if not isinstance(evidence, dict):
        return 'missing evidence'
    command = evidence.get('command')
    if not command:
        return 'evidence command missing'
    if evidence.get('exit_code') != 0:
        return 'evidence exit code %r' % evidence.get('exit_code')
    log = evidence.get('log')
    if not log:
        return 'evidence log missing'
    log_path = Path(log)
    if not log_path.is_absolute():
        log_path = Path(repo) / log_path
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return 'evidence log missing or empty'
    output = evidence.get('output')
    output_sha = evidence.get('output_sha256')
    if not output or not output_sha:
        return 'evidence output sha256 missing'
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path(repo) / output_path
    if not output_path.is_file():
        return 'evidence output missing'
    if file_sha(output_path) != output_sha:
        return 'evidence output sha mismatch'
    return None


class Cache:
    def __init__(self, root):
        self.root = Path(root)

    def _dir(self, test_id):
        safe = str(test_id).replace('/', '__').replace('..', '__')
        return self.root / safe

    def record(self, test_id, digest, status, *, evidence=None, declared=None,
               fingerprint=None, generation=None):
        directory = self._dir(test_id) / str(digest)
        directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        generation = generation or ('%s-%d' % (now.strftime('%Y%m%dT%H%M%S%fZ'), os.getpid()))
        payload = {'schema': SCHEMA, 'test_id': str(test_id), 'digest': str(digest),
                   'status': str(status), 'generation': str(generation),
                   'created_at': now.isoformat(), 'evidence': evidence,
                   'declared': declared, 'fingerprint': fingerprint}
        path = directory / (str(generation) + '.json')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        tmp.replace(path)
        return path

    def receipts(self, test_id):
        directory = self._dir(test_id)
        if not directory.is_dir():
            return []
        out = []
        for path in directory.glob('*/*.json'):
            try:
                out.append(json.loads(path.read_text(encoding='utf-8')))
            except (OSError, ValueError):
                continue
        for path in directory.glob('*.json'):
            try:
                out.append(json.loads(path.read_text(encoding='utf-8')))
            except (OSError, ValueError):
                continue
        out.sort(key=lambda row: (str(row.get('created_at', '')), str(row.get('generation', ''))), reverse=True)
        return out

    def lookup(self, test_id, digest=None):
        for receipt in self.receipts(test_id):
            if digest is None or receipt.get('digest') == digest:
                return receipt
        return None

    def lookup_valid(self, test_id, digest, *, repo=REPO):
        receipt = self.lookup(test_id, digest)
        if receipt is None:
            return None, 'no receipt for this fingerprint'
        reason = validate_receipt(receipt, digest=digest, repo=repo)
        if reason is not None:
            return None, reason
        return receipt, None


def _previous_inputs(receipt, repo):
    fingerprint = receipt.get('fingerprint') or {}
    keys = set(fingerprint.get('dependencies') or {})
    for name in ('code', 'tests', 'assets', 'configs', 'extra'):
        keys.update(fingerprint.get(name) or {})
    declared = receipt.get('declared') or {}
    for name in ('code', 'tests', 'assets', 'configs', 'extra'):
        keys.update(_key(path, repo) for path in (declared.get(name) or []))
    return keys


def plan(specs, *, cache, changed=(), repo=REPO):
    repo = Path(repo).resolve()
    changed_keys = {_key(p, repo) for p in changed}
    run, reuse = [], []
    for spec in specs:
        fp = spec.fingerprint(repo=repo)
        reasons = []
        if not spec.cacheable:
            reasons.append('mandatory gate')
        if fp['unresolved']:
            reasons.append('unresolved dependency: ' + '; '.join(fp['unresolved']))
        if fp['missing']:
            reasons.append('missing declared input: ' + '; '.join(fp['missing']))
        receipt, reason = (None, None)
        if not reasons:
            receipt, reason = cache.lookup_valid(spec.test_id, fp['digest'], repo=repo)
        if receipt is not None:
            reuse.append({'test_id': spec.test_id, 'digest': fp['digest'],
                          'receipt': str(cache._dir(spec.test_id) / fp['digest'])})
            continue
        if reason:
            reasons.append(reason)
        latest = cache.receipts(spec.test_id)
        if latest:
            previous = _previous_inputs(latest[0], repo)
            hit = sorted(previous & changed_keys)
            if hit:
                reasons.append('changed dependency (previous closure): ' + ', '.join(hit))
        run.append({'test_id': spec.test_id, 'digest': fp['digest'],
                    'reason': '; '.join(reasons) or 'no receipt for this fingerprint',
                    'declared': spec.declared(), 'unresolved': fp['unresolved'],
                    'missing': fp['missing']})
    return {'schema': 'genrec.pipeline.plan/2', 'run': run, 'rerun': run, 'reuse': reuse,
            'changed': sorted(changed_keys)}
