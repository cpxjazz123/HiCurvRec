"""Source-bound formal Stage2 workspaces; GPU dispatch remains launcher.launch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from tools.issue252.archive_audit import sha256, source_hashes

SCHEMA = 'genrec.stage2.run_workspace/1'
SAFE_COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z')
ENTRY_NAME = 'execution_entry.py'

# Deliberately path-only: neither the trainer recipe nor its data is rewritten.
ENTRY = '''"""Framework path adapter for one immutable formal Stage2 generation."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
REPO = next(p for p in SOURCE.parents
            if (p / "tools/research/stage2/contract.py").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if sys.path[0] != str(SOURCE):
    sys.path.insert(0, str(SOURCE))
from tools.issue252.run_workspace import load_run
from tools.research.stage2 import launcher

run = load_run(REPO, {run_id!r}, {arm!r})
RESULT = Path(run["result"])
if __name__ == "__main__":
    if not launcher.is_child_rank():
        raise SystemExit(launcher.launch(Path(__file__), RESULT, python=sys.executable))
    import train_rqvae as t
    t.RUN_ID = run["run_id"]
    t.RESULT_DIR = RESULT
    t.LOG_DIR = RESULT / "logs"
    t.METRICS_PATH = t.LOG_DIR / "training_metrics.jsonl"
    t.OUTPUT_SID = RESULT / "sids_for_hgrec.npy"
    t.OUTPUT_JSON = RESULT / "item_sids.json"
    t.CHECKPOINT = RESULT / "rqvae_best.pth"
    t.main()
'''


def _component(value, label):
    if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value):
        raise ValueError('invalid ' + label + ': expected a single safe path component')
    return value


def _paths(repo, run_id, arm):
    repo = Path(repo).resolve()
    _component(run_id, 'run_id')
    _component(arm, 'arm')
    runs = repo / 'results/stage2_RQ-VAE/_runs'
    directory = runs / run_id / arm
    # Resolve existing ancestors too: a symlink cannot redirect a safe component.
    if any(path.resolve() != path for path in
           (directory, directory / 'src', directory / 'result', directory / 'workspace.json')):
        raise ValueError('run workspace contains a redirected path')
    source, result = directory / 'src', directory / 'result'
    logs = result / 'logs'
    return {
        'schema': SCHEMA, 'run_id': run_id, 'arm': arm,
        'source': str(source), 'result': str(result),
        'entry': str(source / ENTRY_NAME), 'workspace': str(directory / 'workspace.json'),
        'logs': {'run_log': str(logs / 'train_run.log'),
                 'native_log': str(logs / 'train_migrated.log'),
                 'metrics': str(logs / 'training_metrics.jsonl')},
    }


def freeze_run(repo, tree, run_id, arm, *, source_digest=None,
               graph_identity=None, checks_fingerprint=None):
    """Reserve an unused run ID, copy its exact admitted Python/JSON inventory.

    Failure never deletes an existing or partially captured generation. The caller
    must obtain machine admission before this function and recheck its generation
    before spawning the returned entry. Issue252 additionally requires its explicit
    training receipt before reserving any run directory.
    """
    repo, tree = Path(repo).resolve(), Path(tree).resolve()
    run = _paths(repo, run_id, arm)
    if tree.parent != repo / 'stage2_RQ-VAE':
        raise ValueError('formal source must be a direct Stage2 candidate tree')
    from tools.issue252.training_admission import check_receipt, required
    credential = check_receipt(repo, tree) if required(tree) else None
    admitted = source_hashes(tree)
    if ENTRY_NAME in admitted:
        raise ValueError('candidate source collides with generated execution entry')
    if source_digest is not None:
        from tools.research.verification.admission import source_digest as digest_of
        if digest_of(tree) != source_digest:
            raise ValueError('admitted source digest changed before snapshot')
    source, result = Path(run['source']), Path(run['result'])
    # Reserve the ID itself, not only its arm subdirectory: it identifies one run.
    reservation = source.parent.parent
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation.mkdir(exist_ok=False)
    source.mkdir(parents=True)
    result.mkdir()
    for name, expected in admitted.items():
        origin, destination = tree / name, source / name
        if origin.is_symlink() or origin.resolve() != origin:
            raise ValueError('source symlink cannot be frozen: ' + name)
        data = origin.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('admitted source changed while copying: ' + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as sink:
            sink.write(data)
        if sha256(destination) != expected:
            raise ValueError('snapshot copy bytes differ: ' + name)
    if source_hashes(tree) != admitted or source_hashes(source) != admitted:
        raise ValueError('source inventory changed while freezing the run')
    if source_digest is not None and digest_of(tree) != source_digest:
        raise ValueError('admitted source digest changed during snapshot')
    Path(run['entry']).write_text(ENTRY.format(run_id=run_id, arm=arm), encoding='utf-8')
    run.update(admitted_tree=str(tree), admitted_sources=admitted,
               source_digest=source_digest, graph_identity=graph_identity,
               checks_fingerprint=checks_fingerprint, training_admission=credential,
               snapshot_sources=source_hashes(source))
    Path(run['workspace']).write_text(json.dumps(run, indent=2, sort_keys=True) + '\n',
                                      encoding='utf-8')
    return run


def check_run_admission(repo, run):
    """Recheck the exact stored formal credential immediately before dispatch."""
    from tools.issue252.training_admission import check_receipt, required
    tree = Path(run['admitted_tree'])
    if required(tree):
        credential = check_receipt(repo, tree)
        if run.get('training_admission') != credential:
            raise ValueError('training admission credential changed after snapshot')
        return credential
    return None


def load_run(repo, run_id, arm, *, verify_source=True):
    """Read one exact generation and fail closed on redirected/mutated sources."""
    expected = _paths(repo, run_id, arm)
    run = json.loads(Path(expected['workspace']).read_text(encoding='utf-8'))
    if not isinstance(run, dict) or any(run.get(k) != v for k, v in expected.items()):
        raise ValueError('persisted run workspace identity mismatch')
    source = Path(run['source'])
    if verify_source and run.get('snapshot_sources') != source_hashes(source):
        raise ValueError('immutable source generation changed after admission')
    return run


def run_from_sid(repo, sid):
    """Return the exact workspace for a nested SID, or None for explicit history."""
    repo = Path(repo).resolve()
    sid = Path(sid).resolve()
    relative = sid.relative_to(repo / 'results/stage2_RQ-VAE')
    if not relative.parts or relative.parts[0] != '_runs':
        return None
    if len(relative.parts) != 5 or relative.parts[3:] != ('result', 'item_sids.json'):
        raise ValueError('invalid immutable Stage2 SID path')
    run = load_run(repo, relative.parts[1], relative.parts[2])
    if sid != Path(run['result']) / 'item_sids.json':
        raise ValueError('SID differs from its immutable run result')
    return run


def persisted_run(repo, state, *, verify_source=True):
    """Resolve state fields as one generation, never a latest/canonical fallback."""
    if not state.get('run_id'):
        return None
    run = load_run(repo, state['run_id'], state['arm'], verify_source=verify_source)
    for field, key in (('source_tree', 'source'), ('entrypoint', 'entry'),
                       ('result_dir', 'result')):
        if state.get(field) != run[key]:
            raise ValueError('persisted ' + field + ' crosses immutable run generations')
    if state.get('logs') != run['logs']:
        raise ValueError('persisted logs cross immutable run generations')
    return run
