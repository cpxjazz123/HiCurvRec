"""Live candidate sources must freeze without owning the generated path adapter."""
from pathlib import Path
import hashlib
import shutil

from tools.issue252.run_workspace import ENTRY_NAME, freeze_run, load_run

TREE = Path(__file__).resolve().parents[1]


def test_live_source_freezes_with_one_framework_owned_execution_entry(tmp_path):
    repo = tmp_path/'repo'
    candidate = repo/'stage2_RQ-VAE'/TREE.name
    candidate.mkdir(parents=True)
    # Preserve the actual producer and any colliding adapter, not synthetic code text.
    for path in (TREE/'train_rqvae.py', TREE/ENTRY_NAME):
        if path.is_file():
            shutil.copyfile(path, candidate/path.name)
    shutil.copytree(TREE/'model', candidate/'model', ignore=shutil.ignore_patterns('__pycache__'))
    run = freeze_run(repo, candidate, 'formal_entry_regression', TREE.name)
    stored = load_run(repo, 'formal_entry_regression', TREE.name)
    expected = hashlib.sha256((TREE/'train_rqvae.py').read_bytes()).hexdigest()
    assert stored['admitted_sources']['train_rqvae.py'] == expected
    assert ENTRY_NAME not in stored['admitted_sources'], 'candidate must not own the generated adapter'
    assert ENTRY_NAME in stored['snapshot_sources'], 'framework must generate exactly its immutable adapter'
    assert Path(run['entry']).is_file() and Path(run['entry']).parent == Path(run['source'])
