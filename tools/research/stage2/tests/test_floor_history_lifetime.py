"""Installed quantizers own rotating history only for their object lifetime."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tools.research.stage2.anticollapse import patch_tree

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
BASELINE = ROOT/'stage2_RQ-VAE/baseline/TIGER_RQ-VAE'


def test_installed_quantizer_releases_history_without_changing_live_rotation():
    with tempfile.TemporaryDirectory(prefix='floor_lifetime_', dir=ROOT/'results/research') as workspace:
        tree = Path(workspace)/'TIGER_floor_lifetime_probe'
        shutil.copytree(BASELINE, tree,
            ignore=shutil.ignore_patterns('__pycache__', 'logs', 'dataset'))
        patch_tree.apply_floor(tree)
        code = '''
import gc
import weakref
import torch
import assignment_floor
from model.layers import VQLayer
cost = torch.tensor([[0., 1., 2., 3.], [1., 0., 3., 2.]])
first = VQLayer(4, 2, sk_epsilon=-1, sk_iters=-1)
second = VQLayer(4, 2, sk_epsilon=-1, sk_iters=-1)
assert first._indices(cost, False).tolist() == [0, 1], 'fresh owner must use cheap codewords'
assert first._indices(cost, False).tolist() == [2, 3], 'live owner must preserve rotating coverage'
assert second._indices(cost, False).tolist() == [0, 1], 'different live owners must not share history'
refs = [weakref.ref(first), weakref.ref(second)]
del first, second
gc.collect()
assert all(ref() is None for ref in refs), 'coverage history must not keep quantizers alive'
assert not assignment_floor._COUNTS, 'destroyed quantizer must release its rotating-coverage history'
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=tree,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
        assert patch_tree.apply_floor(tree)['changed'] is False
