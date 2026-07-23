#!/usr/bin/env python3
"""
Task #73 (rerun) — Retrain SASRec on Musical_Instruments with hidden_size=32
to serve as LETTER cf_emb (paper LETTER default). Extract item embedding.

Usage: python3 task73_sasrec32_retrain.py <gpu_id>
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

# ray must be mocked BEFORE recbole import
import importlib
spec = importlib.util.find_spec('ray')
if spec is None:
    import types
    from unittest.mock import MagicMock
    ray = types.ModuleType('ray')
    ray.__version__ = '0.0.0'
    tune = types.ModuleType('tune')
    tune.run = MagicMock(return_value={})
    tune.analysis = types.ModuleType('analysis')
    ray.tune = tune
    ray.train = types.ModuleType('ray.train')
    ray.train.report = MagicMock()
    sys.modules['ray'] = ray
    sys.modules['ray.tune'] = tune
    sys.modules['ray.train'] = ray.train
else:
    # ray is installed but its logger breaks on numpy.bool8; patch VALID_NP_HPARAMS
    import numpy as np
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_  # numpy 1.24+ removed bool8 alias

# NumPy 2.0 removed aliases (np.float, np.bool, np.int, np.complex, np.object, np.str, np.long).
# RecBole 1.0.0 uses the old aliases. Patch them in before importing recbole.
import numpy as np
for _alias, _real in [
    ('float', 'float64'), ('bool', 'bool_'), ('int', 'int64'),
    ('complex', 'complex128'), ('object', 'object_'), ('str', 'str_'), ('long', 'int64'),
]:
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(np, _real))

import torch
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

# Now safe to import recbole
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='1')
args_cli = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu_id

from recbole.quick_start import run

# Run with hidden_size=32 override (paper LETTER default)
run(
    model='SASRec',
    dataset='Musical_Instruments',
    config_file_list=['/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper.yaml'],
    config_dict={
        'gpu_id': 0,           # after CUDA_VISIBLE_DEVICES, device 0 is the actual GPU
        'hidden_size': 32,    # paper LETTER cf_emb dim
        'inner_size': 64,     # proportionally smaller
        'epochs': 30,         # SASRec converges fast on this small dataset
        'stopping_step': 5,   # stop if no improvement for 5 evals
        'train_batch_size': 1024,
        'eval_step': 2,
    }
)

# After training, extract item embedding
print("=" * 60)
print("SASRec 32-d training complete. Extracting item embedding...")
print("=" * 60)

import glob
ckpts = sorted(glob.glob('/home/wlia0047/ar57/wenyu/GeneRec/RecBole/saved/SASRec*.pth'))
assert ckpts, "No SASRec ckpt found"
latest = ckpts[-1]
print(f"Latest ckpt: {latest}")

ckpt = torch.load(latest, map_location='cpu', weights_only=False)
sd = ckpt.get('state_dict', ckpt)
item_emb = None
for k, v in sd.items():
    if 'item_embedding' in k and v.ndim == 2 and v.shape[1] == 32:
        item_emb = v.detach().clone()
        print(f"Found: {k} shape={tuple(item_emb.shape)}")
        break

if item_emb is None:
    print("ERROR: 32-d item embedding not found in state_dict")
    sys.exit(1)

# L2 normalize each row for stable cf_loss
norms = item_emb.norm(dim=-1, keepdim=True)
norms = torch.clamp(norms, min=1e-8)
item_emb_normed = item_emb / norms

out_path = '/home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/RQ-VAE/ckpt/Instruments-sasrec32-v2.pt'
torch.save(item_emb_normed, out_path)
print(f"\nSaved L2-normalized 32-d SASRec embedding: {out_path}")
print(f"  shape={tuple(item_emb_normed.shape)}")
print(f"  norm range: [{item_emb_normed.norm(dim=-1).min():.4f}, {item_emb_normed.norm(dim=-1).max():.4f}]")
print(f"  mean={item_emb_normed.mean():.6f}, std={item_emb_normed.std():.4f}")