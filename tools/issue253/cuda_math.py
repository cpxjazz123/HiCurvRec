"""Compile live three-expert geometry and dense operators; never cache learned values."""
import os
from pathlib import Path
import tempfile

import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'tools/research/stage2/contract.py').is_file())
CACHE = ROOT / 'results/research/issue253/compiler_cache'


def _configure():
    # Cache locations are source-owned and remain inside the repository.
    roots = {'TORCHINDUCTOR_CACHE_DIR': CACHE / 'inductor',
             'TRITON_CACHE_DIR': CACHE / 'triton',
             'TORCH_EXTENSIONS_DIR': CACHE / 'extensions',
             'TMPDIR': CACHE / 'temporary', 'TMP': CACHE / 'temporary',
             'TEMP': CACHE / 'temporary'}
    for name, path in roots.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(roots['TMPDIR'])
    import torch._inductor.config as config
    config.compile_threads = 1
    import torch._dynamo.config as dynamo
    # Keep the existing bounded shape/layout/curvature/autograd specializations.
    dynamo.recompile_limit = 64
    import torch._functorch.config as autograd
    # Calibration and independent real-loss VJPs retain their actual graphs.
    autograd.donated_buffer = False


def compile_geometry(experts, mixed, fixed):
    _configure()
    return tuple(torch.compile(function, fullgraph=True, dynamic=True, mode='default')
                 for function in (experts, mixed, fixed))


def compile_probabilities(function):
    _configure()
    return torch.compile(function, fullgraph=True, dynamic=True, mode='default')


def compile_dense(function):
    """Compile a live dense affine layer; input/parameter gradients stay native."""
    _configure()
    return torch.compile(function, fullgraph=True, dynamic=True, mode='default')
