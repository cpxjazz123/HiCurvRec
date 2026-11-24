"""Compile unchanged pure geometry; generated code is not a learned cache."""
import os
from pathlib import Path
import tempfile

import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'tools/research/stage2/contract.py').is_file())
CACHE = ROOT / 'results/research/issue252/compiler_cache'


def compile_geometry(experts, mixed, fixed):
    # Source-owned cache roots, not a user recipe/path override. No system temp
    # or CUDA-graph output reuse; compile failures propagate to admission.
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
    # Up to48 layout/curvature/autograd/alias/singleton specializations.
    # This bounds generated-code caching, not a model or optimizer parameter.
    dynamo.recompile_limit = 64
    import torch._functorch.config as autograd
    # Calibration and independent real-loss VJPs retain their actual graphs.
    # Donating saved buffers would violate that consumer-visible lifetime.
    autograd.donated_buffer = False
    return tuple(torch.compile(function, fullgraph=True, dynamic=True, mode='default')
                 for function in (experts, mixed, fixed))
