"""Measure only the rank-side trainer import; no main, no CUDA, no data."""
from __future__ import annotations
import importlib.util, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
STAGE3=ROOT/'stage3_T5Train'
if str(STAGE3) not in sys.path: sys.path.insert(0,str(STAGE3))
start=time.perf_counter()
spec=importlib.util.spec_from_file_location('genrec_stage3_worker_import', STAGE3/'train_HG-Rec.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
print(f'WORKER_IMPORT_SECONDS={time.perf_counter()-start:.3f}')
print('CUDA_VISIBLE_DEVICES='+repr(os.environ.get('CUDA_VISIBLE_DEVICES')))
print('IMPORT_OK')
