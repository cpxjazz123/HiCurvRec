"""CPU-only measurement of the Stage3 parent import cost.

This does not call main(), touch CUDA, or read recommendation test scores.
It reproduces the parent harness import: importlib loads train_HG-Rec.py, which
imports torch, data, model and transformers.
"""
from __future__ import annotations
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STAGE3 = ROOT / "stage3_T5Train"
SCRIPT = STAGE3 / "train_HG-Rec.py"
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))
start = time.perf_counter()
spec = importlib.util.spec_from_file_location("genrec_stage3_parent_import", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
elapsed = time.perf_counter() - start
print(f"PARENT_IMPORT_SECONDS={elapsed:.3f}")
print("CUDA_VISIBLE_DEVICES=" + repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("TORCH_CUDA_AVAILABLE=" + str(getattr(module.torch.cuda, "is_available", lambda: None)()))
print("IMPORT_OK")
