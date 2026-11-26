"""Measure dependency staging and local import; no CUDA, no training."""
from __future__ import annotations
import os, subprocess, sys, sysconfig, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.stage3.startup_cache import prepare
cache_root = Path("/dev/shm/genrec-stage3-deps-issue264-v2")
site = Path(sysconfig.get_paths()["purelib"])
start = time.perf_counter()
target = prepare(site, cache_root)
cold = time.perf_counter() - start
start = time.perf_counter()
target2 = prepare(site, cache_root)
warm = time.perf_counter() - start
print(f"CACHE_PREPARE_COLD_SECONDS={cold:.3f}")
print(f"CACHE_PREPARE_WARM_SECONDS={warm:.3f}")
print(f"CACHE_TARGET={target}")
print(f"CACHE_REUSED={target == target2}")
env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""
def run(python_path, extra=()):
    cmd = [sys.executable, "-c", "import transformers; print('TRANSFORMERS_OK')", *extra]
    start = time.perf_counter()
    done = subprocess.run(cmd, env=python_path, capture_output=True, text=True, timeout=900)
    return time.perf_counter() - start, done.returncode, (done.stdout+done.stderr).strip()[-300:]
base = env.copy()
local = env.copy(); local["PYTHONPATH"] = str(target) + os.pathsep + local.get("PYTHONPATH", "")
for label, e in (("source", base), ("local", local), ("source_repeat", base), ("local_repeat", local)):
    dt, rc, out = run(e)
    print(f"{label}_IMPORT_SECONDS={dt:.3f} exit={rc} out={out!r}")
