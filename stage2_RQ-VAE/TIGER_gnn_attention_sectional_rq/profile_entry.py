"""Instrument one actual full-sized trainer step; never dispatch formal epochs."""
import time
BOOT_TIME=time.perf_counter()
import os
from pathlib import Path
import tempfile
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
RUNTIME_TEMP=ROOT/'results/research/issue252/runtime_temp'
RUNTIME_TEMP.mkdir(parents=True,exist_ok=True)
os.environ['TMPDIR']=str(RUNTIME_TEMP)
tempfile.tempdir=str(RUNTIME_TEMP)
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import train_rqvae as training
from tools.issue252.profiling import run


if __name__=='__main__':
    run(training,ROOT/'results/research/issue252/profiling_baseline',BOOT_TIME)
