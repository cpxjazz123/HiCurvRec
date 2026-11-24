"""Launcher-owned independent unit cases; never formal/DDP training evidence."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tools/research/stage2/contract.py').is_file())
sys.path.insert(0,str(ROOT))
from tools.issue256.r02.run_tests import child

if __name__=='__main__':
    raise SystemExit(child())
