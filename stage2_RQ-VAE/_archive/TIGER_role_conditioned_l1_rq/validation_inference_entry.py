"""Post-training framework adapter; not part of the immutable R02 training source.

The frozen R02 training snapshot was admitted and executed before this file
existed. It only drives the read-only recommendation validation of the final
Stage3 checkpoint; it cannot start, resume or alter any training.
"""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.r03.external_validation import main

if __name__ == '__main__':
    raise SystemExit(main())
