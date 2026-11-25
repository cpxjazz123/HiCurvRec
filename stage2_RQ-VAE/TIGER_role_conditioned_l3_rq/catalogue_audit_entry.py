"""Framework-owned launch adapter for the read-only CUDA catalogue audit."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.r04.extract_catalogues import main

if __name__ == '__main__':
    raise SystemExit(main())
