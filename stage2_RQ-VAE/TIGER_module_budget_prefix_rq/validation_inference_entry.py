"""Post-training framework adapter; not part of the immutable R01 training source."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.external_validation import main

if __name__ == '__main__':
    raise SystemExit(main())
