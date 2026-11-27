"""Framework adapter for the frozen-source four-rank validation diagnostic."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.layer_validation_entry import main

if __name__ == '__main__':
    main()
