"""Read-only final-weight global-gradient diagnostic; never a formal trainer."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.final_global_gradients import main

if __name__ == '__main__':
    raise SystemExit(main())
