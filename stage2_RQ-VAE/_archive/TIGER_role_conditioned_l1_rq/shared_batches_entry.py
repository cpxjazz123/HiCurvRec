"""One actual epoch then fixed-state batch arithmetic discrimination only."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.r03.diagnose_shared_batches import child

if __name__ == '__main__':
    child()
