"""Fresh static-cache/component discriminator; never a formal training entry."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
from tools.issue256.r02.diagnose_static_consumers import child

if __name__ == '__main__':
    child()
