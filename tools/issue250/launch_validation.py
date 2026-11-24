"""Launch only Issue250's four-GPU controlled microstep validation."""
from pathlib import Path
import sys


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'stage2_RQ-VAE').is_dir():
            return parent
    raise RuntimeError('missing repository marker')


ROOT = repository_root()
STAGE2_PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY = ROOT / 'stage2_RQ-VAE/TIGER_gnn_gate_sectional_rq/validation_entry.py'
RESULT_DIR = ROOT / 'results/research/issue250/runtime'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.research.stage2.launcher import launch


def main():
    return launch(ENTRY, RESULT_DIR, python=STAGE2_PYTHON,
                  log_name='controlled_validation.log')


if __name__ == '__main__':
    raise SystemExit(main())
