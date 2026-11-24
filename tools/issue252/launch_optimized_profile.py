"""Source-fixed optimized diagnostic; preserves original baseline reports."""
import os
from pathlib import Path
from tools.research.stage2.launcher import launch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY = ROOT / 'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq/optimized_profile_entry.py'
OUTPUT = ROOT / 'results/research/issue252/optimized_profile'


def main():
    return launch(ENTRY, OUTPUT / ('launcher_' + str(os.getpid())), python=PYTHON,
                  log_name='optimized_profile.log')


if __name__ == '__main__':
    raise SystemExit(main())
