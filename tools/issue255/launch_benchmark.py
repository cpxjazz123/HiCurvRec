"""Source-fixed issue255 60-epoch inclusive benchmark; the sanctioned launcher."""
import os
from pathlib import Path

from tools.research.stage2.launcher import launch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY = ROOT / 'stage2_RQ-VAE/_archive/TIGER_prefix_curvature_gate_rq/benchmark_entry.py'
OUTPUT = ROOT / 'results/research/issue255/benchmark'


def main():
    return launch(ENTRY, OUTPUT / ('launcher_' + str(os.getpid())), python=PYTHON,
                  log_name='benchmark.log')


if __name__ == '__main__':
    raise SystemExit(main())
