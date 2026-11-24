"""Launch the isolated CUDA CSR-adjoint regression through the frozen launcher."""
import os
from pathlib import Path
from tools.research.stage2.launcher import launch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY = ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/sparse_validation_entry.py'
OUTPUT = ROOT / 'results/research/issue252/sparse_validation'


def main():
    return launch(ENTRY, OUTPUT / ('launcher_' + str(os.getpid())), python=PYTHON,
                  log_name='sparse_validation.log')


if __name__ == '__main__':
    raise SystemExit(main())
