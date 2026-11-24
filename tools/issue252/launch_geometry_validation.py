"""Source-fixed four-rank geometry diagnostic; no trainer or optimizer."""
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'stage2_RQ-VAE').is_dir()
            and (p / 'tools/research/stage2/contract.py').is_file())
# The launcher runs children from a result directory, not from the repository.
sys.path.insert(0, str(ROOT))
from tools.research.stage2.launcher import launch

PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY = ROOT / 'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq/geometry_validation_entry.py'
OUTPUT = ROOT / 'results/research/issue252/geometry_validation'
WORLD_SIZE = 4
PORT = 50202


def main():
    run_id = 'geometry_%d_%d_%s' % (time.time_ns(), os.getpid(), uuid.uuid4().hex[:12])
    output = OUTPUT / run_id
    print('geometry_validation result_dir=' + str(output), flush=True)
    return launch(ENTRY, output, python=PYTHON, world_size=WORLD_SIZE, port=PORT,
                  env_extra={'TMPDIR': str(output), 'TEMP': str(output),
                             'TMP': str(output), 'PYTHONDONTWRITEBYTECODE': '1'},
                  log_name='geometry_validation.log')


if __name__ == '__main__':
    raise SystemExit(main())
