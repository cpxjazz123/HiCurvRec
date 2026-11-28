"""Issue #287: wait for round 4 to finish, then export the checkpoint SID and evaluate it."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('.').resolve()
RUN = Path('results/stage2_RQ-VAE/_runs/stage2_d859278421c34aae85b97747a5b66d3c/TIGER_l1margin_pairrank_rq')
METRICS = RUN / 'result' / 'logs' / 'training_metrics.jsonl'
PY = '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10'


def wait_for_end(limit_s=3600):
    deadline = time.time() + limit_s
    while time.time() < deadline:
        if METRICS.is_file():
            for line in open(METRICS):
                if '"train_end"' in line:
                    return True
        time.sleep(20)
    return False


def main() -> None:
    if not wait_for_end():
        raise SystemExit('the round-4 run did not finish in time')
    print('[finish] the run finished; exporting the checkpoint SID', flush=True)
    subprocess.run([PY, 'results/research/issue287/export_eval287.py', str(RUN)], check=True)
    subprocess.run([PY, 'results/research/issue287/eval287.py', str(RUN / 'result' / 'item_sids.json')], check=True)
    subprocess.run([PY, 'results/research/issue287/summarize287.py', str(RUN),
                    str(RUN / 'result' / 'item_sids.json')], check=True)
    print('[finish] done', flush=True)


if __name__ == '__main__':
    main()
