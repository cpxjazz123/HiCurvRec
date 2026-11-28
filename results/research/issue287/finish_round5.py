"""Issue #287 round 5: wait for the run to finish, then export the checkpoint SID and evaluate it."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path('.').resolve()
RUN = Path('results/stage2_RQ-VAE/_runs/stage2_70a5a9cc3d754cd48d45e08de055e6d1/TIGER_sameanchor_r5_pairrank_rq')
METRICS = RUN / 'result' / 'logs' / 'training_metrics.jsonl'
PY = '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10'


def wait_for_end(limit_s: int = 3600) -> bool:
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
        raise SystemExit('the round-5 run did not finish in time')
    print('[finish] the run finished; exporting the checkpoint SID', flush=True)
    subprocess.run([PY, 'results/research/issue287/export_eval287.py', str(RUN)], check=True)
    subprocess.run([PY, 'results/research/issue287/eval287.py', str(RUN / 'result' / 'item_sids.json')], check=True)
    subprocess.run([PY, 'results/research/issue287/record_round.py', 'round5', str(RUN),
                    str(RUN / 'result' / 'item_sids.json')], check=True)
    print('[finish] done', flush=True)


if __name__ == '__main__':
    main()
