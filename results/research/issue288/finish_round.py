"""Issue #288: wait for the newest round-<TAG> run to finish, then export the checkpoint SID,
evaluate it with the frozen #277 evaluator and write the round record.

Usage: python3.10 results/research/issue288/finish_round.py <TAG> <ARM>
The export reuses the existing checkpoint-bound export interface (issue287/export_eval287.py); the
record is written by record_round.py.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path('.').resolve()
PY = '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10'


def newest_run(arm: str) -> Path:
    candidates = sorted((REPO / 'results/stage2_RQ-VAE/_runs').glob(f'*/{arm}'),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f'no run directory found for {arm}')
    return candidates[-1]


def wait_for_end(run: Path, limit_s: int = 5400) -> bool:
    metrics = run / 'result' / 'logs' / 'training_metrics.jsonl'
    deadline = time.time() + limit_s
    while time.time() < deadline:
        if metrics.is_file():
            for line in open(metrics):
                if '"train_end"' in line:
                    return True
        time.sleep(20)
    return False


def main() -> None:
    tag, arm = sys.argv[1], sys.argv[2]
    run = newest_run(arm)
    print('[finish] run', run, flush=True)
    if not wait_for_end(run):
        raise SystemExit('the run did not finish in time')
    print('[finish] the run finished; exporting the checkpoint SID', flush=True)
    subprocess.run([PY, 'results/research/issue287/export_eval287.py', str(run)], check=True)
    subprocess.run([PY, 'results/research/issue288/record_round.py', tag, str(run),
                    str(run / 'result' / 'item_sids.json')], check=True)
    print('[finish] done', flush=True)


if __name__ == '__main__':
    main()
