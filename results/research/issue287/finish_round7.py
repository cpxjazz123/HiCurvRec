"""Issue #287 round 7: wait for the run to finish, then export the checkpoint SID and evaluate it."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path('.').resolve()
PY = '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10'


def newest_run() -> Path:
    candidates = sorted((REPO / 'results/stage2_RQ-VAE/_runs').glob('*/TIGER_bucketsize_pairrank_rq'),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit('no round-7 run directory found')
    return candidates[-1]


def wait_for_end(run: Path, limit_s: int = 3600) -> bool:
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
    run = newest_run()
    print('[finish] run', run, flush=True)
    if not wait_for_end(run):
        raise SystemExit('the round-7 run did not finish in time')
    print('[finish] the run finished; exporting the checkpoint SID', flush=True)
    subprocess.run([PY, 'results/research/issue287/export_eval287.py', str(run)], check=True)
    subprocess.run([PY, 'results/research/issue287/eval287.py', str(run / 'result' / 'item_sids.json')], check=True)
    subprocess.run([PY, 'results/research/issue287/record_round.py', 'round7', str(run),
                    str(run / 'result' / 'item_sids.json')], check=True)
    print('[finish] done', flush=True)


if __name__ == '__main__':
    main()
